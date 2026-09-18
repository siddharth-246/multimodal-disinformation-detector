import io
import re
import time
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("disinfo_detector")

# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB upload cap
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "is", "are", "was",
    "were", "and", "or", "but", "with", "by", "from", "as", "it", "this",
    "that", "has", "have", "had", "be", "been", "will", "would", "could",
    "after", "over", "into", "out", "up", "down", "than", "then", "not",
}

text_model = None
tokenizer = None
ai_image_detector = None

# Pretrained ViT-based classifier fine-tuned to distinguish AI-generated vs
# real photos. This is the actual detection signal now; ELA/Laplacian/EXIF
# below are kept only as weak supplementary flags, not the primary decision.
AI_IMAGE_DETECTOR_MODEL = "Organika/sdxl-detector"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_model, tokenizer, ai_image_detector
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        text_model.eval()
        logger.info("Text style model loaded successfully.")
    except Exception:
        # Don't crash the whole app on a cold-start model download failure;
        # degrade gracefully and report it through the style-analysis result instead.
        logger.exception("Failed to load text style model.")
        text_model, tokenizer = None, None

    try:
        from transformers import pipeline
        ai_image_detector = pipeline("image-classification", model=AI_IMAGE_DETECTOR_MODEL)
        logger.info("AI-image detector model loaded successfully.")
    except Exception:
        logger.exception("Failed to load AI-image detector model.")
        ai_image_detector = None

    yield


app = FastAPI(title="Multimodal Disinformation & Deepfake Detector", lifespan=lifespan)

# =========================================================
# 1. REAL-TIME NEWS VERIFICATION ENGINE (DuckDuckGo Search)
# =========================================================
TRUSTED_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "cnn.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "ndtv.com",
    "thehindu.com", "indianexpress.com", "timesofindia.indiatimes.com", "forbes.com",
]

HIGH_RISK_KEYWORDS = [
    "dead", "died", "passed away", "killed", "assassinated", "arrested",
    "resigned", "nuke", "war declared", "explosion", "attack", "hostage",
]


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _claim_overlap_score(claim: str, article_text: str) -> float:
    """
    Cheap lexical-overlap heuristic (Jaccard-style over non-stopword tokens)
    used as a proxy for whether a matched article actually corroborates the
    claim, rather than merely sharing a domain. This is NOT true semantic
    entailment -- it will not catch paraphrases and can still be fooled by
    keyword-stuffed unrelated articles -- but it is a meaningfully stronger
    signal than "a trusted domain appeared in the results at all."
    """
    claim_tokens = _tokenize(claim)
    article_tokens = _tokenize(article_text)
    if not claim_tokens or not article_tokens:
        return 0.0
    overlap = claim_tokens & article_tokens
    return len(overlap) / len(claim_tokens)


def verify_realtime_claim(query: str) -> dict:
    if not query or len(query.strip()) == 0:
        return {"realtime_risk": 0.0, "status": "No Text Provided", "matched_sources": []}

    query_clean = query.strip()
    words = query_clean.split()
    short_query = " ".join(words[:7])  # Truncate query to prevent rate limits

    matched_sources = []
    best_overlap = 0.0
    search_failed = False

    try:
        results = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=short_query, max_results=8))
        except Exception:
            logger.warning("DDGS text search failed, falling back to news search.", exc_info=True)
            time.sleep(0.5)
            with DDGS() as ddgs:
                results = list(ddgs.news(keywords=short_query, max_results=8))

        for result in results:
            url = (result.get("url") or result.get("href") or "").lower()
            title = result.get("title", "") or ""
            snippet = result.get("body", "") or result.get("excerpt", "") or ""

            if not any(domain in url for domain in TRUSTED_DOMAINS):
                continue

            overlap = _claim_overlap_score(query_clean, f"{title} {snippet}")
            best_overlap = max(best_overlap, overlap)
            matched_sources.append({"title": title, "url": url, "overlap": round(overlap, 2)})

    except Exception:
        logger.exception("Real-time verification search failed.")
        search_failed = True

    has_high_risk_keyword = any(kw in query_clean.lower() for kw in HIGH_RISK_KEYWORDS)

    if search_failed:
        return {
            "realtime_risk": 30.0,
            "status": "Search Engine Unavailable / Rate Limited",
            "matched_sources": [],
        }

    # Require the matched article to actually share substantial vocabulary
    # with the claim before treating it as corroboration. A trusted domain
    # merely appearing in results is not enough -- it may be an unrelated
    # article that happened to share a couple of keywords.
    OVERLAP_CONFIRM_THRESHOLD = 0.5

    if matched_sources and best_overlap >= OVERLAP_CONFIRM_THRESHOLD:
        realtime_risk = 5.0
        status = "Corroborated by Trusted Outlets"
    elif matched_sources:
        # Trusted domains showed up but don't clearly corroborate the specific claim.
        realtime_risk = 55.0
        status = "Related Coverage Found, Not Clearly Corroborated"
    elif has_high_risk_keyword:
        realtime_risk = 90.0
        status = "Unverified Critical Claim"
    else:
        realtime_risk = 25.0
        status = "Unconfirmed General Statement"

    return {
        "realtime_risk": float(realtime_risk),
        "status": status,
        "matched_sources": [{"title": s["title"], "url": s["url"]} for s in matched_sources],
    }


# =========================================================
# 2. TEXT STYLE DETECTION ENGINE (BERT Transformer)
# =========================================================
def analyze_text(text: str) -> dict:
    if not text or len(text.strip()) == 0:
        return {"style_risk": 0.0, "label": "No Text Provided"}

    if text_model is None or tokenizer is None:
        return {"style_risk": 0.0, "label": "Style Model Unavailable"}

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        fake_prob = probs[0][1].item()
    except Exception:
        logger.exception("Text style model inference failed.")
        return {"style_risk": 0.0, "label": "Style Model Error"}

    style_risk = round(fake_prob * 100, 2)

    # Short headlines are noisy for this small model; damp extreme scores
    # instead of hard-overriding them, so short text still influences the
    # score proportionally rather than being pinned to one fixed value.
    if len(text.split()) < 20 and style_risk > 80:
        style_risk = round(50.0 + (style_risk - 80.0) * 0.5, 2)
        label = "Sensationalist Style (Low Confidence, Short Text)"
    elif style_risk > 50:
        label = "Sensationalist Writing Style"
    else:
        label = "Standard Writing Style"

    return {"style_risk": style_risk, "label": label}


# =========================================================
# 3. IMAGE ANALYSIS: PRETRAINED AI-DETECTOR (primary) +
#    EXIF / ELA / LAPLACIAN (secondary, supporting flags only)
# =========================================================
def _run_ai_image_detector(original: Image.Image) -> dict:
    """
    Runs a pretrained image-classification model fine-tuned to distinguish
    AI-generated images from real photos. Returns ai_generated_prob in
    [0, 1] plus the raw label/score for debugging/calibration.
    This is the primary signal -- ELA/Laplacian are not reliable enough
    to use as the main decision (see analyze_image below).
    """
    if ai_image_detector is None:
        return {"available": False, "ai_generated_prob": None, "raw": None}

    try:
        results = ai_image_detector(original)  # list of {"label": ..., "score": ...}, sorted desc
        top = results[0]
        label = str(top["label"]).lower()
        score = float(top["score"])

        # Different checkpoints use different label names; handle the common
        # conventions defensively rather than assuming one exact string.
        if any(k in label for k in ["artificial", "fake", "ai", "generated", "synthetic"]):
            ai_generated_prob = score
        elif any(k in label for k in ["human", "real", "authentic"]):
            ai_generated_prob = 1.0 - score
        else:
            # Unknown label scheme -- log it so it can be mapped correctly,
            # and don't let it silently skew the score either direction.
            logger.warning("Unrecognized AI-detector label '%s'; ignoring this signal.", label)
            return {"available": False, "ai_generated_prob": None, "raw": results}

        return {"available": True, "ai_generated_prob": round(ai_generated_prob, 4), "raw": results}
    except Exception:
        logger.exception("AI-image detector inference failed.")
        return {"available": False, "ai_generated_prob": None, "raw": None}


def _run_supporting_forensics(original: Image.Image, raw_img: Image.Image) -> dict:
    """
    EXIF metadata check + Error Level Analysis + Laplacian sharpness.
    Kept only as weak, explainable *supporting* flags -- these are legacy
    Photoshop-splicing forensics techniques, not reliable AI-generation
    detectors, and should never carry more weight than the pretrained
    classifier above. Thresholds are heuristic and may need recalibration
    against your own sample images (raw values are returned for that).
    """
    exif_data = str(raw_img.info).lower()
    ai_metadata_detected = any(
        tag in exif_data for tag in ["midjourney", "dall-e", "stable diffusion", "civitai", "comfyui"]
    )

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    ela_im = ImageChops.difference(original, recompressed)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    ela_cv = np.array(ela_im)
    ela_variance = float(np.var(cv2.cvtColor(ela_cv, cv2.COLOR_RGB2GRAY)))

    gray = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    flags = []
    supporting_risk = 0.0
    if ai_metadata_detected:
        supporting_risk += 15.0
        flags.append("AI Generator Metadata Signature Found")
    if laplacian_var < 25.0:
        supporting_risk += 8.0
        flags.append("Low Detail / Smooth Texture")
    if ela_variance > 3500:
        supporting_risk += 8.0
        flags.append("High Compression / Edit Discrepancy Detected")

    return {
        "supporting_risk": supporting_risk,
        "flags": flags,
        "ela_variance": round(ela_variance, 2),
        "laplacian_var": round(laplacian_var, 2),
        "ai_metadata_detected": ai_metadata_detected,
    }


def analyze_image_ela(image_bytes: bytes) -> dict:
    try:
        probe = Image.open(io.BytesIO(image_bytes))
        probe.verify()  # confirm it's a genuine image before reopening
        raw_img = Image.open(io.BytesIO(image_bytes))
        original = raw_img.convert("RGB")
    except UnidentifiedImageError:
        return {"image_risk": 0.0, "status": "Image Error: unreadable or corrupt file", "ela_variance": 0.0}
    except Exception as e:
        logger.exception("Image parsing failed.")
        return {"image_risk": 0.0, "status": f"Image Error: {str(e)}", "ela_variance": 0.0}

    try:
        ai_result = _run_ai_image_detector(original)
        forensics = _run_supporting_forensics(original, raw_img)

        if ai_result["available"]:
            # Primary signal dominates the score (0-85 range); supporting
            # forensic flags can only nudge it, not drive it.
            image_risk = ai_result["ai_generated_prob"] * 85.0 + forensics["supporting_risk"]
            status_parts = [
                f"AI-Generated Probability: {ai_result['ai_generated_prob'] * 100:.1f}%"
            ] + forensics["flags"]
            status = " | ".join(status_parts) if status_parts else "No AI signature detected"
        else:
            # Detector unavailable (failed to load / bad label mapping) --
            # fall back to the weak supporting signals alone, and say so
            # explicitly rather than reporting false confidence.
            image_risk = forensics["supporting_risk"] + 5.0
            status_parts = ["AI Detector Unavailable -- score based on limited forensic signals only"]
            status_parts += forensics["flags"]
            status = " | ".join(status_parts)

        final_risk = min(95.0, max(0.0, image_risk))

        return {
            "image_risk": round(float(final_risk), 2),
            "status": status,
            "ela_variance": forensics["ela_variance"],
            "laplacian_var": forensics["laplacian_var"],
            "ai_generated_prob": ai_result["ai_generated_prob"],
        }
    except Exception as e:
        logger.exception("Image analysis failed.")
        return {"image_risk": 0.0, "status": f"Image Error: {str(e)}", "ela_variance": 0.0}


# =========================================================
# 4. WEB INTERFACE AND STRICT DYNAMIC ROUTING
# =========================================================
@app.get("/", response_class=HTMLResponse)
def serve_home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multimodal Disinformation Detector</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .card { background-color: #1e293b; border: 1px solid #334155; color: #f8fafc; border-radius: 12px; }
            .btn-primary { background-color: #3b82f6; border: none; font-weight: 600; padding: 12px; }
            .btn-primary:hover { background-color: #2563eb; }
            .metric-box { background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; height: 100%; }
            #errorBox { display: none; }
        </style>
    </head>
    <body class="py-5">
        <div class="container" style="max-width: 850px;">
            <div class="text-center mb-5">
                <h1 class="fw-bold text-primary">Multimodal Disinformation Detector</h1>
                <p class="text-secondary">Independent Real-Time News + NLP Style + Image Forensics Engine</p>
            </div>

            <div class="card p-4 shadow-lg mb-4">
                <form id="detectorForm">
                    <div class="mb-3">
                        <label for="text" class="form-label fw-bold">News Headline or Article Text</label>
                        <textarea class="form-control bg-dark text-light border-secondary" id="text" name="text" rows="3" placeholder="Enter headline or article text to verify..."></textarea>
                    </div>
                    <div class="mb-4">
                        <label for="image" class="form-label fw-bold">Associated Image (Optional for Forensics, max 8MB)</label>
                        <input class="form-control bg-dark text-light border-secondary" type="file" id="image" name="image" accept="image/png,image/jpeg,image/webp">
                    </div>
                    <button type="submit" class="btn btn-primary w-100" id="submitBtn">Analyze Content Credibility</button>
                </form>
            </div>

            <div id="errorBox" class="alert alert-danger"></div>

            <div id="results" class="card p-4 shadow-lg d-none">
                <h3 class="fw-bold mb-3 text-center">Analysis Telemetry</h3>

                <div class="text-center mb-4">
                    <h2 class="display-4 fw-bold" id="scoreDisplay">0%</h2>
                    <p class="fs-5 fw-bold" id="verdictDisplay"></p>
                </div>

                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="metric-box">
                            <h6 class="text-primary fw-bold">1. Live News Verification</h6>
                            <p class="mb-1"><strong>Risk:</strong> <span id="realtimeRisk">0%</span></p>
                            <p class="mb-0 text-secondary" id="realtimeStatus">-</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-box">
                            <h6 class="text-primary fw-bold">2. NLP Style Model</h6>
                            <p class="mb-1"><strong>Style Risk:</strong> <span id="styleRisk">0%</span></p>
                            <p class="mb-0 text-secondary" id="styleLabel">-</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-box">
                            <h6 class="text-primary fw-bold">3. Image Forensics</h6>
                            <p class="mb-1"><strong>Image Risk:</strong> <span id="imageRisk">0%</span></p>
                            <p class="mb-0 text-secondary" id="imageStatus">-</p>
                        </div>
                    </div>
                </div>

                <div class="mt-4">
                    <h6 class="fw-bold text-light">Cross-Referenced News Outlets Found:</h6>
                    <ul id="sourcesList" class="text-secondary ps-3 mb-0">
                        <li>None</li>
                    </ul>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('detectorForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const errorBox = document.getElementById('errorBox');
                const submitBtn = document.getElementById('submitBtn');
                errorBox.style.display = 'none';
                submitBtn.disabled = true;
                submitBtn.innerText = 'Analyzing...';

                try {
                    const formData = new FormData(e.target);
                    const response = await fetch('/analyze', { method: 'POST', body: formData });

                    if (!response.ok) {
                        const errText = await response.text();
                        throw new Error(`Server error (${response.status}): ${errText}`);
                    }

                    const data = await response.json();
                    document.getElementById('results').classList.remove('d-none');

                    const cred = data.credibility_score;
                    const scoreElem = document.getElementById('scoreDisplay');
                    const verdictElem = document.getElementById('verdictDisplay');

                    scoreElem.innerText = cred + "%";
                    if (cred >= 70) {
                        scoreElem.className = "display-4 fw-bold text-success";
                        verdictElem.innerText = "Likely Authentic Content";
                        verdictElem.className = "fs-5 fw-bold text-success";
                    } else if (cred >= 40) {
                        scoreElem.className = "display-4 fw-bold text-warning";
                        verdictElem.innerText = "Suspicious / Unverified Content";
                        verdictElem.className = "fs-5 fw-bold text-warning";
                    } else {
                        scoreElem.className = "display-4 fw-bold text-danger";
                        verdictElem.innerText = "High Probability Disinformation / Fake News";
                        verdictElem.className = "fs-5 fw-bold text-danger";
                    }

                    document.getElementById('realtimeRisk').innerText = data.telemetry.realtime_news.realtime_risk + "%";
                    document.getElementById('realtimeStatus').innerText = data.telemetry.realtime_news.status;

                    document.getElementById('styleRisk').innerText = data.telemetry.text_style.style_risk + "%";
                    document.getElementById('styleLabel').innerText = data.telemetry.text_style.label;

                    document.getElementById('imageRisk').innerText = data.telemetry.image_forensics.image_risk + "%";
                    document.getElementById('imageStatus').innerText = data.telemetry.image_forensics.status;

                    const sourcesList = document.getElementById('sourcesList');
                    sourcesList.innerHTML = '';
                    const sources = data.telemetry.realtime_news.matched_sources;
                    if (sources && sources.length > 0) {
                        sources.forEach(s => {
                            const li = document.createElement('li');
                            const a = document.createElement('a');
                            a.href = s.url;
                            a.target = '_blank';
                            a.className = 'text-info';
                            a.innerText = s.title;
                            li.appendChild(a);
                            sourcesList.appendChild(li);
                        });
                    } else {
                        sourcesList.innerHTML = '<li class="text-secondary">No matching mainstream news stories found.</li>';
                    }
                } catch (err) {
                    errorBox.innerText = err.message || 'Something went wrong while analyzing the content.';
                    errorBox.style.display = 'block';
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerText = 'Analyze Content Credibility';
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/analyze")
async def analyze_multimodal(text: str = Form(default=""), image: UploadFile = File(default=None)):
    has_text = len(text.strip()) > 0
    has_image = image is not None and image.filename not in (None, "")

    if not has_text and not has_image:
        raise HTTPException(status_code=400, detail="Provide text, an image, or both.")

    if has_text and len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text input too long (max 5000 characters).")

    image_bytes = None
    if has_image:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type '{image.content_type}'. Use JPEG, PNG, or WEBP.",
            )
        image_bytes = await image.read()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="Image exceeds 8MB upload limit.")
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    # Offload blocking work (network search, model inference, CV processing)
    # to a thread pool so it doesn't block the event loop for other requests.
    if has_text:
        realtime_res = await run_in_threadpool(verify_realtime_claim, text)
        style_res = await run_in_threadpool(analyze_text, text)
    else:
        realtime_res = {"realtime_risk": 0.0, "status": "No Text Provided", "matched_sources": []}
        style_res = {"style_risk": 0.0, "label": "No Text Provided"}

    if has_image:
        image_res = await run_in_threadpool(analyze_image_ela, image_bytes)
    else:
        image_res = {"image_risk": 0.0, "status": "No Image Uploaded", "ela_variance": 0.0}

    r_risk = realtime_res["realtime_risk"]
    s_risk = style_res["style_risk"]
    i_risk = image_res["image_risk"]

    if has_text and has_image:
        total_risk = (0.50 * r_risk) + (0.25 * s_risk) + (0.25 * i_risk)
    elif has_image and not has_text:
        total_risk = i_risk
    elif has_text and not has_image:
        total_risk = (0.70 * r_risk) + (0.30 * s_risk)
    else:
        total_risk = 0.0

    credibility_score = round(max(0.0, min(100.0, 100.0 - total_risk)), 2)

    return {
        "credibility_score": credibility_score,
        "telemetry": {
            "realtime_news": realtime_res,
            "text_style": style_res,
            "image_forensics": image_res,
        },
    }