import io
import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Multimodal Disinformation & Deepfake Detector")

# ==========================================
# 1. TEXT DETECTION ENGINE (Fine-Tuned BERT)
# ==========================================
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
text_model.eval()

def analyze_text(text: str) -> dict:
    """Analyzes headline or article text to detect fake news probability."""
    if not text or len(text.strip()) == 0:
        return {"risk_score": 0.0, "label": "No Text Provided"}
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    
    # Class mapping for mrm8488 model: Index 0 = FAKE, Index 1 = REAL
    fake_prob = float(probs[0] if probs.ndim > 0 else probs)
    risk_percentage = round(fake_prob * 100, 2)
    label = "High Suspicion (Fake/Clickbait)" if risk_percentage > 60 else "Likely Authentic"
    
    return {"risk_score": risk_percentage, "label": label}


# ==========================================
# 2. IMAGE FORENSICS ENGINE (Calibrated ELA)
# ==========================================
def perform_ela(image_bytes: bytes, quality: int = 90) -> tuple[float, float]:
    """Calculates JPEG compression difference artifacts."""
    original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    buffer = io.BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    
    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    ela_array = np.array(ela_image)
    avg_variance = float(np.var(ela_array))
    
    return max_diff, avg_variance

def analyze_image(image_bytes: bytes) -> dict:
    """Analyzes image tampering with balanced variance thresholds for compressed images."""
    try:
        max_diff, variance = perform_ela(image_bytes)
        
        # Threshold set to 350.0 to prevent standard social media JPEGs from flagging false positives
        image_risk = min(100.0, (variance / 350.0) * 100)
        image_risk = round(image_risk, 2)
        
        label = "Tampered / AI Generated Artifacts" if image_risk > 65 else "Authentic Image"
        
        return {
            "image_risk_score": image_risk,
            "max_ela_difference": float(max_diff),
            "variance": round(variance, 2),
            "label": label
        }
    except Exception as e:
        return {"image_risk_score": 0.0, "label": f"Error processing image: {str(e)}"}


# ==========================================
# 3. FASTAPI ENDPOINTS & UI DASHBOARD
# ==========================================

@app.post("/api/v1/analyze")
async def analyze_content(
    text: str = Form(None),
    image: UploadFile = File(None)
):
    """Multimodal Endpoint for text and image authentication."""
    text_result = {"risk_score": 0.0, "label": "N/A"}
    image_result = {"image_risk_score": 0.0, "label": "N/A"}
    
    if text:
        text_result = analyze_text(text)
        
    if image:
        contents = await image.read()
        image_result = analyze_image(contents)
        
    t_score = text_result.get("risk_score", 0.0)
    i_score = image_result.get("image_risk_score", 0.0)
    
    if text and image:
        overall_credibility = round(100.0 - (0.4 * t_score + 0.6 * i_score), 2)
    elif text:
        overall_credibility = round(100.0 - t_score, 2)
    elif image:
        overall_credibility = round(100.0 - i_score, 2)
    else:
        overall_credibility = 100.0

    return {
        "overall_credibility_score": f"{overall_credibility}%",
        "text_analysis": text_result,
        "image_analysis": image_result
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Renders clean frontend UI with metric cards."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multimodal Fake News & Image Detector</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="bg-light p-4">
        <div class="container bg-white p-4 rounded shadow" style="max-width: 650px;">
            <h2 class="mb-3 text-primary fw-bold">Content Authenticity Scanner</h2>
            <form id="detectForm">
                <div class="mb-3">
                    <label class="form-label fw-semibold">Article Text or Headline:</label>
                    <textarea id="textInput" class="form-control" rows="3" placeholder="Paste article content..."></textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label fw-semibold">Upload Image:</label>
                    <input type="file" id="imageInput" class="form-control" accept="image/*">
                </div>
                <button type="submit" class="btn btn-primary w-100 fw-bold">Analyze Authenticity</button>
            </form>

            <div id="results" class="mt-4 d-none">
                <hr class="my-4">
                <h4 class="fw-bold mb-3">Analysis Results</h4>
                
                <div class="p-3 mb-3 bg-light rounded border text-center">
                    <span class="fs-5 fw-semibold me-2">Overall Credibility Score:</span>
                    <span id="credScore" class="badge fs-5"></span>
                </div>

                <div class="row g-3">
                    <!-- Text Metrics Card -->
                    <div class="col-md-6">
                        <div class="card h-100 border-0 bg-light shadow-sm">
                            <div class="card-body">
                                <h6 class="card-title text-uppercase text-muted fw-bold mb-3">Text Analysis</h6>
                                <p class="mb-2"><strong>Risk Score:</strong> <span id="textRisk">--</span>%</p>
                                <p class="mb-0"><strong>Status:</strong> <span id="textLabel" class="fw-semibold">--</span></p>
                            </div>
                        </div>
                    </div>

                    <!-- Image Metrics Card -->
                    <div class="col-md-6">
                        <div class="card h-100 border-0 bg-light shadow-sm">
                            <div class="card-body">
                                <h6 class="card-title text-uppercase text-muted fw-bold mb-3">Image Forensics</h6>
                                <p class="mb-1"><strong>Risk Score:</strong> <span id="imageRisk">--</span>%</p>
                                <p class="mb-1"><strong>Status:</strong> <span id="imageLabel" class="fw-semibold">--</span></p>
                                <hr class="my-2">
                                <p class="mb-1 text-muted small">Max ELA Difference: <span id="elaDiff">--</span></p>
                                <p class="mb-0 text-muted small">Variance: <span id="elaVar">--</span></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('detectForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData();
                const text = document.getElementById('textInput').value;
                const imageFile = document.getElementById('imageInput').files[0];
                
                if (text) formData.append('text', text);
                if (imageFile) formData.append('image', imageFile);
                
                const res = await fetch('/api/v1/analyze', { method: 'POST', body: formData });
                const data = await res.json();
                
                document.getElementById('results').classList.remove('d-none');
                
                const scoreElem = document.getElementById('credScore');
                const scoreVal = parseFloat(data.overall_credibility_score);
                scoreElem.innerText = data.overall_credibility_score;
                scoreElem.className = scoreVal > 50 ? 'badge bg-success fs-5' : 'badge bg-danger fs-5';
                
                document.getElementById('textRisk').innerText = data.text_analysis.risk_score;
                document.getElementById('textLabel').innerText = data.text_analysis.label;
                
                document.getElementById('imageRisk').innerText = data.image_analysis.image_risk_score;
                document.getElementById('imageLabel').innerText = data.image_analysis.label;
                document.getElementById('elaDiff').innerText = data.image_analysis.max_ela_difference;
                document.getElementById('elaVar').innerText = data.image_analysis.variance;
            });
        </script>
    </body>
    </html>
    """