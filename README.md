# Multimodal Disinformation & Deepfake Detector

An AI-powered web application that verifies the authenticity of online news and media by evaluating text headlines alongside uploaded images to generate a unified **Content Credibility Score (0–100%)**.

## Key Features
* **Text Disinformation Analysis:** Uses a fine-tuned BERT classifier (`mrm8488/bert-tiny-finetuned-fake-news-detection`) to flag clickbait and fake news patterns.
* **Image Forensics:** Applies Error Level Analysis (ELA) using OpenCV and Pillow to detect image tampering, splices, and AI-generated artifacts.
* **Calibrated Noise Filtering:** Custom variance thresholds prevent false positives on standard compressed social media JPEGs.
* **Interactive Web Dashboard:** Built with FastAPI and Bootstrap 5 for real-time analysis and structured visual telemetry.

## Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **ML & Computer Vision:** PyTorch, Hugging Face Transformers, OpenCV, Pillow, NumPy
* **Frontend:** HTML5, JavaScript, Bootstrap 5

## Scoring Logic
$$\text{Credibility Score} = 100\% - (0.4 \times \text{Text Risk} + 0.6 \times \text{Image Risk})$$

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/siddharth-246/multimodal-disinformation-detector.git](https://github.com/siddharth-246/multimodal-disinformation-detector.git)
   cd multimodal-disinformation-detector