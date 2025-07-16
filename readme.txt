# Emotion Classifier (Gradio App)

This project is a lightweight emotion classification app using **DistilBERT fine-tuned on GoEmotions**, built with **Gradio** for quick deployment.

## 🔥 Features
- Detects top 5 emotions from a sentence.
- Model: `bhadresh-savani/distilbert-base-uncased-emotion` (HuggingFace)
- Frontend: Gradio Interface
- Ready for deployment on HuggingFace Spaces or Render.com

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/emotion-app.git
cd emotion-app
pip install -r requirements.txt
python app.py
