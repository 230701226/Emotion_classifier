import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np

# Load model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Get emotion labels
id2label = model.config.id2label
label_names = list(id2label.values())

# Main prediction function
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    top3_idx = probs.argsort()[-3:][::-1]
    top3_labels = [(label_names[i], round(float(probs[i]), 3)) for i in top3_idx]

    multi_labels = [(label_names[i], round(float(probs[i]), 3)) for i in range(len(probs)) if probs[i] > 0.3]

    return {
        "Top 3 Probable Emotions": {label: prob for label, prob in top3_labels},
        "Multi-Label (Threshold > 0.3)": {label: prob for label, prob in multi_labels}
    }

# Gradio Interface
demo = gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence to analyze emotions..."),
    outputs=[gr.Label(label="Top 3 Probable Emotions"), gr.Label(label="Multi-Label Emotions (> 0.3)")],
    title="🔍 Emotion Classifier with DistilBERT",
    description="Get Top 3 emotion probabilities + multi-label detection (threshold > 0.3). Model: DistilBERT fine-tuned on GoEmotions dataset."
)

demo.launch()
