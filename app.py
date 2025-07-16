import gradio as gr
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

# Load model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Labels for GoEmotions
labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
          'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
          'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
          'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
          'sadness', 'surprise', 'neutral']

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy().flatten()
    top_idx = np.argmax(probs)
    return {labels[i]: float(probs[i]) for i in np.argsort(-probs)[:5]}  # top 5 emotions

interface = gr.Interface(fn=predict_emotion,
                         inputs=gr.Textbox(placeholder="Enter text..."),
                         outputs=gr.Label(num_top_classes=5),
                         title="Emotion Classifier (GoEmotions)",
                         description="Predicts emotions using DistilBERT fine-tuned on GoEmotions.")

if __name__ == "__main__":
    interface.launch()
