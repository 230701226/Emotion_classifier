# 😃 Emotion Classifier - DistilBERT

This is a HuggingFace Spaces app using `DistilBERT` fine-tuned on the **GoEmotions** dataset to classify emotions in text.

## 🔍 Features

- Predicts **Top 3 most probable emotions** with scores.
- Supports **Multi-label emotion detection** (threshold > 0.3).
- Runs on **Gradio** UI, deployable on Hugging Face Spaces.

---

## 🚀 Sample Inputs

Try these for meaningful emotion predictions:

### 1. "I just got promoted today at work!"
- Joy, Excitement, Pride

### 2. "I'm feeling so lonely and tired lately."
- Sadness, Disappointment, Loneliness

### 3. "Ugh, the traffic jam again! I'm so done!"
- Anger, Frustration, Annoyance

### 4. "That was so kind of you, thank you!"
- Gratitude, Joy, Admiration

### 5. "I'm scared to take the exam tomorrow."
- Fear, Nervousness, Apprehension

---

## 🧠 Model

- `bhadresh-savani/distilbert-base-uncased-emotion`
- Trained on Google’s **GoEmotions** dataset (27 emotion classes).

---

## 🛠️ Run Locally

```bash
git clone https://huggingface.co/spaces/your-username/emotion-classifier
cd emotion-classifier
pip install -r requirements.txt
python app.py
