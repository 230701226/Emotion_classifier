# DEEP LEARNING

#  EMOTION CLASSIFIER - DISTILBERT (GOEMOTIONS)

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: POOJA AK
**DOMAIN**: DATA SCIENCE
**DURATION**: 4 WEEKS  
**MENTOR**: NEELA SANTOSH

---

#  DESCRIPTION OF TASK

This project is an advanced **Emotion Classification System** built using `DistilBERT` and fine-tuned on the **GoEmotions dataset** from Google AI .

The goal was to create a system that accepts any English sentence and predicts the most relevant emotions associated with it. The model supports:

 **Multi-label classification** (multiple emotions per sentence)  
 **Top-3 emotion prediction** with confidence scores  
 Deployed with a beautiful **Gradio UI** on **Hugging Face Spaces**

## 🛠️ Technologies Used

- Python
- HuggingFace Transformers 
- DistilBERT
- GoEmotions dataset
- Torch
- Gradio
- HuggingFace Spaces

---

# HOW TO RUN ON HUGGING FACE SPACES

1. Clone or fork the repository:
```bash
git clone https://huggingface.co/spaces/your-username/emotion-classifier
Make sure app.py and requirements.txt are present.

Push the files to your Hugging Face Space repo:

bash
Copy
Edit
git lfs install
git add .
git commit -m "Initial commit"
git push
HuggingFace will automatically detect app.py and launch the app.

🧪 SAMPLE WORKING INPUTS
These inputs give logically correct emotion outputs:

Input Sentence	Expected Emotions
I just got promoted today at work!	joy, excitement, pride
I'm feeling so lonely and tired lately.	sadness, disappointment, loneliness
Ugh, the traffic jam again! I'm so done!	anger, frustration, annoyance
That was so kind of you, thank you!	gratitude, admiration
I'm scared to take the exam tomorrow.	fear, nervousness, anxiety
You inspire me to be better.	admiration, love
I can't believe I got the job!	surprise, joy, relief

📂 PROJECT STRUCTURE
bash
Copy
Edit
emotion-classifier/
├── app.py                # Main Gradio app
├── requirements.txt      # Dependencies
├── README.md             # This file
💡 LEARNING OUTCOMES
 Understood multi-label NLP modeling with Transformers
 Learned how to fine-tune DistilBERT on emotion classification
 Deployed an AI app using Hugging Face Spaces
 Created a production-ready Gradio interface
 Gained hands-on experience with real-world datasets

 OUTPUT INTERFACE
https://huggingface.co/spaces/POOJA1222/distilbert-emotion-classifier


https://1drv.ms/i/c/6167ef7f5441f9a9/EV5uzk-8FBhHkS2LVDKUnD4BSCc81MQfAdkgLhwK_HPSFg?e=2kqVb6

FINAL OUTCOME
A complete, polished Emotion Classifier web app powered by AI, live on Hugging Face and ready to use in real-world applications like sentiment analysis, support bots, social monitoring, and more.
