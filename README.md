# Gendered Abuse Detection in Indic Languages 🛡️

This project is an advanced NLP application that detects gendered abusive content in English, Hindi, and Tamil texts. It leverages a custom deep learning model built on top of multilingual transformer architectures and provides an easy-to-use web interface using Streamlit.

---

## 🚀 Features
- **Multilingual Support:** Detects gendered abuse in English, Hindi, and Tamil.
- **State-of-the-art Model:** Uses a custom neural network with XLM-RoBERTa as the encoder.
- **User-Friendly Web App:** Modern, responsive UI built with Streamlit.
- **Real-time Prediction:** Enter any text and get instant feedback on whether it is gender abusive or not.
- **Open Source:** Easily extensible for more languages or other types of abuse.

---

## 🏗️ Project Structure
```
├── app.py                # Streamlit web app
├── best_gender_abuse_detector.pt  # Trained PyTorch model (add this file after training)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── uli_dataset/          # Dataset (not needed for inference)
└── ...
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Gendered-Abuse-Detection-in-Indic-Languages
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add the trained model file**
   - Place `best_gender_abuse_detector.pt` in the project root (get it from your training environment).

4. **Run the app locally**
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`.

---

## 📝 Usage
- Enter any text (in English, Hindi, or Tamil) in the input box.
- Click **Detect**.
- The app will display whether the text is gender abusive or not.

---

## 📚 Model Details
- **Architecture:** XLM-RoBERTa encoder + custom CNN and FC layers
- **Frameworks:** PyTorch, HuggingFace Transformers
- **Training:** Trained on a labeled dataset of social media posts in English, Hindi, and Tamil

---

## 🙏 Credits
- Developed by: [Karan Yadav]
- Dataset: [ULI Dataset](https://www.kaggle.com/datasets/rtatman/uli-dataset)
- Model inspiration: [HuggingFace Transformers](https://huggingface.co/transformers/)
