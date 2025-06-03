import streamlit as st
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# --- Model Definition ---
class GenderAbuseDetectionModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=2, dropout_rate=0.3):
        super(GenderAbuseDetectionModel, self).__init__()
        self.encoder = pretrained_model
        hidden_size = self.encoder.config.hidden_size
        self.conv1 = nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(256)
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        x = sequence_output.transpose(1, 2)
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        max_pooled = self.max_pool(x1).squeeze(-1)
        avg_pooled = self.avg_pool(x1).squeeze(-1)
        concat_features = torch.cat([max_pooled, avg_pooled], dim=1)
        normalized_features = self.layer_norm(concat_features)
        x = F.relu(self.fc1(normalized_features))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# --- Model & Tokenizer Loader ---
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    model = GenderAbuseDetectionModel(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("best_gender_abuse_detector.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device

# --- Prediction Function ---
def predict_abuse(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        _, preds = torch.max(outputs, dim=1)
    return preds.item()

# --- Page Config ---
st.set_page_config(
    page_title="Gendered Abuse Detection in Indic Languages",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Header ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #4F8BF9;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4F8BF9;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Gendered Abuse Detection in Indic Languages")
st.markdown("""
This web app uses a state-of-the-art NLP model to detect gendered abuse in English, Hindi, and Tamil texts. Enter any text below to check if it contains gendered abusive content.
""")

# --- Instructions ---
st.header("How to use:")
st.markdown("""
- Enter your text in the box below (supports English, Hindi, and Tamil).
- Click **Detect** to see if the text is gender abusive or not.
- The model will highlight if the text is abusive (gendered) or non-abusive.
""")

# --- Input Form ---
st.header("Try it out!")
with st.form("abuse_form"):
    user_text = st.text_area("Enter text here:", height=120, max_chars=500)
    submitted = st.form_submit_button("Detect")

if submitted:
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Loading model and predicting..."):
            model, tokenizer, device = load_model_and_tokenizer()
            pred = predict_abuse(user_text, model, tokenizer, device)
            label = "Abusive" if pred == 1 else "Non-abusive"
            if pred == 1:
                st.error(f"**Prediction:** {label}")
            else:
                st.success(f"**Prediction:** {label}")
        st.write(f"**Input:** {user_text}")

# --- About Section ---
st.sidebar.title("About this project")
st.sidebar.info(
    """
    **Gendered Abuse Detection in Indic Languages**\
    Developed as an NLP project to help identify gendered abusive content in social media and online platforms.\
    \n
    - Supports English, Hindi, and Tamil
    - Built with PyTorch, HuggingFace Transformers, and Streamlit
    - [GitHub Repo](#) (add your link)
    """
) 