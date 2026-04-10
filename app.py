import gradio as gr
import joblib

from src.preprocess import preprocess, load_stopwords
from src.build_features import transform_features


# =========================
# Load model & vectorizer
# =========================
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
stopwords = load_stopwords("data/vietnamese-stopwords.txt")


# =========================
# Label mapping
# =========================
label_map = {
    0: "Negative 😡",
    1: "Positive 😊"
}


# =========================
# Predict function
# =========================
def predict(text):
    if text is None or text.strip() == "":
        return "Please enter a valid text"

    # preprocess
    tokens = preprocess(text, stopwords)
    text_clean = " ".join(tokens)

    # vectorize
    X = transform_features([text_clean], vectorizer)

    # predict
    pred = model.predict(X)[0]

    # probability (optional)
    try:
        proba = model.predict_proba(X).max()
        return f"{label_map[int(pred)]} ({proba:.2f})"
    except:
        return label_map[int(pred)]


# =========================
# Gradio UI
# =========================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter product review here..."
    ),
    outputs="text",
    title="Vietnamese E-commerce Sentiment Analysis",
    description="End-to-end ML pipeline (TF-IDF + Logistic Regression) for classifying product reviews as positive or negative.",
)

# Launch app
demo.launch()