import joblib

from src.preprocess import preprocess, load_stopwords
from src.build_features import transform_features


STOPWORD_PATH = "data/vietnamese-stopwords.txt"


def predict(text):
    # Load
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    stopwords = load_stopwords(STOPWORD_PATH)

    # Preprocess
    tokens = preprocess(text, stopwords)
    text_clean = " ".join(tokens)

    # Feature
    X = transform_features([text_clean], vectorizer)

    # Predict
    return model.predict(X)[0]


if __name__ == "__main__":
    text = input("Enter text: ")
    print("Prediction:", "positive" if predict(text) == 1 else "negative")