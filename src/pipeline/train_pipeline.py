import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from src.preprocess import load_stopwords, preprocess
from src.build_features import build_features
from src.train_model import train_model


# ===== CONFIG =====
DATA_PATH = "data/shopee_sentiment.csv"
STOPWORD_PATH = "data/vietnamese-stopwords.txt"


def run_pipeline():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Clean
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])

    # 3. Preprocess
    stopwords = load_stopwords(STOPWORD_PATH)
    df["text"] = df["text"].apply(lambda x: preprocess(x, stopwords))
    df["text"] = df["text"].apply(lambda x: " ".join(x))

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # 5. Feature
    X_train_vec, vectorizer = build_features(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 6. Train
    model = train_model(X_train_vec, y_train)

    # 7. Save
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Pipeline training completed!")


if __name__ == "__main__":
    run_pipeline()