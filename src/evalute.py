import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.preprocess import load_stopwords, preprocess
from src.build_features import transform_features


DATA_PATH = "data/shopee_sentiment.csv"
STOPWORD_PATH = "data/vietnamese-stopwords.txt"


def main():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])

    stopwords = load_stopwords(STOPWORD_PATH)
    df["text"] = df["text"].apply(lambda x: preprocess(x, stopwords))
    df["text"] = df["text"].apply(lambda x: " ".join(x))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    X_test_vec = transform_features(X_test, vectorizer)

    y_pred = model.predict(X_test_vec)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()