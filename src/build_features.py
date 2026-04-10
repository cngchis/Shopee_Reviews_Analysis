from sklearn.feature_extraction.text import TfidfVectorizer


def build_features(texts, max_features=5000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def transform_features(texts, vectorizer):
    return vectorizer.transform(texts)