from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(sentences, **tfidf_params): 
    vectorizer = TfidfVectorizer(**tfidf_params)
    vectorizer.fit(sentences)

    return vectorizer