import pandas as pd
from labsnlp.CONFIG import *
from labsnlp.preprocessing import preprocess_text
from labsnlp.vectorize import get_tfidf_vectorizer
from labsnlp.model import train_gaussian_nb_classifier

from sklearn.metrics import roc_auc_score
from joblib import dump, load

def train():
    train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)

    preprocessed_text_train_data = preprocess_text(train_data)
    preprocessed_text_test_data = preprocess_text(test_data)

    vectorizer = get_tfidf_vectorizer(preprocessed_text_train_data['headline'].values.tolist(), **TFIDF_OPTIONS)

    # sparse
    X_train = vectorizer.transform(preprocessed_text_train_data['headline'].values.tolist())
    X_test = vectorizer.transform(preprocessed_text_test_data['headline'].values.tolist())

    # only for gothic
    clf = train_gaussian_nb_classifier(X_train.toarray(), train_data[OUTPUT_KEY].values.tolist())

    predictions = clf.predict_proba(X_test.toarray())[:, 1]

    print(f'ROC-AUC: {roc_auc_score(y_true=test_data[OUTPUT_KEY].values.tolist(), y_score=predictions)}')

    # saving vectorizer and model
    dump(clf, 'clf.joblib')
    dump(vectorizer, 'vectorizer.joblib')


def prepare_submission():
    clf = load('clf.joblib')
    vectorizer = load('vectorizer.joblib')

    data = pd.read_csv(SUBMISSION_DATA_TEST_PATH, index_col=0)
    preprocessed_data = preprocess_text(data)

    X = vectorizer.transform(preprocessed_data['headline'].values.tolist())
    predictions = clf.predict_proba(X.toarray())[:, 1].tolist()

    data[OUTPUT_KEY] = predictions
    data[OUTPUT_KEY].to_csv('submission.csv', sep=';')


if __name__ == "__main__":
    train()
    prepare_submission()