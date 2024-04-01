import pandas as pd
import numpy as np

from labsnlp.CONFIG import *
from labsnlp.preprocessing import preprocess_text
from labsnlp.vectorize import get_tfidf_vectorizer
from labsnlp.model import train_gaussian_nb_classifier

from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from joblib import dump, load

def train():
    train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    dev_data = pd.read_csv(DEV_DATA_PATH, index_col=0)
    test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)

    preprocessed_text_train_data = preprocess_text(train_data)
    preprocessed_text_dev_data = preprocess_text(dev_data)
    preprocessed_text_test_data = preprocess_text(test_data)

    vectorizer = get_tfidf_vectorizer(preprocessed_text_train_data['headline'].values.tolist(),**TFIDF_OPTIONS)

    # sparse
    X_train = vectorizer.transform(preprocessed_text_train_data['headline'].values.tolist())
    X_dev = vectorizer.transform(preprocessed_text_dev_data['headline'].values.tolist())
    X_test = vectorizer.transform(preprocessed_text_test_data['headline'].values.tolist())
    
    print("SHAPE: ", X_train.shape)
    
    # була така спроба застосування PCA  
    # n_components= 3866
    #ROC-AUC DEV: 0.48363616947717686
    #ROC-AUC TEST: 0.5618465221360092
    #ROC-AUC BEST_MODEL: 0.5618465221360092
    
    # # serch n_components
    # pca = PCA(n_components=5000)
    # pca.fit(X_train.toarray())

    # explained_variance_ratio = pca.explained_variance_ratio_
    # cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    # n_components_to_retain = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1
    
    # print("n_components=", n_components_to_retain)
    
    # # pca
    # pca = PCA(n_components=n_components_to_retain)
    
    # # fit and transform pca
    # X_train_pca = pca.fit_transform(X_train.toarray())
    
    # # transform 
    # X_dev_pca = pca.transform(X_dev.toarray())
    # X_test_pca = pca.transform(X_test.toarray())

    # only for gothic
    clf1 = train_gaussian_nb_classifier(X_train.toarray(), train_data[OUTPUT_KEY].values.tolist(), 3)
    
    clf2 = train_gaussian_nb_classifier(X_train.toarray(), train_data[OUTPUT_KEY].values.tolist(), 5)

    predictions_dev = clf1.predict_proba(X_dev.toarray())[:, 1]
    roc_auc_clf1 = roc_auc_score(y_true=dev_data[OUTPUT_KEY].values.tolist(), y_score=predictions_dev)
    print(f'ROC-AUC DEV: {roc_auc_clf1}')

    predictions_test = clf2.predict_proba(X_test.toarray())[:, 1]
    roc_auc_clf2 = roc_auc_score(y_true=test_data[OUTPUT_KEY].values.tolist(), y_score=predictions_test)

    print(f'ROC-AUC TEST: {roc_auc_clf2}')
    
    #best model
    best_model = clf1 if roc_auc_clf1 > roc_auc_clf2 else clf2

    print(f'ROC-AUC BEST_MODEL: {roc_auc_clf1 if best_model == clf1 else roc_auc_clf2}')
    # saving vectorizer and model
    dump(best_model, 'clf.joblib')
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