from sklearn.naive_bayes import GaussianNB


def train_gaussian_nb_classifier(X, y):
    clf = GaussianNB()
    clf.fit(X, y)

    return clf
