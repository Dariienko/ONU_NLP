from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

def train_gaussian_nb_classifier(X, y, cv):
    clf = GaussianNB()
    clf_CV = CalibratedClassifierCV(clf, cv = cv)
    clf_CV.fit(X, y)

    return clf_CV
