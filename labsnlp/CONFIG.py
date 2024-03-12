import os

TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
TEST_DATA_PATH =  os.path.join(os.path.dirname(__file__), '../data/test.csv')

# PATH_TO_VECTOR =  os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.50d.txt')

SUBMISSION_DATA_TEST_PATH =  os.path.join(os.path.dirname(__file__), '../data/test.csv')

OUTPUT_KEY = "is_sarcastic"

TFIDF_OPTIONS = {
    "max_df": 0.8,
    "max_features": 5000
}