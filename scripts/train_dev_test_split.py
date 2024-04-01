import pandas as pd
import os
from sklearn.model_selection import train_test_split

def make_split(path_to_csv, path_to_save, test_size):
    data = pd.read_csv(path_to_csv)

    X_train, X_temp = train_test_split(data, test_size=test_size)
    X_dev, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    X_train.to_csv(os.path.join(path_to_save, 'train.csv'))
    X_dev.to_csv(os.path.join(path_to_save, 'dev.csv'))
    X_test.to_csv(os.path.join(path_to_save, 'test.csv'))


if __name__ == "__main__":
    from fire import Fire

    Fire(make_split)