import pandas as pd
import numpy as np

FEATURE_COLUMNS = ['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']
CLASS_COLUMN = 'bird category'
CLASS_LABELS = ['A', 'B', 'C']


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Preprocess gender(male=1,female=0)
    df['gender'] = df['gender'].str.strip().str.lower().map({'male': 1, 'female': 0})
    mode_val = df['gender'].mode()[0]
    df['gender'] = df['gender'].fillna(mode_val)
    return df


def get_two_class_data(df: pd.DataFrame, class1: str, class2: str,
                       feature1: str, feature2: str):

    mask = df[CLASS_COLUMN].isin([class1, class2])
    subset = df[mask].copy()

    X = subset[[feature1, feature2]].values.astype(float)
   
    y = np.where(subset[CLASS_COLUMN].values == class1, 1, -1)

    return X, y, subset


def train_test_split_by_class(df: pd.DataFrame, class1: str, class2: str,
                               feature1: str, feature2: str,
                               n_train: int = 30, n_test: int = 20,
                               random_state: int = 42):

   
    rng = np.random.RandomState(random_state)

    train_dfs, test_dfs = [], []
    for cls in [class1, class2]:
        cls_df = df[df[CLASS_COLUMN] == cls]
        indices = rng.permutation(len(cls_df))
        train_dfs.append(cls_df.iloc[indices[:n_train]])
        test_dfs.append(cls_df.iloc[indices[n_train:n_train + n_test]])

    train_df = pd.concat(train_dfs).sample(frac=1, random_state=random_state)
    test_df  = pd.concat(test_dfs)

    def extract(d):
        X = d[[feature1, feature2]].values.astype(float)
        y = np.where(d[CLASS_COLUMN].values == class1, 1, -1)
        return X, y

    X_train, y_train = extract(train_df)
    X_test,  y_test  = extract(test_df)

    return X_train, y_train, X_test, y_test


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    return (X_train - mean) / std, (X_test - mean) / std
