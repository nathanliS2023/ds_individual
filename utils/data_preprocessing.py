from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# load and transform data from arff files
def load_arff_to_dataframe(file_path):
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df


# split data into features and target values
def preprocess_data(df):
    y_data = df.iloc[:, -1].values
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(y_data)

    # fill missing values with median
    x_copy = df.iloc[:, :-1].copy()
    imputer = SimpleImputer(strategy="most_frequent")
    new_x = imputer.fit_transform(x_copy)
    new_x_df = pd.DataFrame(new_x, columns=x_copy.columns, index=x_copy.index)
    return new_x_df, y


# split data into training and test sets
def split_data(x, y, test_size=0.20, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


# load and preprocess data
def load_and_preprocess_data(file_path):
    df = load_arff_to_dataframe(file_path)
    x, y = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    return x_train, x_test, y_train, y_test
