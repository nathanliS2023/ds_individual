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


def undersample_data(df):
    # Balance the dataset by undersampling the majority class
    majority_class = df['defects'].value_counts().idxmax()
    print("Majority class:", majority_class)
    minority_class_count = df['defects'].value_counts().min()
    print("Minority class count:", minority_class_count)
    df_majority_downsampled = df[df['defects'] == majority_class].sample(n=minority_class_count, random_state=42)
    df_minority = df[df['defects'] != majority_class]
    df_balanced = (pd.concat([df_majority_downsampled, df_minority]).
                   sample(frac=1, random_state=42).reset_index(drop=True))
    return df_balanced


# load and preprocess data
def load_and_preprocess_data(file_path):
    df = load_arff_to_dataframe(file_path)

    # remove rows with 0 values in lOCode column
    df = df[df['lOCode'] != 0]
    df = undersample_data(df)

    num_samples_per_class = 100
    # Sample a specified number of instances from each class
    df_class_0 = df[df['defects'] == b'false'].sample(n=num_samples_per_class, random_state=42)
    df_class_1 = df[df['defects'] == b'true'].sample(n=num_samples_per_class, random_state=42)
    df = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(df.shape)

    x, y = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    return x_train, x_test, y_train, y_test
