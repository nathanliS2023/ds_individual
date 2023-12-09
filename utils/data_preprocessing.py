from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def plot_feature_ranges(df):
    # Creating a box plot for each feature
    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.xticks(rotation=45)
    plt.title("Feature Ranges")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.show()


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

    # fill missing values with mean
    x_copy = df.iloc[:, :-1].copy()
    imputer = SimpleImputer(strategy="mean")
    new_x = imputer.fit_transform(x_copy)

    # Standardize the features
    scaler = StandardScaler()
    standardized_x = scaler.fit_transform(new_x)
    new_x_df = pd.DataFrame(standardized_x, columns=x_copy.columns, index=x_copy.index)

    # new_x_df = pd.DataFrame(new_x, columns=x_copy.columns, index=x_copy.index)
    return new_x_df, y


def balance_data(df, target_column, majority_samples, minority_samples):
    # Get counts of each class
    class_counts = df[target_column].value_counts()

    # Determine majority and minority classes
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    # Adjust the number of samples if they are more than available samples
    majority_samples = min(majority_samples, class_counts[majority_class])
    minority_samples = min(minority_samples, class_counts[minority_class])

    # Sample the specified number of instances from each class
    df_majority = df[df[target_column] == majority_class].sample(n=majority_samples, random_state=42)
    df_minority = df[df[target_column] == minority_class].sample(n=minority_samples, random_state=42)

    # Concatenate and shuffle the dataset
    balanced_df = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


# split data into training and test sets
def split_data(x, y, test_size=0.20, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


# load and preprocess data
def load_and_preprocess_data(file_path):
    df = load_arff_to_dataframe(file_path)

    # remove rows with 0 values in lOCode column
    # df = df[df['lOCode'] != 0]
    df = balance_data(df, 'defects', 200, 200)

    print(df.shape)

    x, y = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    plot_feature_ranges(x_train)
    return x_train, x_test, y_train, y_test
