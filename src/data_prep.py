import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(path):
    """
    Loads the Telco dataset, fixes data types, and drops missing values.
    """
    df = pd.read_csv(path)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing TotalCharges
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Clean SeniorCitizen from 0/1 to Yes/No
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})

    return df


def preprocess_data(df):
    """
    Encodes categorical variables, scales numeric variables,
    and splits features and target.
    """
    df = df.copy()

    # Drop customerID (non-informative)
    df.drop('customerID', axis=1, inplace=True)

    # Convert target label to binary (Yes → 1, No → 0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Identify categorical and numeric columns
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).drop('Churn', axis=1).columns

    # Encode categorical columns using LabelEncoder (simple version)
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Scale numeric columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y
