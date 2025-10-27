import pandas as pd
import os
import numpy as np


def preprocess_hayes_roth(input_path, output_path):
    column_names = ['name', 'hobby', 'age', 'educational_level', 'marital_status', 'class']
    train_file = os.path.join(input_path, 'hayes-roth.data')
    test_file = os.path.join(input_path, 'hayes-roth.test')

    try:
        train_df = pd.read_csv(train_file, header=None, names=column_names)
        test_df = pd.read_csv(test_file, header=None, names=column_names)
    except FileNotFoundError:
        print(f"Error: Data files not found in {input_path}")
        return

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    for column in full_df.columns:
        if full_df[column].isnull().any():
            print(f"Found missing values in '{column}', filling with mode.")
            mode_val = full_df[column].mode()[0]
            full_df[column] = full_df[column].fillna(mode_val)

    processed_df = full_df.drop('name', axis=1)

    X = processed_df.drop('class', axis=1)
    y = processed_df['class']

    y = y.astype(int)

    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, 'X_hayes_processed.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y_hayes_processed.csv'), index=False)
    print("Hayes-Roth preprocessing complete.")


def preprocess_car_evaluation(input_path, output_path):
    column_names = [
        'buying_price', 'maintenance_cost', 'doors', 'person_capacity',
        'trunk_size', 'safety_rating', 'class'
    ]
    data_file = os.path.join(input_path, 'car.data')

    try:
        df = pd.read_csv(data_file, header=None, names=column_names)
    except FileNotFoundError:
        print(f"Error: car.data not found in {input_path}")
        return

    # Mappings for Ordinal Encoding
    price_map = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
    doors_map = {'2': 0, '3': 1, '4': 2, '5more': 3}
    persons_map = {'2': 0, '4': 1, 'more': 2}
    luggage_map = {'small': 0, 'med': 1, 'big': 2}
    safety_map = {'low': 0, 'med': 1, 'high': 2}
    class_map = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

    df['buying_price'] = df['buying_price'].map(price_map)
    df['maintenance_cost'] = df['maintenance_cost'].map(price_map)
    df['doors'] = df['doors'].map(doors_map)
    df['person_capacity'] = df['person_capacity'].map(persons_map)
    df['trunk_size'] = df['trunk_size'].map(luggage_map)
    df['safety_rating'] = df['safety_rating'].map(safety_map)
    df['class'] = df['class'].map(class_map)

    X = df.drop('class', axis=1)
    y = df['class']

    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, 'X_car_processed.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y_car_processed.csv'), index=False)
    print("Car Evaluation preprocessing complete.")


def preprocess_breast_cancer(input_path, output_path):
    column_names = [
        'class', 'age', 'menopause', 'tumor-size', 'involved_lymph_nodes',
        'node-caps', 'malignancy_degree', 'breast', 'tumor_location_quadrant',
        'received_radiation'
    ]
    data_file = os.path.join(input_path, 'breast-cancer.data')

    try:
        df = pd.read_csv(data_file, header=None, names=column_names)
    except FileNotFoundError:
        print(f"Error: breast-cancer.data not found in {input_path}")
        return

    for column in df.columns:
        if any(df[column].astype(str).str.contains('\?')):
            df[column] = df[column].replace('?', np.nan)
            mode_val = df[column].mode()[0]
            df[column] = df[column].fillna(mode_val)

    X = df.drop('class', axis=1)
    y = df['class']

    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, columns=X.columns, drop_first=True)
    y_encoded = y.map({'no-recurrence-events': 0, 'recurrence-events': 1})

    os.makedirs(output_path, exist_ok=True)
    X_encoded.to_csv(os.path.join(output_path, 'X_cancer_processed.csv'), index=False)
    y_encoded.to_csv(os.path.join(output_path, 'y_cancer_processed.csv'), index=False)
    print("Breast Cancer preprocessing complete.")
