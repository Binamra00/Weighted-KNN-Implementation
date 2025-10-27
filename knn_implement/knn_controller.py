import pandas as pd
import os
from datetime import datetime
import numpy as np
from random import seed

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

from knn_scratch.knn_weighted import predict_classification
from knn_scratch.cross_validation import accuracy_metric, cross_validation_split


def run_experiment(config, test_type='t-test'):
    # Load data
    dataset_name = config['display_name']
    try:
        dataset_key_map = {
            'Hayes-Roth': 'hayes',
            'Car Evaluation': 'car',
            'Breast Cancer': 'cancer'
        }
        file_key = dataset_key_map[dataset_name]

        x_filename = os.path.join(config['processed_data_path'], f'X_{file_key}_processed.csv')
        y_filename = os.path.join(config['processed_data_path'], f'y_{file_key}_processed.csv')

        X = pd.read_csv(x_filename)
        y = pd.read_csv(y_filename)
        y = y.iloc[:, 0]
    except FileNotFoundError:
        print(f"Error: Processed data not found for {dataset_name}.")
        print(f"Attempted to load: {x_filename}")
        print("Please run the 'prep' command first: python knn_runner.py prep <dataset_name>")
        return

    # Prepare data formats
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    dataset_list = np.column_stack((X_np, y_np)).tolist()

    n_folds = 10
    seed(42)  # Fixed seed
    folds = cross_validation_split(dataset_list, n_folds)

    custom_knn_scores = []
    sklearn_knn_scores = []

    print(f"Running {n_folds}-fold cross-validation...")

    # Run each fold
    for i, test_fold in enumerate(folds):
        # Create the training set by combining all other folds
        train_set_list = list(folds)
        train_set_list.pop(i)
        train_set_list = sum(train_set_list, [])

        # Run knn scratch
        test_set_no_labels = [row[:-1] for row in test_fold]
        actual_labels = [row[-1] for row in test_fold]

        predictions_custom = []
        for test_row in test_set_no_labels:
            prediction = predict_classification(train_set_list, test_row, config['k'], config['distance_metric'])
            predictions_custom.append(prediction)

        acc_custom = accuracy_metric(actual_labels, predictions_custom)
        custom_knn_scores.append(acc_custom)

        # Run Scikit-Learn
        train_np = np.array(train_set_list)
        X_train, y_train = train_np[:, :-1], train_np[:, -1]

        test_np = np.array(test_fold)
        X_test, y_test = test_np[:, :-1], test_np[:, -1]

        y_train = y_train.astype(y_np.dtype)
        y_test = y_test.astype(y_np.dtype)

        sklearn_metric = config['distance_metric'].__name__.split('_')[0]
        sklearn_model = KNeighborsClassifier(n_neighbors=config['k'], metric=sklearn_metric)
        sklearn_model.fit(X_train, y_train)
        predictions_sklearn = sklearn_model.predict(X_test)

        acc_sklearn = accuracy_score(y_test, predictions_sklearn) * 100.0
        sklearn_knn_scores.append(acc_sklearn)

        print(f"Fold {i + 1}/{n_folds} complete. Custom Acc: {acc_custom:.2f}%, Sklearn Acc: {acc_sklearn:.2f}%")

    # Paired T-Test
    t_statistic, p_value = ttest_rel(custom_knn_scores, sklearn_knn_scores)
    alpha = 0.05

    # Comparison report
    report_path = config['comparison_report_path']
    os.makedirs(report_path, exist_ok=True)
    report_file = os.path.join(report_path, f'{dataset_name.replace(" ", "_")}_comparison_report.txt')

    with open(report_file, 'w') as f:
        f.write(f"--- KNN Comparison Report for {dataset_name} ---\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Parameters: k={config['k']}, Distance Metric='{config['distance_metric'].__name__}'\n")
        f.write("--------------------------------------------------\n\n")
        f.write("Fold-by-Fold Accuracy (%):\n")
        f.write("Fold | From-Scratch KNN | Scikit-Learn KNN\n")
        f.write("-----|------------------|-----------------\n")
        for i in range(n_folds):
            f.write(f"{i + 1:<5}| {custom_knn_scores[i]:<16.2f} | {sklearn_knn_scores[i]:.2f}\n")
        f.write("\n--------------------------------------------------\n")
        f.write("Summary Statistics:\n")
        f.write(f"Mean Accuracy (From-Scratch): {np.mean(custom_knn_scores):.2f}%\n")
        f.write(f"Mean Accuracy (Scikit-Learn): {np.mean(sklearn_knn_scores):.2f}%\n")
        f.write(f"Std Dev (From-Scratch):   {np.std(custom_knn_scores):.2f}\n")
        f.write(f"Std Dev (Scikit-Learn):   {np.std(sklearn_knn_scores):.2f}\n")

    print(f"\nComparison report saved to: {report_file}")

    # Hypothesis Test Report
    test_report_path = config['test_report_path']
    os.makedirs(test_report_path, exist_ok=True)
    test_file = os.path.join(test_report_path, f'{dataset_name.replace(" ", "_")}_hypothesis_test.txt')

    with open(test_file, 'w') as f:
        f.write(f"--- Hypothesis Test Report for {dataset_name} ---\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("--------------------------------------------------\n")
        f.write("Test: Paired t-test\n")
        f.write("Null Hypothesis (H0): The mean accuracy of the From-Scratch KNN and Scikit-Learn KNN is the same.\n")
        f.write("Alternative Hypothesis (H1): The mean accuracy of the two models is different.\n")
        f.write(f"Significance Level (alpha): {alpha}\n")
        f.write("--------------------------------------------------\n\n")
        f.write("Results:\n")
        f.write(f"T-statistic: {t_statistic:.4f}\n")
        f.write(f"P-value: {p_value:.4f}\n\n")
        f.write("Conclusion:\n")
        if p_value < alpha:
            f.write(f"Since the p-value ({p_value:.4f}) is less than alpha ({alpha}), we REJECT the null hypothesis.\n")
            f.write("There is a statistically significant difference in performance between the two implementations.\n")
        else:
            f.write(
                f"Since the p-value ({p_value:.4f}) is greater than or equal to alpha ({alpha}), we FAIL TO REJECT the null hypothesis.\n")
            f.write(
                "There is no statistically significant difference in performance between the two implementations.\n")

    print(f"Hypothesis test report saved to: {test_file}")
