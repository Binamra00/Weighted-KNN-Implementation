import os
from knn_scratch.distance_metrics import hamming_distance, manhattan_distance, euclidean_distance
from knn_implement.dataset_preprocessor import (
    preprocess_hayes_roth,
    preprocess_car_evaluation,
    preprocess_breast_cancer
)

# Base Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_PATH = os.path.dirname(CURRENT_DIR)

# Dynamic Paths
DATASET_PATH = os.path.join(BASE_PROJECT_PATH, 'datasets')
PREPROCESSED_PATH = os.path.join(BASE_PROJECT_PATH, 'data_preprocessing')
COMPARISON_REPORT_PATH = os.path.join(BASE_PROJECT_PATH, 'cross_validation_report')
TEST_REPORT_PATH = os.path.join(BASE_PROJECT_PATH, 'evaluation_test_report')


# Main Configuration Dictionary
CONFIG = {
    'hayes_roth': {
        'display_name': 'Hayes-Roth',
        'raw_data_path': os.path.join(DATASET_PATH, 'hayes+roth'),
        'processed_data_path': os.path.join(PREPROCESSED_PATH, 'hayes_roth'),
        'comparison_report_path': COMPARISON_REPORT_PATH,
        'test_report_path': TEST_REPORT_PATH,
        'preprocess_func': preprocess_hayes_roth,
        'distance_metric': hamming_distance,
        'k': 5
    },
    'car_evaluation': {
        'display_name': 'Car Evaluation',
        'raw_data_path': os.path.join(DATASET_PATH, 'car+evaluation'),
        'processed_data_path': os.path.join(PREPROCESSED_PATH, 'car_evaluation'),
        'comparison_report_path': COMPARISON_REPORT_PATH,
        'test_report_path': TEST_REPORT_PATH,
        'preprocess_func': preprocess_car_evaluation,
        'distance_metric': manhattan_distance,
        'k': 5
    },
    'breast_cancer': {
        'display_name': 'Breast Cancer',
        'raw_data_path': os.path.join(DATASET_PATH, 'breast+cancer'),
        'processed_data_path': os.path.join(PREPROCESSED_PATH, 'breast_cancer'),
        'comparison_report_path': COMPARISON_REPORT_PATH,
        'test_report_path': TEST_REPORT_PATH,
        'preprocess_func': preprocess_breast_cancer,
        'distance_metric': euclidean_distance,
        'k': 5
    }
}
