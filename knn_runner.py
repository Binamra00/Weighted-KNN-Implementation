import argparse
import sys

from knn_implement.config import CONFIG
from knn_implement.knn_controller import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool for preprocessing data and running KNN experiments.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'command',
        choices=['prep', 'run'],
        help="The command to execute:\n"
             "  'prep' - Runs the preprocessing pipeline for a dataset.\n"
             "  'run'  - Runs the full 10-fold CV experiment and comparison."
    )
    parser.add_argument(
        'dataset',
        choices=['hayes_roth', 'car_evaluation', 'breast_cancer'],
        help="The dataset to use for the specified command."
    )
    parser.add_argument(
        '-t', '--test',
        choices=['t-test'],
        default='t-test',
        help="Specify the hypothesis test to run (default: 't-test')."
    )

    # Help message
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Load configuration
    if args.dataset not in CONFIG:
        print(f"Error: Dataset '{args.dataset}' not found in configuration.")
        return

    config = CONFIG[args.dataset]
    dataset_name = config['display_name']

    # Execute command
    if args.command == 'prep':
        print(f"\n--- Running PREPROCESSING for [{dataset_name}] ---")
        try:
            preprocess_func = config['preprocess_func']
            preprocess_func(config['raw_data_path'], config['processed_data_path'])
            print(f"--- Preprocessing for [{dataset_name}] COMPLETE ---\n")
        except Exception as e:
            print(f"\nAn error occurred during preprocessing: {e}")

    elif args.command == 'run':
        print(f"\n--- Running KNN EXPERIMENT for [{dataset_name}] ---")
        try:
            run_experiment(config, test_type=args.test)
            print(f"--- Experiment for [{dataset_name}] COMPLETE ---\n")
        except Exception as e:
            print(f"\nAn error occurred during the experiment: {e}")


if __name__ == '__main__':
    main()
