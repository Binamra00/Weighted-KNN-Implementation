=========================================================
 README: Scratch KNN Implementation and Evaluation
=========================================================

-----------------
1. DESCRIPTION
-----------------

This project provides a from-scratch implementation of the k-Nearest Neighbors (KNN) algorithm and a 10-fold cross-validation harness. It preprocesses three datasets from the UCI Machine Learning Repository, runs the scratch KNN against Scikit-learn's standard implementation, and performs a statistical comparison of their performance.

The final results of the experiments are NOT included in the project folders by default. You must run the scripts as described below to generate the output reports.

-----------------
2. PREREQUISITES
-----------------

Before running the project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- scipy

You can install them using pip:
pip install pandas numpy scikit-learn scipy

-----------------
3. HOW TO RUN
-----------------

All operations are performed from the command line using the `knn_runner.py` script. You must first run the preprocessing step ('prep') for a dataset, followed by the experiment step ('run').

This two-step process must be completed for each of the three datasets.

**Step 1: Open a terminal or command prompt.**

**Step 2: Navigate to the project's root directory.**
   (e.g., `cd path/to/Assignment 1 Python Workspace`)

**Step 3: Run the following commands in order:**

   **For the Hayes-Roth Dataset:**
   python knn_runner.py prep hayes_roth
   python knn_runner.py run hayes_roth

   **For the Car Evaluation Dataset:**
   python knn_runner.py prep car_evaluation
   python knn_runner.py run car_evaluation

   **For the Breast Cancer Dataset:**
   python knn_runner.py prep breast_cancer
   python knn_runner.py run breast_cancer

-----------------
4. EXPECTED OUTPUT
-----------------

After running the commands, the program will generate the following files:

- **Preprocessing (`prep` command):** Creates cleaned and processed data files (`X_..._processed.csv`, `y_..._processed.csv`) inside the `data_preprocessing/` subdirectories.

- **Experiments (`run` command):** Generates the final analysis reports as `.txt` files in two folders:
  - `cross_validation_report/`: Contains detailed fold-by-fold accuracy comparisons between the from-scratch model and Scikit-learn.
  - `evaluation_test_report/`: Contains the formal hypothesis test (paired t-test) results and conclusion for each dataset.

These output folders will be empty until the scripts are run.
