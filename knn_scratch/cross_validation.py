import random


def cross_validation_split(dataset, k_folds):
    dataset_copy = list(dataset)
    random.shuffle(dataset_copy)

    folds = []
    fold_size = len(dataset_copy) // k_folds
    remainder = len(dataset_copy) % k_folds

    start_index = 0
    for i in range(k_folds):
        current_fold_size = fold_size + 1 if i < remainder else fold_size

        end_index = start_index + current_fold_size
        fold = dataset_copy[start_index:end_index]
        folds.append(fold)

        start_index = end_index

    return folds


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, k_folds, k, distance_metric, *args):
    folds = cross_validation_split(dataset, k_folds)
    scores = list()

    for i, fold in enumerate(folds):
        train_set = list(folds)
        train_set.pop(i)
        train_set = sum(train_set, [])

        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = list()
        for row in test_set:
            prediction = algorithm(train_set, row, k, distance_metric, *args)
            predicted.append(prediction)

        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        print(f'Fold {i + 1}/{k_folds} Accuracy: {accuracy:.2f}%')

    return scores
