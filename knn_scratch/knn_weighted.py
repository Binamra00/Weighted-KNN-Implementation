def get_neighbors(train_dataset, test_row, k, distance_metric):
    distances = []
    for train_row in train_dataset:
        dist = distance_metric(test_row, train_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])

    return distances[:k]


def predict_classification(train_dataset, test_row, k, distance_metric):
    neighbors_with_distances = get_neighbors(train_dataset, test_row, k, distance_metric)

    if not neighbors_with_distances:
        return None

    class_weights = {}
    for neighbor_row, distance in neighbors_with_distances:
        # Handle zero division for the weight metric
        if distance == 0:
            return neighbor_row[-1]

        weight = 1 / distance
        label = neighbor_row[-1]

        class_weights[label] = class_weights.get(label, 0.0) + weight

    if not class_weights:
        # Handle edge cases for an empty class_weights dictionary
        return neighbors_with_distances[0][0][-1]

    prediction = max(class_weights, key=class_weights.get)
    return prediction
