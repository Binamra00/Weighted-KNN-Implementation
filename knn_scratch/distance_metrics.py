import math


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2) - 1):
        distance += abs(row1[i] - row2[i])
    return distance


def hamming_distance(row1, row2):
    distance = 0
    for i in range(len(row2) - 1):
        if row1[i] != row2[i]:
            distance += 1
    return distance
