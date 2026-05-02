import math

def accuracy(predictions, labels):
    correct = 0

    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1

    return correct / len(labels)

def average(values):
    return sum(values) / len(values)

def standard_deviation(values):
    avg = average(values)

    variance_sum = 0
    for v in values:
        diff = v - avg
        variance_sum += diff * diff

    variance = variance_sum / len(values)
    std_dev = math.sqrt(variance)

    return std_dev