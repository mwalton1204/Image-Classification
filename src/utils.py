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
        diff = v - avg # Difference from the mean
        variance_sum += diff * diff # Squared difference from the mean

    variance = variance_sum / len(values) # Average of squared differences from the mean
    std_dev = math.sqrt(variance) # Square root of the average of squared differences from the mean

    return std_dev