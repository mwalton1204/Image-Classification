import random
import time

from features import extract_dataset_features
from naive_bayes import NaiveBayesClassifier
from perceptron import PerceptronClassifier
from utils import accuracy, average, standard_deviation

# Return a random subset of the training data based on the given percentage
def get_sample(features, labels, percent):
    sample_size = int(len(features) * percent)
    indexes = random.sample(range(len(features)), sample_size)

    sample_features = []
    sample_labels = []

    for index in indexes:
        sample_features.append(features[index])
        sample_labels.append(labels[index])

    return sample_features, sample_labels

# Train and test chosen model using random subset of the training data
def test_model(model_name, train_features, train_labels, test_features, test_labels, labels, percent):
    sample_features, sample_labels = get_sample(train_features, train_labels, percent)

    start_time = time.time()

    # Create model object
    if model_name == "Naive Bayes":
        model = NaiveBayesClassifier(labels)
    else:
        model = PerceptronClassifier(labels, len(train_features[0]))

    # Train model model
    model.train(sample_features, sample_labels)

    # Get predictions and compare them to truth
    predictions = model.predict_all(test_features)
    acc = accuracy(predictions, test_labels)

    end_time = time.time()
    runtime = end_time - start_time

    return acc, runtime

# Perform pre-defined qty of trials across 10 training set sizes (10% - 100%) for each classifier
# Print average accuracy, standard deviation, and runtimes
def run_experiments(name, train_images, train_labels, test_images, test_labels, labels):
    train_features = extract_dataset_features(train_images)
    test_features = extract_dataset_features(test_images)

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    classifiers = ["Naive Bayes", "Perceptron"]
    trials = 5

    print()
    print(f"{name} Results ({trials} trials):")

    for percent in percentages:
        for classifier in classifiers:
            accuracies = []
            runtimes = []

            # Run multiple trials (randomly selected data points each time)
            for trial in range(trials):
                acc, runtime = test_model(
                    classifier,
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    labels,
                    percent
                )

                accuracies.append(acc)
                runtimes.append(runtime)

            avg_acc = average(accuracies)
            std_acc = standard_deviation(accuracies)
            avg_time = average(runtimes)

            print(
                f"{int(percent * 100)}% | {classifier} | "
                f"Accuracy: {round(avg_acc * 100, 2)} | "
                f"Std Dev: {round(std_acc * 100, 2)} | "
                f"Time: {round(avg_time, 4)}"
            )