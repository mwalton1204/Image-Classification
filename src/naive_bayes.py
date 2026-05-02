import math

class NaiveBayesClassifier:
    # Training:
    # 1. label_priors[label] = P(y)
    # 2. feature_probs[label][feature][value] = P(xi = value | y)
    #
    # Prediction:
    # For each label y, compute:
    # log(P(y)) + Σ log(P(xi | y))
    # Return the label with the highest score

    def __init__(self, labels):
        self.labels = labels  # Set of possible classes
        self.label_priors = {}  # P(label)
        self.feature_probs = {}  # P(feature=value | label)

    def train(self, features_list, labels):
        num_examples = len(features_list)
        num_features = len(features_list[0])

        # Initialize counts for each label
        label_counts = {label: 0 for label in self.labels}

        # Frequency table used to estimate probabilities:
        # feature_counts[label][feature][value] = count
        # "For this label, how often did this feature take this value?"
        #
        # Binary pixels -> values {0,1}
        # Density feature -> values {0–5}
        feature_counts = {}
        for label in self.labels:
            feature_counts[label] = []
            for feature in range(num_features):
                if feature == num_features - 1:  # Last feature = density
                    feature_counts[label].append({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
                else:  # Other features = binary pixels
                    feature_counts[label].append({0: 0, 1: 0})

        # Build frequency tables from training data
        for image in range(num_examples):
            features = features_list[image]
            label = labels[image]
            label_counts[label] += 1

            # Count this feature value under this image's known label
            for feature in range(num_features):
                value = features[feature]
                feature_counts[label][feature][value] += 1

        # Convert counts to probabilities
        for label in self.labels:
            # Smoothed prior P(y), prevents log(0) if a random sample misses a label
            self.label_priors[label] = (label_counts[label] + 1) / (num_examples + len(self.labels))

            # Convert raw counts into conditional probabilities P(xi | y)
            self.feature_probs[label] = []
            for feature in range(num_features):
                num_values = len(feature_counts[label][feature])  # 2 for binary, 6 for density
                probs = {}

                # Laplace smoothing:
                # Add 1 to avoid zero probabilities.
                # If P(xi | y) = 0 -> log(0) = -infinity -> score breaks
                for value, count in feature_counts[label][feature].items():
                    probs[value] = (count + 1) / (label_counts[label] + num_values)

                self.feature_probs[label].append(probs)

    def predict(self, features):
        best_label = None
        best_score = None
        num_features = len(features)

        # Score each possible label and choose the highest
        for label in self.labels:
            # Naive Bayes:
            # P(y) * ∏ P(xi | y)
            #
            # Using logs:
            # log(P(y) * ∏ P(xi | y))
            # log(P(y)) * log(∏ P(xi | y))
            #
            # Product rule for logs [log(a * b) = log(a) + log(b)]:
            # log(P(y)) + Σ log(P(xi | y))
            #
            # Converts multiplication -> addition (prevents underflow)

            score = math.log(self.label_priors[label])

            for feature in range(num_features):
                value = features[feature]
                score += math.log(self.feature_probs[label][feature][value])

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict_all(self, features_list):
        return [self.predict(features) for features in features_list]

def accuracy(predictions, labels):
    # Fraction of correct predictions
    correct = 0

    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1

    return correct / len(labels)