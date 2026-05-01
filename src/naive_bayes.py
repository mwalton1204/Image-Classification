import math


class NaiveBayesClassifier:
    def __init__(self, labels):
        self.labels = labels
        self.label_priors = {}   # P(label) "How common is each digit"
        self.feature_probs = {}  # P(feature=value | label) "Given a digit, how likely is each feature value?"

    def train(self, features_list, labels):
        num_examples = len(features_list)
        num_features = len(features_list[0])

        # Initialize counts for each label type to 0
        label_counts = {label: 0 for label in self.labels}

        # Initialize counts for each (label, feature, value) combination to 0
        feature_counts = {}
        for label in self.labels:
            feature_counts[label] = []
            for feature in range(num_features):
                if feature == num_features - 1:  # Last feature = density
                    feature_counts[label].append({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
                else: # Other features = binary pixels
                    feature_counts[label].append({0: 0, 1: 0})

        # For each image, tally its label and pixel values
        for image in range(num_examples):
            features = features_list[image]
            label = labels[image]
            label_counts[label] += 1

            # For each feature in image, record what value it had for this label
            for feature in range(num_features):
                value = features[feature]
                feature_counts[label][feature][value] += 1

        # Convert counts to probabilities
        for label in self.labels:
            # How often did this label appear out of all training examples?
            self.label_priors[label] = label_counts[label] / num_examples

            # For each feature, convert its raw counts into probabilities and store them
            self.feature_probs[label] = []
            for feature in range(num_features):
                num_values = len(feature_counts[label][feature])  # 2 for binary, 6 for density
                probs = {}

                for value, count in feature_counts[label][feature].items():
                    # +1 to count and +num_values to total avoids any probability being 0
                    probs[value] = (count + 1) / (label_counts[label] + num_values)

                self.feature_probs[label].append(probs)

    def predict(self, features):
        best_label = None
        best_score = None
        num_features = len(features)

        for label in self.labels:
            # Use log probabilities so thousands of small probabilities do not underflow to 0
            score = math.log(self.label_priors[label])

            for feature in range(num_features):
                value = features[feature]

                # log(a * b * c) = log(a) + log(b) + log(c)
                score += math.log(self.feature_probs[label][feature][value])

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict_all(self, features_list):
        return [self.predict(features) for features in features_list]


def accuracy(predictions, labels):
    correct = 0

    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1

    return correct / len(labels)