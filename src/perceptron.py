class PerceptronClassifier:
    # Training:
    # 1. Predict a label using the current weights
    # 2. If prediction is wrong, add the feature vector to the correct label's weights and subtract from wrong label's weights
    #
    # Prediction:
    # For each label, compute:
    # score = weight vector · feature vector
    # Return the label with the highest score

    def __init__(self, labels, num_features):
        self.labels = labels  # Set of possible classes
        self.num_features = num_features

        # One weight vector per label
        self.weights = {}
        for label in labels:
            self.weights[label] = [0.0] * num_features # Initialize all weights to 0.0

    def train(self, features_list, labels):
        num_examples = len(features_list)
        max_passes = 1 # How many full passes over the training data
        num_passes = 0

        while True:
            updates = 0

            # One full pass over the training data
            for i in range(num_examples):
                features = features_list[i]
                true_label = labels[i]

                predicted = self.predict(features)

                if predicted != true_label:
                    updates += 1

                    # Update weights
                    for j in range(self.num_features):
                        self.weights[true_label][j] += features[j] # Add the feature vector to the correct label's weights
                        self.weights[predicted][j] -= features[j] # Subtract the feature vector from the incorrect label's weights

            num_passes += 1

            # Stop if no mistakes were made, or after the max allowed passes
            if updates == 0 or num_passes >= max_passes:
                break

    def predict(self, features):
        best_label = None
        best_score = None

        # Score every possible label and choose the highest
        for label in self.labels:
            score = 0

            # Dot product:
            # score = w1*x1 + w2*x2 + ... + wn*xn
            # Higher score means better match for current weights
            for i in range(self.num_features):
                score += self.weights[label][i] * features[i] # self.weights[label][i] = wi; features[i] = xi

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict_all(self, features_list):
        return [self.predict(features) for features in features_list]