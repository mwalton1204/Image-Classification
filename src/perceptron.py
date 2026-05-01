class PerceptronClassifier:
    def __init__(self, labels, num_features):
        self.labels = labels
        self.num_features = num_features

        # Initialize weights for each label
        self.weights = {}
        for label in labels:
            self.weights[label] = [0.0] * num_features

    def train(self, features_list, labels):
        num_examples = len(features_list)
        max_passes = 1
        num_passes = 0

        while True:
            updates = 0

            for i in range(num_examples):
                features = features_list[i]
                true_label = labels[i]

                predicted = self.predict(features)

                if predicted != true_label:
                    updates += 1

                    for j in range(self.num_features):
                        self.weights[true_label][j] += features[j]
                        self.weights[predicted][j] -= features[j]

            num_passes += 1

            # Stop if no updates happened in this full pass or if max number of allowed passes is met
            if updates == 0 or num_passes >= max_passes:
                break

    def predict(self, features):
        best_label = None
        best_score = None

        for label in self.labels:
            score = 0

            for i in range(self.num_features):
                score += self.weights[label][i] * features[i]

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict_all(self, features_list):
        return [self.predict(features) for features in features_list]