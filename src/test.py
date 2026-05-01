from load_data import load_data
from features import pixels_binary, pixels_density, extract_features, extract_dataset_features
from naive_bayes import NaiveBayesClassifier, accuracy

def test_load_data(name, image_file, label_file):
    images, labels, width, height = load_data(image_file, label_file)
    print("\n")
    print(name, "test")
    print("------")
    print("Number of images:", len(images))
    print("Number of labels:", len(labels))
    print("Width:", width)
    print("Height:", height)

def test_features(name, image_file, label_file):
    images, labels, width, height = load_data(image_file, label_file)
    first_image = images[0]
    binary_features = pixels_binary(first_image)
    density_feature = pixels_density(first_image)
    combined_features = extract_features(first_image)
    dataset_features = extract_dataset_features(images)
    print("\n")
    print(name, "feature test")
    print("------")
    print("Binary feature length:", len(binary_features))
    print("Expected binary length:", width * height)
    print("Density feature value (first image):", density_feature[0])
    print("Density feature length:", len(density_feature))
    print("Combined feature length:", len(combined_features))
    print("Expected combined length:", width * height + len(density_feature))
    print("Qty feature vectors produced:", len(dataset_features))

def test_naive_bayes(name, image_file, label_file, labels):
    images, true_labels, width, height = load_data(image_file, label_file)
    features = extract_dataset_features(images)
    classifier = NaiveBayesClassifier(labels)
    classifier.train(features, true_labels)
    predictions = classifier.predict_all(features)
    acc = accuracy(predictions, true_labels)
    print("\n")
    print(name, "naive bayes test")
    print("------")
    print("Accuracy:", round(acc * 100, 2), "%")