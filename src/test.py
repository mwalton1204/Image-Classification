from load_data import load_data
from features import pixels_binary, pixels_density, extract_features, extract_dataset_features
from naive_bayes import NaiveBayesClassifier, accuracy
from perceptron import PerceptronClassifier

def test_load_data(name, image_file, label_file):
    images, labels, width, height = load_data(image_file, label_file)

    print("\n")
    print(name, "data test")
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
    print("Density feature value:", density_feature[0])
    print("Density feature length:", len(density_feature))
    print("Combined feature length:", len(combined_features))
    print("Expected combined length:", width * height + len(density_feature))
    print("Qty feature vectors produced:", len(dataset_features))

def test_naive_bayes(name, train_image_file, train_label_file, test_image_file, test_label_file, labels):
    train_images, train_labels, train_width, train_height = load_data(train_image_file, train_label_file)
    test_images, test_labels, test_width, test_height = load_data(test_image_file, test_label_file)

    train_features = extract_dataset_features(train_images)
    test_features = extract_dataset_features(test_images)

    classifier = NaiveBayesClassifier(labels)
    classifier.train(train_features, train_labels)

    predictions = classifier.predict_all(test_features)
    acc = accuracy(predictions, test_labels)

    print("\n")
    print(name, "naive bayes test")
    print("------")
    print("Training examples:", len(train_features))
    print("Testing examples:", len(test_features))
    print("Accuracy:", round(acc * 100, 2), "%")

def test_perceptron(name, train_image_file, train_label_file, test_image_file, test_label_file, labels):
    train_images, train_labels, train_width, train_height = load_data(train_image_file, train_label_file)
    test_images, test_labels, test_width, test_height = load_data(test_image_file, test_label_file)

    train_features = extract_dataset_features(train_images)
    test_features = extract_dataset_features(test_images)

    classifier = PerceptronClassifier(labels, len(train_features[0]))
    classifier.train(train_features, train_labels)

    predictions = classifier.predict_all(test_features)
    acc = accuracy(predictions, test_labels)

    print("\n")
    print(name, "perceptron test")
    print("------")
    print("Training examples:", len(train_features))
    print("Testing examples:", len(test_features))
    print("Accuracy:", round(acc * 100, 2), "%")