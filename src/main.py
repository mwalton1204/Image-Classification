from test import test_load_data, test_features, test_naive_bayes, test_perceptron


def main():
    digit_train_images = "data/digitdata/trainingimages"
    digit_train_labels = "data/digitdata/traininglabels"
    digit_test_images = "data/digitdata/testimages"
    digit_test_labels = "data/digitdata/testlabels"

    face_train_images = "data/facedata/facedatatrain"
    face_train_labels = "data/facedata/facedatatrainlabels"
    face_test_images = "data/facedata/facedatatest"
    face_test_labels = "data/facedata/facedatatestlabels"

    digit_labels = list(range(10))
    face_labels = [0, 1]

    test_load_data("Digit training", digit_train_images, digit_train_labels)
    test_load_data("Face training", face_train_images, face_train_labels)

    test_features("Digit training", digit_train_images, digit_train_labels)
    test_features("Face training", face_train_images, face_train_labels)

    test_naive_bayes(
        "Digit",
        digit_train_images,
        digit_train_labels,
        digit_test_images,
        digit_test_labels,
        digit_labels
    )

    test_naive_bayes(
        "Face",
        face_train_images,
        face_train_labels,
        face_test_images,
        face_test_labels,
        face_labels
    )

    test_perceptron(
        "Digit",
        digit_train_images,
        digit_train_labels,
        digit_test_images,
        digit_test_labels,
        digit_labels
    )

    test_perceptron(
        "Face",
        face_train_images,
        face_train_labels,
        face_test_images,
        face_test_labels,
        face_labels
    )


if __name__ == "__main__":
    main()