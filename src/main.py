from load_data import load_data
from evaluation import run_experiments

def main():
    DIGIT_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Possible labels for digits
    FACE_LABELS = [0, 1] # Possible labels for faces

    # Load digit training data
    digit_train_images, digit_train_labels = load_data(
        "data/digitdata/trainingimages",
        "data/digitdata/traininglabels"
        )

    # Load digit testing data
    digit_test_images, digit_test_labels = load_data(
        "data/digitdata/testimages",
        "data/digitdata/testlabels"
        )

    # Load face training data
    face_train_images, face_train_labels = load_data(
        "data/facedata/facedatatrain",
        "data/facedata/facedatatrainlabels"
        )

    # Load face testing data
    face_test_images, face_test_labels = load_data(
        "data/facedata/facedatatest",
        "data/facedata/facedatatestlabels"
        )

    # Run experiments for digits
    run_experiments("Digit",
                    digit_train_images,
                    digit_train_labels,
                    digit_test_images,
                    digit_test_labels,
                    DIGIT_LABELS
                    )
    
    # Run experiments for faces
    run_experiments("Face",
                    face_train_images,
                    face_train_labels,
                    face_test_images,
                    face_test_labels,
                    FACE_LABELS
                    )

if __name__ == "__main__":
    main()