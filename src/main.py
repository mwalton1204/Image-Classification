from test import test_load_data, test_features

def main():
    test_load_data("Digit training data", "data/digitdata/trainingimages", "data/digitdata/traininglabels")
    test_load_data("Face training data", "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels")

    test_features("Digit training data", "data/digitdata/trainingimages", "data/digitdata/traininglabels")
    test_features("Face training data", "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels")

if __name__ == "__main__":
    main()