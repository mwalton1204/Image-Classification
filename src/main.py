from test import test_load_data, test_features, test_naive_bayes

def main():
    test_load_data("Digit training data", "data/digitdata/trainingimages", "data/digitdata/traininglabels")
    test_load_data("Face training data", "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels")
    test_features("Digit training data", "data/digitdata/trainingimages", "data/digitdata/traininglabels")
    test_features("Face training data", "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels")
    test_naive_bayes("Digit test data", "data/digitdata/testimages", "data/digitdata/testlabels", list(range(10)))
    test_naive_bayes("Face test data", "data/facedata/facedatatest", "data/facedata/facedatatestlabels", [0, 1])

if __name__ == "__main__":
    main()