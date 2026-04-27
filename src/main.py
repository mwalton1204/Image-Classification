from load_data import load_data

def test_load_data(name, image_file, label_file):
    images, labels, width, height = load_data(image_file, label_file)

    print("\n")
    print(name)
    print("------")
    print("Number of images:", len(images))
    print("Number of labels:", len(labels))
    print("Width:", width)
    print("Height:", height)

def main():
    test_load_data("Digit training data", "data/digitdata/trainingimages", "data/digitdata/traininglabels")

    test_load_data("Face training data", "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels")

if __name__ == "__main__":
    main()