def load_labels(label_file):
    labels_list = []

    with open(label_file, "r") as l:
        for line in l:
            labels_list.append(int(line.strip()))
    
    return labels_list

def load_images(image_file, num_images):
    images = [] # 2D array, holds individual images

    with open(image_file, "r") as i:
        lines = i.readlines()

    height = len(lines) // num_images

    for i in range(num_images):
        image = [] # 2D array, holds lines of a single image
        start = i * height # Start at next unprocessed image

        for row_num in range(height):
            line = lines[start + row_num].rstrip("\n") # Get the next line
            image.append(list(line)) # Add line to image array

        images.append(image) # Add image to images array

    return images

def load_data(image_file, label_file):
    labels = load_labels(label_file)
    images = load_images(image_file, len(labels))

    return images, labels