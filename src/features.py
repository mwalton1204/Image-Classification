def pixels_binary(image):
    features = [] # Binary list of all pixels in an image

    for row in image:
        for pixel in row:
            if pixel == " ": # If unfilled
                features.append(0)
            else:
                features.append(1)

    return features

def pixels_density(image):
    filled = 0
    total = 0

    for row in image:
        for pixel in row:
            total += 1
            if pixel != " ":
                filled += 1

    return [filled / total] # Ratio of filled pixels

def extract_features(image): # Single image feature extraction
    return pixels_binary(image) + pixels_density(image) # Concats lists: [binary pixels] + [density ratio]

def extract_dataset_features(images): # Feature extractions for all images
    return [extract_features(image) for image in images]