import argparse
import cv2
import numpy as np
import os
from tensorflow import keras


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", help="Image file to manipulate", required=True)
    parser.add_argument("-m", "--model", help="model name", required=True)
    args = parser.parse_args()

    # Load and preprocess the image
    image_path = args.image_path
    modelName = args.model

    if not os.path.isfile(image_path):
        print("Error: Image file not found!")
        exit()

    image_height, image_width = 224, 224

    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add a batch dimension

    # Load the saved model
    model = keras.models.load_model(modelName)

    # Make predictions on the image
    predictions = model.predict(image)

    # Get the predicted class labels and their percentage matches
    class_names = ["Cloudy", "Rainy", "Sunny"]
    top_classes = predictions.argsort()[0, ::-1][:3]  # Get indices of top 3 classes
    top_scores = predictions[0, top_classes] * 100

    print("Predicted weather conditions:")
    for i in range(len(top_classes)):
        class_index = top_classes[i]
        class_label = class_names[class_index]
        class_score = top_scores[i]
        print("- {} ({:.2f}% match)".format(class_label, class_score))
