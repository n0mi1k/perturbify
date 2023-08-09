from fgsm import *
from check import *
from deepfool import *
import argparse
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import shutil
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parseOptions(options, attackMethod):
    settings = options.split()
    if attackMethod == "fgsm":
        print(f"[+] FGSM Epsilon: {settings[0]}, Index: {settings[1]}")
        return settings[0], settings[1]
    else:
        print(f"[+] Deepfool MaxIter: {settings[0]}, PertMultiplier: {settings[1]}")
        return settings[0], settings[1]

def main():
    print("""  _____          _              _     _  __       
 |  __ \        | |            | |   (_)/ _| v1.0     
 | |__) |__ _ __| |_ _   _ _ __| |__  _| |_ _   _ 
 |  ___/ _ \ '__| __| | | | '__| '_ \| |  _| | | |
 | |  |  __/ |  | |_| |_| | |  | |_) | | | | |_| |
 |_|   \___|_|   \__|\__,_|_|  |_.__/|_|_|  \__, |
                By github.com/n0mi1k         __/ |
                                            |___/ """)
    parser = argparse.ArgumentParser(prog='adversarial', 
                                    description='A adversarial machine learning tool in Python',
                                    usage='%(prog)s -m manipulate -i IMAGE -m MODEL -a ADVERSARIAL -o OPTIONS')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Image file to manipulate", required=True)
    parser.add_argument("-m", '--model', help="Trained ML model to validate", required=True)
    parser.add_argument("-a", '--adversarial', help="Type of adversarial attack to use (fgsm)", required=True)
    parser.add_argument("-o", '--options', help="Custom values for adversarial", required=True)
    args = parser.parse_args()

    image_path = args.image
    model = args.model
    attack = args.adversarial
    options = args.options

    # Add on to list for other attacks 
    attackMethods = ["fgsm", "deepfool"]

    print(f"[*] Image to Manipulate: {image_path}")
    print(f"[*] Using Model: {model}")
    print(f"[*] Adversarial Method: {attack}")

    if attack not in attackMethods:
        print(f"[-] Invalid {attack} adversarial method used!")
        exit()

    if attack == attackMethods[0]:
        mpl.rcParams['figure.figsize'] = (8, 8)
        mpl.rcParams['axes.grid'] = False
        epsilon, modelindex = parseOptions(options, attackMethods[0])
        trained_model = tf.keras.models.load_model(model)

        # Modify the following variables to identify the best adversarial results
        model_index = int(modelindex)
        epsilons = [float(epsilon)]
        # Preprocess the image
        image = load_image(image_path)
        preprocessed_image = preprocess(image)

        # Get the number of output classes in your model
        num_classes = trained_model.output.shape[-1]

        # Get the input label of the image.
        label = tf.one_hot(model_index, num_classes)
        label = tf.reshape(label, (1, num_classes))

        # Create the adversarial pattern
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        perturbations = create_adversarial_pattern(loss_object, preprocessed_image, label, trained_model)

        for i, eps in enumerate(epsilons):
            adv_x = preprocessed_image + eps * perturbations
            adv_x = tf.clip_by_value(adv_x, 0, 1)
            adv_x_masked = apply_mask(adv_x, create_mask(preprocessed_image, 0.1)) # Modify the 0.1 to our preference, higher leads to more selective mask, lower is less restrictive means more manipulation
            save_images(image_path, adv_x_masked, i+1)

        images = []

        cleanName = os.path.splitext(image_path)[0]
        for filename in os.listdir(os.getcwd()):
            # Check if the filename starts with "cloudy1"
            if filename.startswith(cleanName):
                # Add the filename to the list
                images.append(filename)
        
        for image in images:
            strippedImageExt = os.path.splitext(image)[0]
            print(f"[+] Checking {strippedImageExt}.jpg:")
            command = ["python", "check.py", "-i", f"{strippedImageExt}.jpg", "-m", model]
            output = subprocess.check_output(command, universal_newlines=True)
            print(output)

    elif attack == attackMethods[1]:
        # Load your trained model
        trained_model = tf.keras.models.load_model(model)
        iterations, pertMulti = parseOptions(options, attackMethods[1])

        # Define number of classes
        num_classes = 3  # Modify this based on your model's output

        # Load and preprocess the input image
        image = plt.imread(image_path)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0  # Normalize pixel values to [0, 1]

        # Generate adversarial example using DeepFool attack
        adversarial_image = deepfool_attack(image, trained_model, num_classes, int(iterations), int(pertMulti))

        # Save the adversarial image
        cleanName = os.path.splitext(image_path)[0]
        advName = f"{os.path.splitext(image_path)[0]}_deepfool.jpg" 
        plt.imsave(f'{advName}.jpg', np.clip(adversarial_image, 0, 1))

        print(f"[+] Checking {cleanName}.jpg:")
        command = ["python", "check.py", "-i", f"{cleanName}.jpg", "-m", model]
        output = subprocess.check_output(command, universal_newlines=True)
        print(output)

        print(f"[+] Checking {advName}")
        command = ["python", "check.py", "-i", f"{advName}.jpg", "-m", model]
        output = subprocess.check_output(command, universal_newlines=True)
        print(output)

    else:
        print("Attack method not found!")
        exit()


if __name__ == "__main__":
    main()