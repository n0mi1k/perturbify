# perturbify
perturbify is a Tensorflow adversarial machine learning attack toolkit to add perturbations and cause image recognition models to misclassify an image. The goal of an adversarial attack is to introduce minumum perturbations to an image to avoid detection while effectively causing a misclassification or significant change in prediction. Such attacks can be performed in a white, grey or black box approach. 

**NOTE:** In this toolkit, fgsm (targeted) and deepfool attacks have been implemented

## Usage
Our Tensorflow surrogate model is trained using Tensorflow and Keras. Perturbify takes in an input image and outputs the perturbed image.   
`check.py` is automatically invoked once the adversarial image is generated to display the predictions before and after perturbations are added.

```
Usage:
  perturbify.py -i [Input Image] -m [surrogate model] -a [attack method] -o [attack options]

Flags:
  -i, --image string        The target input image [Required]
  -m, --model string        The surrogate / attacker model [Required]
  -a, --adversarial         Adversarial attack type (fgsm/deepfool) [Required]
  -o, --options             Adversarial attack options (Refer below) [Required]
  -h, --help                Display the help page
```

FGSM Options: <epsilon (float), classification index to perturb towards (int)>  
`python perturbify.py -i image.jpg -m surrogate_model.h5 -a fgsm -o  "0.2 2"`  

Deepfool Options: <max iterations (int), perturbations multiplier (int)>  
`python perturbify.py -i image.jpg -m surrogate_model.h5 -a deepfool -o  "50 1"`

**NOTE:** Some attributes can only be modified on the script such as `num_classes`, read the added comments

## Dependencies
The requirements.txt file should list all Python libraries required, and they will be installed using:  
`pip install -r requirements.txt`

## Disclaimer
This tool is for educational and testing purposes only. Do not use it to exploit the vulnerability on any system that you do not own or have permission to test. The authors of this script are not responsible for any misuse or damage caused by its use.