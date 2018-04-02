# Colorization

We develop a fully automatic approach that can bring realistic color to those grayscale images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
python 3.6

Tensorflow 1.6

scikit-image 0.14

### File Explaination
read_FlowersDataset.py: download the flower dataset to the folder, and randomly split into train, validation and test data.

BatchDatsetReader.py: resize images and convert images to LAB color format, allows training to get next batch of training data.

TensorflowUtils.py: construct tensorflow layers, weights and bias, write summary for tensorboard, download vgg19 models.

main.py: build network and train the model

evaluation.py: restore the trained model, predict validation and test data, and calculate the loss.

compare_image.ipynb: compare result of prediction with grayscale image and ground truth image
