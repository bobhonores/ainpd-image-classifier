# AIPND Image classifier project
This repo contains my solution for Udacity's AIPND final project of image classifier.

It is a command line utility with to main script:
- train
- predict

## train
It helps you to create a trained model. Also, it allows you to customize this model since enables you to enter multiple parameters for it.

* Establish a directory in which the model will be saved.
* Establish the architecture of the model.
* Establish the hidden layers of your model.
* Set the learning rate and epochs to execute in the training process.
* Set the use of GPU for this process.

The current supported architectures are:

* resnet34
* resnet50
* restnet101
* densenet161
* densenet169
* vgg19
* vgg16

## predict
Once you have a trained model, you could establish predictions based on it. This script allows you to do that. Just like the train part, here we have a set of multiple parameters which will enabled you to perform the following actions.

* Indicate the path of the file to evaluate.
* Establish the path in which the model file was stored.
* Retreive the top KK most likely classes.
* Load a file for mapping categories to different names.
* Set the use of GPU for this process.