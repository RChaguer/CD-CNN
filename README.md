# Face Anti-Spoofing using Central Difference Convolution Neural Networks
This project was conducted by AI major students of Illinois Institute of Technology. We would like to thank Dr. Yan Yan, our supervisor for this research project and for the "Machine Learning" course. We are extremely glad for this opportunity to learn about numerous fields and expand our knowledge and expertise in many fields such as convolutional neural networks.

## Team members

- Ismail Elomari Alaoui
- Reda Chaguer

## Original research paper:
Our report is present in this depository. Additionally, the project was inspired to a certain extent from a research paper published in 2019, named: Searching Central Difference Convolutional Networks for Face Anti-Spoofing. 

## Abstract 
Face detection is a computer technology being used in a variety of applications that identifies human faces in digital images. In order to limit miscellaneous acts, the latest technology focuses on protecting this field from spoofing. Thus, a new field of AI and deep learning is born: Face anti-spoofing. 

Facial anti-spoofing is the task of preventing false facial verification by using a photo, video, mask or a different substitute for an authorized person’s face. 

## Github Depository Arborescence

This depository contains 4 folders:

- data: contains a two DataLoader classes (one for each dataset used NUAA and MSU-MSFD). It also contains some essential helper and utility functions such as normalisation and image cropping.

- model: contains 3 files

    - Layers: describing utilized custom layers such as central difference convolutional layer.
    - Loss: describes some custom losses we created such as ContrastDepthLoss. This type of loss would normally be used, in addition to MSE loss, if we choose not to use the last neural network layers, else we use bianary cross-entropy loss.
    - Model: contains the complete structure of our FAS model.

- trained_models: contains 4 trained models (3 for the NUAA dataset - Raw, FaceDetected and Normalized & 1 for MSU-MSFD). The used theta is defaulted to 0.7.

- training: defines functions used to train and benchmark the model with simplicity and transparence.

We have also 2 jupyter notebooks in this depository msu-msfd.ipynb and nuaa.ipynb, where we train and benchmark our model. In these notebooks, we train, validate, test and cross-test btween datasets our model.