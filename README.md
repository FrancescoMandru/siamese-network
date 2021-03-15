# Siamese Neural Network ![](sc.png) 

## Usage

A siamese network is used to perform image similarity. It uses the same weights working in tandem on two inputs at the same time. In this specific implementation we have at the last layer a sigmoid activation function with just one neuron to compute the similarity/dissimilarity score.

## Instructions
- Clone the repository.
- This is a particular network which work on a couple of images at time, so the structure is different from standard networks. At the base siamese leg
there is a pre trained VGG16 model. Instead of VGG, you can decide in `train.py` to use a base CNN as a leg for the siamese which will be trained. The model will be saved in the **saved_model** directory. Logs of the Tensorboard will be saved in the **logs** directory.
- You need to build your dataset as follows:

    ```bash
    ├── ...
    ├── data                          # dataset
    │   ├── category0                 # category directory
    │   │   ├──imgcat_0_one.png       # first image of the couple
    │   │   ├──imgcat_0_two.png       # second image of the couple
    │   │
    │   ├── category1                 # category directory
    │   │   ├──imgcat_0_one.png       # first image of the couple
    │   │   ├──imgcat_0_two.png       # second image of the couple
    │   └── ...             
    └── ...
    ```

where **imgcat_0_one.png** is the first image of the first couple regarding category **X**. The negative samples will be created online using the scripts in `dataset_utils.py`. In the `train.py` you can choose the number of negative sample to create by tuning **neg_factor** parameter. 
- **Training**: run `python train.py`
- **Evaluation**: run `python test.py`. Actually you need to add the path of two image that you want to test

## Requirements

This network has been tested using Tensorflow 2.4 on Ubuntu 18.04 with CUDA 11.0 and cuDNN 8. The packages required are listed in the `requirements.txt`. Please refer at https://www.tensorflow.org/install/gpu to setup your machine properly.


## Metrics estimation

To track the goodness of our network we look at the binary accuracy, precision, recall, AUC, Recal at precision (80%, 90% and 95%). These can be monitored using the Tensorboard. 
