# FCNAE-Tensorflow (2018/09/21)

## Introduction
I implement a tensorflow model of a Staked Autoencoder using Fully Convolutional Network. 
- I use mnist dataset as training dataset.

## Environment
- Ubuntu 16.04
- Python 3.5

## Depenency
- Numpy
- matplotlib

## Files
- fcnae.py : Model definition.
- main.py : Train the model and pass the default value.

## How to use
### Training
```shell
python main.py

# Default args: training_epoch = 200, batch_size = 128, n_layers = 3, stride = 2, learning_rate = 0.0001
# You can change args: training_epoch = 300, batch_size = 64, n_layers = 2, stride = 4, learning_rate = 0.0005
python main.py --training_epoch 300 --batch_size 64 --n_layers 2 --stride 4 --learning_rate 0.0005
```
