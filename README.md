## Physics-Informed Neural Network for Option Pricing
This repository contains an implementation of a Physics-Informed Neural Network (PINN) designed to solve the Black-Scholes-Merton (BSM) equation for option pricing. The code leverages TensorFlow and other scientific libraries to build, train, and validate the PINN.

# Overview
A Physics-Informed Neural Network (PINN) is a type of neural network that incorporates physical laws into the training process. In this project, the PINN is trained to approximate the solution of the Black-Scholes-Merton equation, a fundamental partial differential equation (PDE) in financial mathematics used for pricing European call options.

# Features
Neural Network Architecture: The network is built using TensorFlow's Keras API with dense and dropout layers for regularization.
Custom Loss Function: The loss function includes the residual of the Black-Scholes-Merton equation, ensuring the solution adheres to the underlying financial model.
Training Process: The network is trained using a custom training loop with the Adam optimizer.
Visualization: Includes functionality to plot the predicted option prices against true values.

## Read the PDF
