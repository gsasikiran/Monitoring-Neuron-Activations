[![Build Status](https://travis-ci.com/gsasikiran/Monitoring-Neuron-Activations.svg?token=Tx91FqTgxZM9ucHAnbfR&branch=master)](https://travis-ci.com/gsasikiran/Monitoring-Neuron-Activations/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# Monitoring-Neuron-Activations

## Authors
- Jaswanth Bandlamudi
- Sasi Kiran Gaddipati

## Overview

A descriptive visualization of neuron activations for image classification and analyzing the robustness of the classification model by comparing the test image output and corresponding activation patterns.
![Proposed architecture](/images/architecture.png)

Project link: https://github.com/gsasikiran/Monitoring-Neuron-Activations

## Requirements

- Language : Python 3.7
  
- Coding convention : PEP08
  
- Testing framework : python unittest
  
- Data structure:
  - **Input** : Image and layer number<br>
        - Input the image to get the activation layers

  - **Output** : Image
        - Plot the images of the activation layers

- Design Patterns: Bridge pattern

## Installation and Running
- Clone the repo
> git  clone  https://github.com/gsasikiran/Monitoring-Neuron-Activations

- Run visualize_activation.py
> python  visualize_activation.py  image_path(with quotes)  layer_number

- Example command line
> python  visualize_activation.py  'images/test_image.png'  2

## Description
- The layer number is not zero-based index.
  
- The 'test_image.png' used for demostration purposes is taken from the triangles dataset[2].  
![Example](/images/example.png)

## Limitations and Future Work

- Currently it works only for 28 x 28 image. This has to be generalized for any image.
  
- As the number of activation layers, alter for model to model, the visualization of number of activation layers assists in selecting the layer number in index range.

- Monitoring for out of distribution is not achievaible with the implemented approach.


## Reference

[1] Cheng, Chih-Hong, Georg Nührenberg, and Hirotoshi Yasuoka. "Runtime monitoring neuron activation patterns." 2019 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 2019.

[2] Azmi, Mohd Sanusi, et al. "Exploiting features from triangle geometry for digit recognition." 2013 International Conference on Control, Decision and Information Technologies (CoDIT). IEEE, 2013.

