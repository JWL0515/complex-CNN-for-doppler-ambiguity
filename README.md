# Moment-Not-Open-Source
Due to possible publishing reasons, it is not open source moment.

## Introduction
This repo is based on my Diplom thesis. The goal of my thesis is solving the doppler ambiguity problem. That means using complex range-Doppler maps that contain complex numbers to train CNN models. To achieve this, we need CNN which is applicable to complex numbers. The code in order complexPyTorch is based on the open source complexPyTorch LINK. 

## Installation

pip install 

## Added functions in complexPyTorch

The orginal is provided by [wavefrontshaping]([https://github.com/wavefrontshaping/complexPyTorch](https://github.com/wavefrontshaping)). More details about complexPyTorch is [here](https://github.com/wavefrontshaping/complexPyTorch)

Added functions by me:
- in complexFunctions.py:
  - complex_dropout3d
  - complex_avg_pool3d
  - _retrieve_elements_from_indices_3D
  - complex_max_pool3d
  - complex_leaky_relu

- in complexLayers.py
  - ComplexConv3d
  - NaiveComplexBatchNorm3d
  - ComplexDropout3d
  - ComplexAvgPool3d
  - ComplexMaxPool3d

## Dataset
Eample of the raw Dataset is on https://ieee-dataport.org/documents/doppler-ambiguity-dataset. All data is generated using Matlab to simulate a 77GHz FMCW millimeter-wave radar sensing in the road scenario.

## prepare_dataset.py
This script provides various functions to preprocess the raw dataset.
Provided functions are:


## baseline.py	
This script shows the structure of CNN and the training- and testing phase. 
