# Perceptron Classfier Implementation
Code suite with Python __version__ 3.8
- Author: Chuan Qin


## Requirements
- numpy>=1.12.1
- pandas>=0.19.2
- matplotlib>=3.0.1

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

# Input Data:
Fashin MNIST data (https://github.com/zalandoresearch/fashion-mnist).

# Run main code *BinaryPerceptron.py*:
Code can be execute directly from terminal.

# Notations:
## input variables:
D = training examples,
T = interation,
k = number of classes.

# parameters:
w = weights,
Foobar is a Python library for dealing with word pluralization.

## Installation
No need of installation

## Usage
models.py contains model classes, defined as below:

''' Binary classifier'''
class BinaryPerceptron() # Binary classifier with perceptron 
class BinaryPA() # Binary classifier with passive-aggressive
class BinaryAveragedPerceptron() # Binary classifier with averaged perceptron 

''' Multi-class classifier'''
class MCPerceptron() # Multi-class classifier with perceptron 
class MCPA() # Multi-class classifier with Passive-Aggressive (PA) 
class MCAveragedPerceptron() # Multi-class classifier with perceptron 

# License
Free
