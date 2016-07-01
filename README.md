# nnet

A (relatively) simple feed-forward neural network in NumPy/Cython.
Only basic layers for multi-layer perceptrons and convolutional neural networks are implemented.

This code is meant as an example implementation and for small experiments.
Its performance does not scale to large quantities of data!

Steps to run:
1. pip install virutalenv
2. git clone the repository using the command git clone 
3. cd into the repository
4. virtualenv dlw
5. source dlw/bin/activate
6. python setup.py build
7. pip install -r requirements.txt
7. python setup.py build
8. cd CNN
9. python cnn_mnist.py will start cnn training
