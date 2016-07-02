# nnet

A (relatively) simple feed-forward neural network in NumPy/Cython.
Only basic layers for multi-layer perceptrons and convolutional neural networks are implemented.

This code is meant as an example implementation and for small experiments.
Its performance does not scale to large quantities of data!

Steps to run programs
Option 1(both OS X and linux):
1. pip install virutalenv
2. git clone the repository using the command git clone 
3. cd into the repository
4. virtualenv dlw
5. source dlw/bin/activate
6. pip install -r requirements.txt
7. python setup.py build

Option 2(linux):
1. sudo apt-get install python-dev
2. sudo apt-get install python-numpy
3. sudo apt-get install cython
4. sudo apt-get install python-scipy
5. sudo apt-get install python-matplotlib
6. python setup.py build

This should create build folder in the root folder

Running fuly connected networks:
1. cd fullyconnected
2. run wrapper.py
3. This will start training the feedforward network.
4. It will start training in batches.
5. The files will also save receptive fields as npz files
6. run rf.py 'nameofweighfile.npz' 'nameofweightfile.npz' ...

Running RBM code:
1. cd rbm
2. python rbm_multiple_layers.py
3. This will show after training reconstrunction of output
4. python rbm_svm.py 
5. This will build a classification model of svm over rbm

Running CNN code:
1. cd CNN
2. python cnn_mnist.py 
3. This will start the cnn training


