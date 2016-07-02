from tempfile import TemporaryFile
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
tr_d, va_d,te_d = mnist_loader.load_data()
import network
net =network.Network([784,64,32,10]) #network architecture design
np.savez('weightsinitial',weights=net.weights)
net.SGD(training_data, 1,10,.1 )
np.savez('weightsafter1',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter10',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter20',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter30',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter40',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter50',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter60',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter70',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter80',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter90',weights=net.weights)
net.SGD(training_data, 10,10,0.1 )
np.savez('weightsafter100',weights=net.weights)
test_x ,test_y = te_d #this is for displaying the image
test_i = [x[0] for x in test_data] #this has to be sent to evaluate function
test_j = [x[1] for x in test_data] #this contains validation input
res = np.argmax(net.feedforward(test_i[2]))
print test_j[2]
print res
plt.imshow(test_i[2].reshape((28,28)), cmap = cm.Greys_r)
plt .show()
