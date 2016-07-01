# import cPickle
# import gzip
# import numpy as np
# def load_data():
#   f = gzip.open('mnist.pkl.gz','rb')
#   training_data, validation_data, test_data  =cPickle.load(f)
#   f.close()
#   return (training_data,validation_data,test_data)
# def load_data_wrapper():
#     tr_d, va_d, te_d = load_data()
#     training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
#     training_results = [vectorized_result(y) for y in tr_d[1]]
#     training_data = zip(training_inputs, training_results)
#     validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
#     validation_data = zip(validation_inputs, va_d[1])
#     test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
#     test_data = zip(test_inputs, te_d[1])
#     return (training_data, validation_data, test_data)
from tempfile import TemporaryFile
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
tr_d, va_d,te_d = mnist_loader.load_data()
import network
net =network.Network([784,64,32,10])
#weightsinitial = TemporaryFile()
#weightsafter1 = TemporaryFile()
#weightsafter10 = TemporaryFile()
np.savez('weightsinitial',weights=net.weights)
# f = open('biasdump.txt', 'a')
# f.write('biases = ' + repr(net.biases) + '\n')
# f.close()
# f1 = open('weight0.txt', 'a')
# f1.write('weights = ' +repr(net.weights) + '\n')
# f1.close()
net.SGD(training_data, 1,10,.1 )
np.savez('weightsafter1',weights=net.weights)
# f = open('biasesafter.txt','a')
# f.write('biases = ' +repr(net.biases) + '\n')
# f.close()
# f1 = open('weightsafter1.txt' ,'a')
# f1.write('weights = ' + repr(net.weights) + '\n')
# f1.close()
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
# f = open('biasesafter10.txt','a')
# f.write('biases = ' +repr(net.biases) + '\n')
# f.close()
# f1 = open('weightsafter10.txt' ,'a')
# f1.write('weights = ' + repr(net.weights) + '\n')
# f1.close()
#print len(training_data)
#print len(validation_data)
#print len(test_data)
test_x ,test_y = te_d #this is for displaying the image
test_i = [x[0] for x in test_data] #this has to be sent to evaluate function
test_j = [x[1] for x in test_data] #this contains validation input
res = np.argmax(net.feedforward(test_i[2]))
print test_j[2]
print res
plt.imshow(test_i[2].reshape((28,28)), cmap = cm.Greys_r)
plt .show()
