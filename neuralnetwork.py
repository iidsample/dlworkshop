import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from .layers import ParamMixin
from .helpers import one_hot, unhot


class NeuralNetwork:
    def __init__(self, layers, rng=None):
        self.layers = layers
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.tr_error = []
        self.te_error = []
        self.tr_loss = []
        
        self.filter = plt.figure()
        self.loss = plt.figure()
        # self.tr_loss_fig = plt.figure()
        # self.te_error_fig = plt.figure()
        
    def _setup(self, X, Y):
        # Setup layers sequentially
        next_shape = X.shape
        for layer in self.layers:
            layer._setup(next_shape, self.rng)
            next_shape = layer.output_shape(next_shape)
#            print(next_shape)
        if next_shape != Y.shape:
            raise ValueError('Output shape %s does not match Y %s'
                             % (next_shape, Y.shape))

    def fit(self, X, Y, X_test, Y_test, test_iter=2, learning_rate=0.1, max_iter=10, batch_size=64):
        """ Train network on the given data. """
        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size
        Y_one_hot = one_hot(Y)
        self._setup(X, Y_one_hot)
        iter = 0
        # Stochastic gradient descent with mini-batches
        while iter < max_iter:
            iter += 1
            for b in range(n_batches):
                batch_begin = b*batch_size
                batch_end = batch_begin+batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y_one_hot[batch_begin:batch_end]

                # Forward propagation
                X_next = X_batch
                for layer in self.layers:
                    X_next = layer.fprop(X_next)
                Y_pred = X_next

                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch, Y_pred)
                for layer in reversed(self.layers[:-1]):
                    next_grad = layer.bprop(next_grad)

                # Update parameters
                for layer in self.layers:
                    if isinstance(layer, ParamMixin):
                        for param, inc in zip(layer.params(),
                                              layer.param_incs()):
                            param -= learning_rate*inc

            # Output training status
            loss = self._loss(X, Y_one_hot)
            error = self.error(X, Y)
            self.tr_error.append((iter,error))
            self.tr_loss.append((iter,loss))
            print('iter %i, loss %.4f, train error %.4f' % (iter, loss, error))
            # self.call_plot()
            # if iter % test_iter == 0:
            error, cm = self.error(X_test, Y_test, conf=True)
            self.conf_mat = cm
            self.te_error.append((iter,error))
            print('iter %i, test error %.4f' % (iter, error))
                #self.call_plot()
            self.call_plot(flag = False)
                
            # visualize the filters
            W,b = self.layers[0].params()           
            self.vis_square(W.transpose(1,2,3,0),iter)
            
    def _loss(self, X, Y_one_hot):
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        Y_pred = X_next
        return self.layers[-1].loss(Y_one_hot, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        Y_pred = unhot(X_next)
        return Y_pred

    def error(self, X, Y, conf=False):
        """ Calculate error on the given data. """
        Y_pred = self.predict(X)
        error = Y_pred != Y
        if conf:
            cm = confusion_matrix(Y, Y_pred)
            return np.mean(error), cm
        return np.mean(error)

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        # Warning: the following is a hack
        Y_one_hot = one_hot(Y)
        self._setup(X, Y_one_hot)
        for l, layer in enumerate(self.layers):
            if isinstance(layer, ParamMixin):
                print('layer %d' % l)
                for p, param in enumerate(layer.params()):
                    param_shape = param.shape

                    def fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        return self._loss(X, Y_one_hot)

                    def grad_fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation
                        X_next = X
                        for layer in self.layers:
                            X_next = layer.fprop(X_next)
                        Y_pred = X_next

                        # Back-propagation of partial derivatives
                        next_grad = self.layers[-1].input_grad(Y_one_hot,
                                                               Y_pred)
                        for layer in reversed(self.layers[l:-1]):
                            next_grad = layer.bprop(next_grad)
                        return np.ravel(self.layers[l].param_grads()[p])

                    param_init = np.ravel(np.copy(param))
                    err = sp.optimize.check_grad(fun, grad_fun, param_init)
                    print('diff %.2e' % err)


    def call_plot(self, flag = False):
        # plt.close('all')
        # plt.close(fig)
        self.sp1 = self.loss.add_subplot(1,1,1)
        self.sp1.plot([i[0] for i in self.tr_error], [i[1] for i in self.tr_error], 'r')
        self.sp1.plot([i[0] for i in self.te_error], [i[1] for i in self.te_error], 'b')
        cm = self.conf_mat
        #plt.figure(3)
        if cm is not None:
            cm1 = np.asarray(cm).astype(np.float)
            for i,r in enumerate(cm1):
                cm1[i,:] = (cm1[i,:]/cm1[i,:].sum())*100.0
            # fig = plt.figure()
            plt.matshow(cm)
            plt.title('Test Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # plt.ion()
            #plt.show(block=False)
            if flag:
                plt.show(block = True)
            else:
                plt.show(block = False)
            # time.sleep(5)
            # plt.close(fig)
    def vis_square(self, data, iter, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        if data.shape[2]==1:
            data = np.tile(data,3)
        self.ax = self.filter.add_subplot(111)
        self.ax.imshow(data)
        #self.filter.show(block=False)
        self.filter.savefig('weights_'+str(iter)+'.png')
        plt.show(block = False)
        print 'Continue'
