"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#Libraries
import numpy as np
import random


# Miscellaneous Functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# Main Network Class
class Network:
    def __init__(self, sizes):
        """defining network parameters"""
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip (sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Returns the output of the network if "a" is the input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)  # we make sure dot product is valid: w*a = [6x3]*[3x1] = [6x1]
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Defining the "Stochastic Gradient Descent" Back Propagation algorithm.
        training_data - list of tuples "(x,y)" representing the training inputs and the desired outputs.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the networks weights and biases by applying BackPropagation to a single mini_batch.
        mini_batch is a list of tuples "(x, y)".
        eta is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """The actual BackPropagation algorithm.
        returns "delta_nabla_b" and "delta_nabla_w" which are layer-by-layer list of numpy arrays,
         similar to the biases and weights."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # FeedForward
        activation = x  # the first activation, which is the input
        activations = [x]  # list to store all activations, starts with the first.
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)

            zs.append(z)
            activations.append(activation)

        # Backward Pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  # HADAMARD PRODUCT
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            sp = sigmoid_prime(zs[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """returns a list of d(Cx)/d(a).
        This is only true for the the Quadric-Cost function."""
        return(output_activations - y)
