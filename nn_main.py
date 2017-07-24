import numpy as np


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


X = np.array([[-0.2, 0.3, -0.4]])

y = np.array([[0.1]])


np.random.seed(1)

n_hidden = 2
# randomly initialize our weights with mean 0
syn0 = 2 * np.random.random((X[0].shape[0], n_hidden)) - 1
syn0 = np.array([[2.0, -3.0], [0.5, -0.5], [7.0, -7.0]])
syn1 = 2 * np.random.random((n_hidden, 1)) - 1
syn1 = np.array([[4.0], [1.0]])

print('number of inputs: ', X[0].shape[0])
print('number of hidden neurons: ', n_hidden)
print('number of outputs: ', y[0].shape[0])
print('activation function: 1 / (1 + np.exp(-x))')

for j in xrange(1):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    print('-----------------------------')
    print('weights (input-->hidden): ')
    print(syn0)
    print('-----------------------------')
    print('weights (hidden-->output): ')
    print(syn1)
    print('-----------------------------')
    print('input: ')
    print(l0)

    print('-----------------------------')
    print('hidden neurons: ')
    print(l1)

    print('-----------------------------')
    print('network output: ')
    print(l2)

    # how much did we miss the target value?
    l2_error = y - l2
    print('-----------------------------')
    print('true output: ')
    print(y)
    print('-----------------------------')
    print('error: ')
    print(l2_error)

    # if (j % 10000) == 0:
    #     print "Error:" + str(np.mean(np.abs(l2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    print('-----------------------------')
    print('hidden layer error: ')
    print(l1_error)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)

    dw_hidden_output = l1.T.dot(l2_delta)
    dw_input_hidden = l0.T.dot(l1_delta)

    print('-----------------------------')
    print('change in weights hidden-->output: ')
    print(dw_hidden_output)
    print('-----------------------------')
    print('change in weights input-->hidden: ')
    print(dw_input_hidden)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

