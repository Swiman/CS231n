import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *
#from layers import *
#from layer_utils import *


class TwoLayerNet(object):
    def __init__(
            self,
            input_dim=3 * 32 * 32,
            hidden_dim=100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
    ):
        self.params = {}
        self.reg = reg
        self.params["W1"] = np.random.normal(0, weight_scale,
                                             (input_dim, hidden_dim))
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = np.random.normal(0, weight_scale,
                                             (hidden_dim, num_classes))
        self.params["b2"] = np.zeros(num_classes)

    def loss(self, X, y=None):
        h1, h1_cache = affine_relu_forward(X, self.params["W1"],
                                           self.params["b1"])
        scores, s_cache = affine_forward(h1, self.params["W2"],
                                         self.params["b2"])
        if y is None:
            return scores
        grads = {}
        loss, dscores = softmax_loss(scores, y)
        dh1, grads['W2'], grads['b2'] = affine_backward(dscores, s_cache)
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dh1, h1_cache)
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) +
                                  np.sum(self.params['W2']**2))
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """
    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        -************* dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.*************************
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        for i in range(1, self.num_layers + 1):
            wi = input_dim if i == 1 else hidden_dims[i - 2]
            wj = num_classes if i == self.num_layers else hidden_dims[i - 1]
            self.params['W' + str(i)] = np.random.normal(
                0, weight_scale, (wi, wj))
            self.params['b' + str(i)] = np.random.normal(0, weight_scale, (wj))
            if normalization and i != self.num_layers:
                self.params['gamma' + str(i)] = np.ones(wj)
                self.params['beta' + str(i)] = np.zeros(wj)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{
                "mode": "train"
            } for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode

        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        cache = {}
        d_cache = {}
        cin = X
        l2_regs = 0
        for i in range(1, self.num_layers):
            cW, cb = 'W' + str(i), 'b' + str(i)
            cG, cB = 'gamma' + str(i), 'beta' + str(i)
            l2_regs += np.sum(self.params[cW]**2)
            if not self.normalization:
                cin, cache[i] = affine_relu_forward(cin, self.params[cW],
                                                    self.params[cb])
            elif self.normalization == 'batchnorm':
                cin, cache[i] = affine_bn_relu_forward(cin, self.params[cW],
                                                       self.params[cb],
                                                       self.params[cG],
                                                       self.params[cB],
                                                       self.bn_params[i - 1])
            else:
                cin, cache[i] = affine_ln_relu_forward(cin, self.params[cW],
                                                       self.params[cb],
                                                       self.params[cG],
                                                       self.params[cB],
                                                       self.bn_params[i - 1])
            if self.use_dropout:
                cin, d_cache[i] = dropout_forward(cin, self.dropout_param)

        #its not like c++ : after a for loop value of i will be self.num_layers - 1 not self.num_layers
        scores, s_cache = affine_forward(
            cin, self.params['W' + str(self.num_layers)],
            self.params['b' + str(self.num_layers)])
        l2_regs += np.sum(self.params['W' + str(self.num_layers)]**2)

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, dscores = softmax_loss(scores, y)
        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss += 0.5 * self.reg * l2_regs

        dH, grads['W' + str(self.num_layers)], grads[
            'b' + str(self.num_layers)] = affine_backward(dscores, s_cache)
        grads['W' + str(self.num_layers)] += self.reg * self.params[
            'W' + str(self.num_layers)]
        for j in range(self.num_layers - 1, 0, -1):
            cW, cb = 'W' + str(j), 'b' + str(j)
            if self.use_dropout:
                dH = dropout_backward(dH, d_cache[j])

            if not self.normalization:
                dH, grads[cW], grads[cb] = affine_relu_backward(dH, cache[j])

            cGamma, cBeta = 'gamma' + str(j), 'beta' + str(j)
            if self.normalization == "batchnorm":
                dH, grads[cW], grads[cb], grads[cGamma], grads[
                    cBeta] = affine_bn_relu_backward(dH, cache[j])
            elif self.normalization == "layernorm":
                dH, grads[cW], grads[cb], grads[cGamma], grads[
                    cBeta] = affine_ln_relu_backward(dH, cache[j])
            grads[cW] += self.reg * self.params[cW]

        return loss, grads
