from builtins import range
import numpy as np


def affine_forward(x, w, b):
    x_test = x.reshape(x.shape[0], -1)
    out = np.dot(x_test, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    x_test = x.reshape(x.shape[0], -1)
    dx = np.dot(dout, w.T).reshape(*x.shape)
    dw = np.dot(x_test.T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    r = np.zeros_like(dout)
    r[x > 0] = 1
    dx = dout * r
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
  """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    cache = None
    if mode == "train":
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        x_normal = (x - x_mean) / np.sqrt(x_var + eps)
        bn_param['running_mean'] = momentum * running_mean + (
            1 - momentum) * x_mean
        bn_param['running_var'] = momentum * running_var + (1 -
                                                            momentum) * x_var
        out = x_normal * gamma + beta
        cache = (x, x_normal, x_mean, x_var, gamma, beta, eps)

    elif mode == "test":
        x_normal = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_normal * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache


def batchnorm_backward(dout, cache):
    (x, xn, xm, xv, g, b, e) = cache
    xdiff = x - xm
    istd = 1 / np.sqrt(xv + e)
    m = x.shape[0]
    dxn = dout * g
    dvar = np.sum(dxn * xdiff * (-0.5 * istd**3), axis=0)

    dmean = np.sum(dxn *
                   (-istd), axis=0) + dvar * (-2 / m) * (np.sum(xdiff, axis=0))
    dx = dxn * istd + dvar * (2 / m * xdiff) + dmean / m
    db = np.sum(dout, axis=0)
    dg = np.sum(xn * dout, axis=0)

    ######################################################
    # original paper: (https://arxiv.org/abs/1502.03167) #
    # might prove to be helpful.                         #
    ######################################################
    return dx, dg, db


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    #############################################################################################################################
    # WARNING : SIMPLIFICATION BY:  https://costapt.github.io/2016/07/09/batch-norm-alt/                                        #
    # USEFUL : https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html #
    #############################################################################################################################

    (x, xn, xm, xv, g, b, e) = cache
    xdiff = x - xm
    istd = 1 / np.sqrt(xv + e)
    m = x.shape[0]

    db = np.sum(dout, axis=0)
    dg = np.sum(xn * dout, axis=0)

    dx = (g * istd / m) * (m * dout - xn * dg - db)

    return dx, dg, db


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
  """
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    eps = ln_param.get("eps", 1e-5)

    x = x.T

    x_mean = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)
    xdiff = x - x_mean
    istd = 1 / np.sqrt(x_var + eps)
    x_normal = xdiff * istd
    x_normal = x_normal.T
    out = x_normal * gamma + beta
    cache = (x_normal, istd, gamma)

    return out, cache


def layernorm_backward(dout, cache):
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    xn_T, istd, g = cache
    dg = np.sum(xn_T * dout, axis=0)
    db = np.sum(dout, axis=0)
    dxn_T = dout * g
    xn = xn_T.T
    dxn = dxn_T.T
    m = xn.shape[0]
    dx = 1 / m * istd * (m * dxn - np.sum(dxn, axis=0) -
                         xn * np.sum(dxn * xn, axis=0))
    dx = dx.T
    return dx, dg, db


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = mask * x
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    if mode == "train":
        dx = dout * mask

    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    p, s = conv_param['pad'], conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prim = 1 + (H + 2 * p - HH) // s
    W_prim = 1 + (W + 2 * p - WW) // s
    #print(type(W_prim))
    out = np.zeros((N, F, W_prim, H_prim))
    for n in range(N):
        x_p = np.pad(x[n], ((0, 0), (p, p), (p, p)), 'constant')
        for f in range(F):
            for h in range(H_prim):
                for wp in range(W_prim):
                    window = x_p[:, h * s:h * s + HH, wp * s:wp * s + WW]
                    out[n, f, h, wp] = np.sum(window * w[f]) + b[f]
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    p, s = conv_param['pad'], conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_prim, W_prim = dout.shape
    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)

    for n in range(N):
        x_p = np.pad(x[n], ((0, 0), (p, p), (p, p)), 'constant')
        dx_p = np.pad(dx[n], ((0, 0), (p, p), (p, p)), 'constant')
        for f in range(F):
            db[f] += np.sum(dout[n, f])
            for h in range(H_prim):
                for wp in range(W_prim):
                    lh, hh = h * s, h * s + HH
                    lw, hw = wp * s, wp * s + WW
                    dx_p[:, lh:hh, lw:hw] += dout[n, f, h, wp] * w[f]
                    dw[f] += dout[n, f, h, wp] * x_p[:, lh:hh, lw:hw]

        dx[n] = dx_p[:, p:-p, p:-p]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    ph, pw, s = pool_param['pool_height'], pool_param[
        'pool_width'], pool_param['stride']
    (N, C, H, W) = x.shape
    H_prim = 1 + (H - ph) // s
    W_prim = 1 + (W - pw) // s
    out = np.zeros((N, C, H_prim, W_prim))
    for n in range(N):
        for c in range(C):
            for h in range(H_prim):
                for w in range(W_prim):
                    out[n, c, h, w] = np.max(x[n, c, h * s:h * s + ph, w *
                                               s:w * s + pw])

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    ph, pw, s = pool_param['pool_height'], pool_param[
        'pool_width'], pool_param['stride'],
    N, C, H, W = x.shape
    _, _, H_prim, W_prim = dout.shape
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for h in range(H_prim):
                for w in range(W_prim):
                    lh, hh = h * s, h * s + ph
                    lw, hw = w * s, w * s + pw
                    t = x[n, c, lh:hh, lw:hw] == np.max(x[n, c, lh:hh, lw:hw])
                    dx[n, c, lh:hh, lw:hw] += t * dout[n, c, h, w]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    (N, C, H, W) = x.shape

    x_t = x.transpose(0, 2, 3, 1).reshape(-1, C)
    x_t, cache = batchnorm_forward(x_t, gamma, beta, bn_param)
    out = x_t.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """

    (N, C, H, W) = dout.shape
    dout_t = dout.transpose(0, 2, 3, 1).reshape(-1, dout.shape[1])
    dx_t, dgamma, dbeta = batchnorm_backward_alt(dout_t, cache)
    dx = dx_t.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
