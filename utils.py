import tensorflow as tf

def conv_layer(layer_name, input, out_channels, filter = [3,3], strides = [1,1,1,1], is_pretrain = True):
    '''    
    :param layer_name: e.g. conv1, pool1
    :param data: tensor, [batch_size, height, width, channels]
    :param out_channels: convolutional kernels
    :param kernel_size: the size of convolutional kernel
    :param strides: 1-D of length 4
    :param is_pretrain: if load pretrained parameters, freeze all convolutional layers
    :return: 4D tensor
    '''

    in_channels = input.get_shape()[-1]
    # data_shape --> [batch, in_height, in_width, in_channels]
    # filter_shape --> [filter_height, filter_width, in_channels, out_channels]
    with tf.variable_scope(layer_name) as scope:
        weights = tf.get_variable(name = 'weights',
                                  shape = [filter[0], filter[1], in_channels, out_channels],
                                  trainable = is_pretrain,
                                  initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name = 'biases',
                                 shape = [out_channels],
                                 initializer = tf.constant_initializer(0.0))
        x = tf.nn.conv2d(input, weights, strides, padding = 'SAME', name = scope.name)
        x = tf.nn.bias_add(x, biases, name = 'bias_add')
        x = tf.nn.relu(x, name = 'relu')
        return x

# pool layer
def pool_layer(layer_name, input, filter = [1,2,2,1], strides = [1,2,2,1], is_max_pool = True):
    '''
    :param layer_name: 
    :param input:
    :param filter: 
    :param strides: 
    :param is_max_pool: 
    :return: 
    '''
    if is_max_pool:
        x = tf.nn.max_pool(input, ksize = filter, strides = strides, padding = 'SAME', name = layer_name)
    else:
        x = tf.nn.avg_pool(input, ksize = filter, strides = strides, padding = 'SAME', name = layer_name)
    return x

# batch normalization
def batch_normalization(input, epsilon = 1e-3):
    batch_mean, batch_varization = tf.nn.moments(input, [0])
    x = tf.nn.batch_normalization(input,
                                  mean = batch_mean,
                                  variance = batch_varization,
                                  offset = None,
                                  scale = None,
                                  variance_epsilon = epsilon)
    return x

# fully connection layer
def fc_layer(layer_name, input, num_output):
    '''
    :param layer_name: e.g. fc1, fc2
    :param input: 
    :param num_output: 
    :return: 
    '''
    shape = input.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        weights = tf.get_variable('weights',
                                  shape = [size, num_output],
                                  initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',
                                 shape = [num_output],
                                 initializer = tf.constant_initializer(0.0))
        input_flat = tf.reshape(input, [-1, size])
        x = tf.matmul(input_flat, weights)
        x = tf.nn.bias_add(x, biases)
        x = tf.nn.relu(x)
        return x

# cost function
def loss(layer_name, logits, labels):
    '''
    :param logits: tensor --> [batch_size, n_classes]
    :param labels: one-hot labels
    :return: 
    '''

    with tf.variable_scope(layer_name) as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                                labels = labels,
                                                                name = 'cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name = scope.name)
        tf.summary.scalar(scope + '/loss', loss)
        return loss

# accuracy
def compute_accuracy(layer_name, logits, labels):
    with tf.variable_scope(layer_name) as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
        return accuracy

# num correct prediction
def correct_prediction(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct = tf.cast(correct, tf.int32)
    correct = tf.reduce_sum(correct)
    return correct

# optimize
def optimize(loss, learning_rate, global_step):
    '''
    optimization, Adam as default
    :param loss: 
    :param learning_rate: 
    :param global_step: 
    :return: 
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step = global_step)
        return train_op
