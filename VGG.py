import tensorflow as tf
import utils

def VGG16(x, n_classes, is_pretrain = True):
    x = utils.conv_layer('conv1_1', x, 64, filter = [3,3], strides = [1,1,1,1], is_pretrain = is_pretrain)
    x = utils.conv_layer('conv1_2', x, 64, filter = [3,3], strides = [1,1,1,1], is_pretrain = is_pretrain)
    x = utils.pool_layer('pool1', x, filter = [1,2,2,1], strides = [1,2,2,1], is_max_pool = True)

    x = utils.conv_layer('conv2_1', x, 128, filter = [3,3], strides = [1,1,1,1], is_pretrain = is_pretrain)
    x = utils.conv_layer('conv2_2', x, 128, filter = [3,3], strides = [1,1,1,1], is_pretrain = is_pretrain)
    x = utils.pool_layer('pool2', x, filter = [1,2,2,1], strides = [1,2,2,1], is_max_pool = True)

    x = utils.conv_layer('conv3_1', x, 256, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv_layer('conv3_2', x, 256, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv_layer('conv3_3', x, 256, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool_layer('pool3', x, filter=[1,2,2,1], strides=[1,2,2,1], is_max_pool=True)

    x = utils.conv_layer('conv4_1', x, 512, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv_layer('conv4_2', x, 512, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv_layer('conv4_3', x, 512, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool_layer('pool4', x, filter=[1,2,2,1], strides=[1,2,2,1], is_max_pool=True)

    x = utils.conv_layer('conv5_1', x, 512, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv_layer('conv5_2', x, 512, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.conv_layer('conv5_3', x, 512, filter=[3, 3], strides=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = utils.pool_layer('pool5', x, filter=[1, 2, 2, 1], strides=[1, 2, 2, 1], is_max_pool=True)

    x = utils.fc_layer('fc6', x, num_output = 4096)
    x = utils.batch_normalization(x)
    x = utils.fc_layer('fc7', x, num_output = 4096)
    x = utils.batch_normalization(x)
    x = utils.fc_layer('fc8', x, num_output = n_classes)

    return x

