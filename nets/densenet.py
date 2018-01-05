"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

def transit(net,compression_rate,scope='transit'):
    net=slim.conv2d(net, int(net.shape.as_list()[3]*compression_rate), [1,1], scope=scope + '_conv')
    net=slim.avg_pool2d(net, [2, 2], stride=2, padding='VALID')
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates Densenet-121 model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 12
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            # 224*224*3
            net=slim.conv2d(images, 2*growth, [7, 7], stride=2,
                        padding='same', scope='Conv2d_0a_7x7')
            # 112*112*2k
            net=slim.max_pool2d(net, [3, 3], stride=2, padding='same',
                                     scope='MaxPool_0b_3x3')
            # 56*56*2k
            end_points['input_layer'] = net

            # dense block1
            net=block(net, 6, growth, scope='block1')
            # 56*56*(2k+6k)
            end_points['dense_block1'] = net

            # transition layer1
            net=transit(net,compression_rate,scope='transit1')
            # 28*28*((2k+6k)*compression_rate)
            end_points['transition_layer1'] = net

            # dense block2
            net=block(net, 12, growth, scope='block2')
            # 28*28*((2k+6k)*compression_rate+12k)
            end_points['dense_block2'] = net

            # transition layer2
            net = transit(net, compression_rate, scope='transit2')
            # 14*14*((2k+6k)*compression_rate+12k)*compression_rate
            end_points['transition_layer2'] = net

            # dense block3
            net=block(net, 24, growth, scope='block3')
            # 14*14*(~+24k)
            end_points['dense_block3'] = net

            # transition layer3
            net = transit(net, compression_rate, scope='transit3')
            # 7*7*(~+24k)*compression_rate
            end_points['transition_layer3'] = net

            # dense block4
            net=block(net, 16, growth, scope='block4')
            # 7*7*(~+16k)
            end_points['dense_block4'] = net

            # classification layer
            # 7*7 global average pooling
            net = slim.avg_pool2d(net, [7, 7], stride=1, padding='VALID')
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_classes,
                                              activation_fn=None,
                                              scope='logits')
            end_points['classify_layer'] = net

    return net, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004,activation_fn=tf.nn.relu):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=activation_fn, biases_initializer=None, padding='same',
            stride=1) as sc:
            return sc


densenet.default_image_size = 224
