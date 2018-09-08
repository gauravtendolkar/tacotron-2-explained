import tensorflow as tf


def n_layer_1d_convolution(inputs, n, filter_width, channels, name='1d_convolution', is_training=False):
    conv_output = inputs
    for _ in range(n):
        conv_output = tf.layers.conv1d(conv_output, filters=channels, kernel_size=filter_width,
                                       padding='same', name=name+'_layer'+str(_))
        conv_output = tf.layers.dropout(conv_output, rate=0.5, name=name+'_layer'+str(_)+"_dropout", training=is_training)
    return conv_output


def decoder_prenet(inputs, is_training, name='decoder_prenet'):
    # 2 layer dense prenet
    # paper says -
    # "The prediction from the previous time step is first passed through a small pre-net containing
    # 2 fully connected layers of 256 hidden ReLU units.
    # We found that the pre-net acting as an information bottleneck was essential for learning attention"
    output = tf.layers.dense(inputs, units=256, activation=tf.nn.relu, name=name+"_layer_1")
    if is_training:
        output = tf.layers.dropout(output, rate=0.5, name=name+"_layer_1_dropout")
    output = tf.layers.dense(output, units=256, activation=tf.nn.relu, name=name + "_layer_2")
    if is_training:
        output = tf.layers.dropout(output, rate=0.5, name=name+"_layer_2_dropout")
    return output


def postnet(inputs, is_training=False, name='postnet'):
    conv_output = tf.expand_dims(inputs, -1)
    for _ in range(4):
        conv_output = tf.layers.conv2d(conv_output, filters=512, kernel_size=(1, 5),
                                       padding='same', name=name + '_layer_' + str(_), activation=tf.nn.tanh)
        conv_output = tf.layers.dropout(conv_output, rate=0.5, name=name + '_layer' + str(_) + "_dropout",
                                        training=is_training)
    # Final layer has no activation
    conv_output = tf.layers.conv2d(conv_output, filters=512, kernel_size=(1, 5),
                                   padding='same', name=name + '_layer_' + str(5))
    return conv_output




