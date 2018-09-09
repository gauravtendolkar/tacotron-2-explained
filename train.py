import tensorflow as tf
from model.model_fn import create_model
from model.input_fn import train_input_fn
from model.loss import composite_loss
from model.utils import Vocabulary
import os, json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
STATIC_CONFIG = dict(json.load(open('config.json', 'r')))
RUNTIME_CONFIG = {"root_path": ROOT_PATH}
CONFIG = {**STATIC_CONFIG, **RUNTIME_CONFIG}

sess = tf.Session()
vocabulary = Vocabulary()
next_training_batch = train_input_fn(vocabulary, CONFIG)

mel_outputs, residual_mels = create_model(next_training_batch,
                                          CONFIG, is_training=True)

loss = composite_loss(mel_outputs, residual_mels, next_training_batch['targets'])

opt = tf.train.AdamOptimizer()

opt_op = opt.minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(500):
    ntb, o, _ = sess.run([next_training_batch, loss, opt_op])
    print(o)
    # [a, b, c] = sess.run([next_element, output, loss])
    # print('-----------INPUTS------------')
    for key, value in ntb.items():
        print('{} - {}'.format(key, value.shape))
    # print('-----------OUTPUTS------------')
    # print('Mel output - ', b[0].shape)
    # print('Stop token - ', b[1].shape)
    # print('Residual Mels - ', b[2].shape)
    # print('-----------LOSS------------')
    # print('Loss - ', c)
    # print(b.shape)