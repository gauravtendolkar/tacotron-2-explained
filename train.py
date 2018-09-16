import tensorflow as tf
from model.model_fn import create_model
from model.input_fn import train_input_fn
from model.loss import composite_loss
from model.utils import Vocabulary
import os, json

'''
This is how a standard DL training file would look - 

1. Load all configuration (defined in a separate config file)
    Keep 2 files - static config and dynamic config. 
    dynamic config involves some code that uses system dependent information and configuration

2. Define input data pipeline (defined in a separate input pipeline file)

3. Define model graph (defined in a separate model file)
    A single create_model function which takes in a flag is_training.
    The function then builds separate graph based on whether is_training is true or false
    THE CRITICAL ASPECT OF GRAPH BUILDING TO GRASP:
        (a.) Always create variables with tf.get_variable and pass a name. If a variable with same name exists, 
        tf will reuse it (which means its value will be its current value). If it doesnt exist, 
        it will create a new one. This makes it easy to build different graph for training and inference 
        (since share a lot of variables. They share variables because inference graph wants its variables 
        to have values trained during training). 
        (b.) Use name scopes with tf.variable_scope 
        (NOT tf.name_scope as explained here - https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow)
        and use the flag reuse=tf.AUTO_REUSE (why? read on)
        (c.) What about when variables are created for us by a higher level API like tf.layers?
        tf.layers has a reuse flag which should be set to True. This will ensure that when the same create_model
        function creates inference graph, the layer is reused from the training graph previously trained
        (d.) But what if I am creating multiple 2d convolution layers with tf.layers.conv2d? wont all layers be shared?
        NOT if you pass unique name when creating each of those layers. In summary, you need to set reuse=True and 
        pass a unique name everytime you call the API 
        (e.) What about tf.nn.conv2d ? I dont see any reuse parameter for it?
        All higher level APIs use tf.get_variable. When your entire graph is nested in a variable scope with tf.AUTO_REUSE,
        the tf.nn.conv2d will reuse the variables. Again it is important to pass a unique name if you have multiple 
        such distinct layers
        (f.) An example code that might be helpful is the function n_layer_1d_convolution in model.modules
        It creates n layers of 1d convolution. Note how each layer is uniquely named in the loop. If not, 
        they will share weights. Also note that dropout is not needed during inference and so the dropout 
        layer is only created if is_training flag is true. Therefore when inference graph is created, it reuses all 
        the same n convolution layers of training graph but does not create the dropout layers

4. Define loss (defined in a separate loss file)

5. Define optimizer

6. Create session

7. Initialize all variables of graph

8. Submit your graph/sub-graph to session for execution

9. Print stats (Also add stats to Tensorboard)
'''

# Load config
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
STATIC_CONFIG = dict(json.load(open('config.json', 'r')))
RUNTIME_CONFIG = {"root_path": ROOT_PATH}
CONFIG = {**STATIC_CONFIG, **RUNTIME_CONFIG}


# This is the class that holds the mappings of characters to numbers and vice versa
# The input pipeline uses this to map characters in text to indexes into embedding matrix
vocabulary = Vocabulary()

# Define input pipeline
next_training_batch = train_input_fn(vocabulary, CONFIG)

# Create the training model. Inference model is created with same function in the synthesize_results.py file
mel_outputs, residual_mels = create_model(next_training_batch,
                                          CONFIG, is_training=True)

# Define loss. Loss is a tensor containing the value to minimize.
loss = composite_loss(mel_outputs, residual_mels, next_training_batch['targets'])

# Create optimizer instance
opt = tf.train.AdamOptimizer()

# Call the minimize op. This, unlike most ops that return tensors, is an op which returns another op
# Call to minimize is equivalent to calling 2 things - compute_gradients() and apply_gradients(), in that order
# If we wanted to modify gradients before applying, we could use those 2 instead
# Example - gradient clipping (although this can also be implemented with gate_gradients parameter to minimize)
opt_op = opt.minimize(loss)

# Create session
sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(ROOT_PATH+'/logs' + '/train', sess.graph)

saver = tf.train.Saver()

# Initialize all variables
# Variables only have values after they are initialized and everything must have a value during execution
# Therefore variable initializers must be run explicitly before any other ops.
# The easiest way to do that is to add an op that runs all the variable initializers,
# and run that op before using the model. This is what tf.global_variables_initializer() does
sess.run(tf.global_variables_initializer())

steps = 0
while True:
    steps += 1
    try:

        '''
        One call to sess.run evaluates every variable once. Even if some
        elements passed in the input array are subgraphs of other, they are not evaluated twice
        For example -
        a = tf.random_uniform(shape=(2,1), name='random_a')
        b = tf.random_uniform(shape=(2,1), name='random_b')
        c = a+b
        sess.run([c,b]) will initialize b only once
        b1, b2 = sess.run([b,b]) will also initialize b only once and b1, b2 will have same values
        but different calls to sess.run, like b1, b2 = sess.run(b), sess.run(b)
        will give different values for b1 and b2
        '''

        ntb, training_loss, summary, _ = sess.run([next_training_batch, loss, merged, opt_op])
        train_writer.add_summary(summary, steps)

        print("-------OUTPUT---------")
        print("Loss {} at batch {}".format(training_loss, steps))

        print("----INPUT BATCH DETAILS------")
        for key, value in ntb.items():
            print('{} - {}'.format(key, value.shape))

        if steps % 10 == 0:
            saver.save(sess, './logs/tacotron-2-explained', global_step=steps)

    except tf.errors.OutOfRangeError:
        saver.save(sess, './logs/tacotron-2-explained-final', global_step=steps)
        print("----TRAINING OVER------")
        break

