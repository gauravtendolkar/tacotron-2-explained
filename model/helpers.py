import random
import tensorflow as tf
import numpy as np
from tensorflow.contrib.seq2seq import Helper

# We saw the TrainingHelper and GreedyEmbeddingHelper used for decoder RNNs
# But we a slightly different functionality than those inbuilt helpers
# Specifically, for inference, we want to stop predicting when we encounter end of audio
# i.e. the stop token projection layer predicts 1
# For training, we want to predict timesteps = target sequence lengths
# At target sequence length timestep, stop token projection layer should predict 1,
# and 0 for all timesteps before


# We will modify the code given for TrainingHelper in tensorflow source


class CustomTrainingHelper(Helper):
    def __init__(self, inputs, sequence_length, target_inputs, batch_size, time_major=False, name=None, teacher_forcing_p=0.5):
        self._lengths = sequence_length
        self._teacher_forcing_p = teacher_forcing_p
        self._batch_size = batch_size# if not time_major else inputs.shape[1]
        self._output_dim = inputs.shape[2]
        self._inputs = inputs
        self._target_inputs = target_inputs

        # IMPORTANT NOTE: We have to use the same name for the training and test helper so
        # that the variables are reused
        self.stop_token_projection_layer = tf.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                           name='stop_token_projection')

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = tf.tile([False], multiples=[self._batch_size])
        next_inputs = tf.tile([[0.0]], [self._batch_size, self._output_dim])
        return finished, next_inputs

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        # Check if stop projection layer tells us to stop
        # Note that outputs are [batch_size, output_dimension]
        # All the tensors are tensors at a particular time

        finished = tf.greater_equal(time+1, self._lengths)

        # finished has dimension [batch_size] and consists of True/False
        # for each sequence in the batch for the current time

        # Now here we introduce teacher forcing
        # TEACHER FORCING -
        # Choose a uniform random number R between 0 and 1
        # If R > self._teacher_forcing_p:
        #   use output of current timestep as input to next
        # else:
        #   use next target as input to next time step
        #   (current timestep from target inputs, which are a shifted version of targets)

        # Amount of teacher forcing depends on value of self._teacher_forcing_p
        # The value is not specified in paper, we will use 1
        # Which means always use targets during training
        if random.random() > self._teacher_forcing_p:
            # Again, note that outputs are [batch_size, output_dimension]
            next_inputs = outputs
        else:
            # target_inputs were passed by us and had the time axis too
            next_inputs = self._target_inputs[:, time, :]

        return finished, next_inputs, state

    # We have to define few other methods too which we dont use
    # but since they are defined on ABC Helper, we need them
    # Else you get error
    # TypeError: Can't instantiate abstract class CustomTrainingHelper with
    # abstract methods sample, sample_ids_dtype, sample_ids_shape
    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def sample(self, time, outputs, state, name=None):
        # We are not using this but have to return a tensor
        return tf.tile([0], multiples=[self._batch_size])


class CustomTestHelper(Helper):
    def __init__(self, inputs, sequence_length, target_inputs, batch_size, time_major=False, name=None):
        self._lengths = sequence_length
        self._batch_size = batch_size# if not time_major else inputs.shape[1]
        self._output_dim = inputs.shape[2]
        self._inputs = inputs
        self._target_inputs = target_inputs

        # IMPORTANT NOTE: We have to use the same name for the training and test helper so
        # that the variables are reused
        self.stop_token_projection_layer = tf.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                           name='stop_token_projection')

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = tf.tile([False], multiples=[self._batch_size])
        next_inputs = tf.tile([[0.0]], [self._batch_size, self._output_dim])
        return finished, next_inputs

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        # Check if stop projection layer tells us to stop
        # Note that outputs are [batch_size, output_dimension]
        # All the tensors are tensors at a particular time

        finished = tf.reduce_all(tf.equal(outputs, 0.0), axis=1)

        # finished has dimension [batch_size] and consists of True/False
        # for each sequence in the batch for the current time

        next_inputs = outputs

        return finished, next_inputs, state

    # We have to define few other methods too which we dont use
    # but since they are defined on ABC Helper, we need them
    # Else you get error
    # TypeError: Can't instantiate abstract class CustomTrainingHelper with
    # abstract methods sample, sample_ids_dtype, sample_ids_shape
    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32