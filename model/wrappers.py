import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from model.modules import decoder_prenet

'''
This would be a good tutorial on how to write custom wrappers

1. Create a class that inherits from RNNCell

2. Initialize the superclass (RNNCell) in __init__. Set the _cell attribute of RNNCell to a cell of your choice.
It has to be an instance of RNNCell though. For us, everything else remains the same except 
that inputs to our cell should first be passed through a pre-net. 
We also need the is_training flag to set/unset dropout in pre-net

3. We have to define 2 properties on our cell - state_size, output_size. And 2 methods - call, zero_state
We have to define these 4 because the RNNCell computation will use them but we are using cell of our choice. 
For example, if c = DecoderPrenetWrapper(...), when we do c(inputs, states),
behind the scenes, c.call(inputs, states) is called. This will use the call method on the RNNCell which we dont want. 
We want call method of the custom cell we used. 
So we override the call method of RNNCell to return self._cell(inputs, states)

If the cell we wish to use is already implemented in TF, like LSTMCell or GRUCell, we can write,
DecoderPrenetWrapper that directly inherits from LSTMCell or GRUCell. In this case we would just have to override
call method since we want to use prenet
'''


class DecoderPrenetWrapper(RNNCell):
    def __init__(self, cell, is_training=True):
        super(DecoderPrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state):
        prenet_out = decoder_prenet(inputs, self._is_training, name='decoder_prenet')
        return self._cell(prenet_out, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


'''
Now we have to write another wrapper for concatenating attention vector and prenet output
But why another wrapper? cant we do the modification inside the prenet wrapper itself?

No. Because, the cell there is an order among computations as given in model_fn comments - 
# Summary -
    # 1. Create a LSTM cell with zoneout by using ZoneoutWrapper.
    # This new cell defines a unit of computation. 2 inputs, 2 outputs. Now we modify this
    # 2. Pass input (first input) to cell through a pre-net
    # 3. Use input state (second input) to the decoder cell and encoder outputs (memory) to compute attention vector
    # 4. Concatenate pre-net output to attention
    # 5. Then this concatenated vector is used as input and we already have input state.
    # They are used for RNN computation
    
Therefore, we first wrap with zoneout wrapper then decoder prenet wrapper, 
then with attention wrapper, then with concatenate wrapper

The zoneout and attention wrappers are available in external directory
Their code was taken from external sources and is slightly more complicated than these basic wrappers
It still follows the same recipe and is heavily commented
'''


class ConcatOutputAndAttentionWrapper(RNNCell):
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class OutputProjectionWrapper(RNNCell):
    def __init__(self, cell, linear_projection_size):
        super(OutputProjectionWrapper, self).__init__()
        self._cell = cell
        self.linear_projection_size = linear_projection_size
        self.out_projection_layer = tf.layers.Dense(units=linear_projection_size, activation=tf.nn.relu, name='output_projection')

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_projection_size

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)

        out_projection = self.out_projection_layer(output)

        return out_projection, res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)