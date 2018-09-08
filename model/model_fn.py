import tensorflow as tf
from model.modules import n_layer_1d_convolution, postnet
from model.external.zoneout_wrapper import ZoneoutWrapper
from model.external.attention import LocationSensitiveAttention
from model.wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper, OutputProjectionWrapper
from model.helpers import CustomTrainingHelper, CustomTestHelper
from tensorflow.contrib.rnn import LSTMCell

VOCABULARY_SIZE = 100


def create_model(data, config, is_training=True):

    with tf.variable_scope('tacotron_2', reuse=tf.AUTO_REUSE):
        # Why tf.AUTO_REUSE?

        # reuse=True raises an error if get_variable requests a variable
        # with given name that has not been created before

        # AUTO_REUSE on the other hand will create the variable if it doesnt exist
        # It will reuse if it exists

        inputs, input_sequence_lengths, \
        targets, target_sequence_lengths, \
        target_inputs = data['inputs'], data['input_sequence_lengths'], \
                        data['targets'], data['target_sequence_lengths'], \
                        data['target_inputs']

        # STEP 1
        # Get batch size from inputs
        # There 2 ways to get shape in tensorflow - tf.get_shape and tf.shape
        # tf.get_shape returns static shape which inferred from operations in graph and may contain ?
        # tf.shape gives dyamic shape of tensor which never has ?
        # And as always, inputs - (batch_size, ...).
        # In our case (batch_size, timesteps, vocabulary_size) if input is one hot
        # or (batch_size, timesteps, 1) if input is sequence of index into vacobulary
        batch_size = tf.shape(inputs)[0]

        # STEP 2
        # Create variables for embedding table
        # We will use tf.get_variable
        # As noted in this article (***), we will use get_variable only once for every variable
        # We will not use reuse=True flag, but instead store pointer to the embedding layer
        # and use that for inference
        # Shape of our embedding table would be (vocabulary_size, embedding_output_dimension)
        embedding_table = tf.get_variable('embedding_table', [VOCABULARY_SIZE, 128], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.5))
        # Once we have embedding table variables, we use a lookup operation to get embedded tensor
        # Remember that inputs have to be (batch_size, timesteps, 1)
        # i.e. batches of sequences of indexes in vocabulary
        embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)

        # STEP 3
        # We pass these embedded inputs through 3 layer 1-D convolution with padding='same'
        conv_outputs = n_layer_1d_convolution(embedded_inputs, n=3, filter_width=5, channels=512, name='encoder_convolution')

        # STEP 4
        # Create the single layer bidirectional LSTM
        # The paper says -
        # "The output of the final convolutional layer is passed into a single bi-directional LSTM
        # layer containing 512 units (256 in each direction) to generate the encoded features"

        # First we create a cell, which is one unit of computation
        # There is no recurrence here. The cell takes 2 inputs and returns 2 outputs
        # In our case, the 2 inputs should be previous state and current input
        # and outputs would be interpreted as current output and cell state

        # As the paper says -
        # "In contrast to the original Tacotron, our model uses simpler building blocks, using vanilla LSTM"
        # Original Tacotron used GRU cells

        # As you would discover, there are 2 LSTM cells in TF API - BasicLSTMCell, LSTMCell
        # LSTMCell has support for optional peep-hole connections, optional cell clipping,
        # and an optional projection layer,
        # whereas the BasicLSTMCell does not have support for any of those. We will go with BasicLSTMCell

        # Note that paper also says -
        # "LSTM layers are regularized using zoneout with probability 0.1"
        # There is a convention followed when modifying a cell which is to create a "wrapper class"
        # that inherits from base class and overrides/adds functionality
        # Tensorflow has a ZoneoutWrapper which we can use here
        # (If Google uses an op in a paper, you can expect it to be implemented in Tensorflow,
        # but not available in stable release)

        # In this case we will use the ZoneoutWrapper defined on TF master branch, hidden under research
        # https://github.com/tensorflow/models/blob/master/research/maskgan/regularization/zoneout.py

        cell_fw = ZoneoutWrapper(cell=LSTMCell(256, name='encoder_lstm_forward'),
                                 zoneout_drop_prob=0.1, is_training=is_training)
        cell_bw = ZoneoutWrapper(cell=LSTMCell(256, name='encoder_lstm_backward'),
                                 zoneout_drop_prob=0.1, is_training=is_training)

        # Once we have defined the cells, we use to unroll the RNN
        # Documentation says -
        # "Takes input and builds independent forward and backward RNNs.
        # The input_size of forward and backward cell must match.
        # The initial state for both directions is zero by default (but can be set optionally)
        # and no intermediate states are ever returned --
        # the network is fully unrolled for the given (passed in) length(s) of the sequence(s)
        # or completely unrolled if length(s) is not given."

        # Returns -
        # A tuple (outputs, output_states) where: outputs: A tuple (output_fw, output_bw)
        # containing the forward and the backward rnn output Tensor.
        # If time_major == False (default), output_fw will be a Tensor shaped:
        # [batch_size, max_time, cell_fw.output_size]
        # and output_bw will be a Tensor shaped: [batch_size, max_time, cell_bw.output_size]

        # Output states is the final state for both directions and we discard that
        # The initial state of decoder is 0 and uses attention of encoder outputs
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            conv_outputs,
            sequence_length=input_sequence_lengths,
            dtype=tf.float32,
        )

        outputs = tf.concat(outputs, axis=1, name='concat_blstm_outputs')

        # Now we are done with the encoder part and we will be moving on to the third critical aspect of model - Attention
        # The paper says -
        # "The encoder output is consumed by an attention network which
        # summarizes the full encoded sequence as a fixed-length context vector
        # for each decoder output step. We use the location-sensitive attention
        # from [21], which extends the additive attention mechanism [22] to
        # use cumulative attention weights from previous decoder time steps as an additional feature."

        # STEP 1:
        # Create decoder cell. Decoder is a unidirectional RNN (obviously)
        decoder_cell_layer_1 = ZoneoutWrapper(cell=LSTMCell(256, name='decoder_lstm_layer_1'), zoneout_drop_prob=0.1,
                                      is_training=is_training)
        decoder_cell_layer_2 = ZoneoutWrapper(cell=LSTMCell(256, name='decoder_lstm_layer_2'), zoneout_drop_prob=0.1,
                                      is_training=is_training)

        # Now we need to modify functionality of this cell to add attention computation
        # So as per convention, we write another wrapper class
        # But we dont have to write this class, TF has AttentionWrapper class that
        # takes attention mechanism and cell as inputs and outputs the wrapped cell

        # So first we define the attention mechanism which attends to encoder outputs
        attention_mechanism = LocationSensitiveAttention(num_units=128, memory=outputs, name='decoder_attention_mechanism')

        # Then we wrap decoder cell using this mechanism
        # BUT...
        # This is not the only modification we need to make.

        # Additional modification no. 1
        # The output of decoder is MFCC which is passed through
        # a pre-net before passing to next decoder time step
        # which means, at every timestep, the decoder cell has to pass its inputs through a pre-net

        # Additional modification no. 2
        # The output of the pre-net is not directly passed to next timestep. We have to pass the attention too,
        # The output of pre-net is first concatenated with attention vector of that timestep
        # Only after that all the cell computations are performed on the concatenated vector

        # Summary -
        # 1. Create a LSTM cell with zoneout by using ZoneoutWrapper.
        # This new cell defines a unit of computation. 2 inputs, 2 outputs. Now we modify this
        # 2. Pass input (first input) to cell through a pre-net
        # 3. Use input state (second input) to the decoder cell and encoder outputs (memory) to compute attention vector
        # 4. Concatenate pre-net output to attention
        # 5. Then this concatenated vector is used as input and we already have input state.
        # They are used for RNN computation

        # Pop Quiz 1.3 - Why do the following wrappers and MultiRNNCell do not expect a name to be provided?

        decoder_cell_layer_1 = DecoderPrenetWrapper(decoder_cell_layer_1)
        decoder_cell_layer_1 = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell_layer_1,
                                                                   attention_mechanism=attention_mechanism,
                                                                   output_attention=False,
                                                                   alignment_history=True)
        decoder_cell_layer_1 = ConcatOutputAndAttentionWrapper(decoder_cell_layer_1)

        # TF has supplies a OutputProjectionWrapper to wrap our RNN cell
        # The wrapper projects the output at every timestep to a specified dimension
        # For example, if decoder cell has 256 units, but our num mel bins are only 80,
        # we project down each 256 vector to 80
        # internally, this is just a matrix multiplication of 256 size vector with a 256x80 variable matrix

        # Note that projection is on cell in the last layer of RNN stack
        # Rest of the wrappers, concerning attention, are on the first layer in the stack

        decoder_cell_layer_2 = OutputProjectionWrapper(decoder_cell_layer_2,
                                                       linear_projection_size=config['data']['num_mel_bins'])

        # Then we use MultiRNNCell to stack our 2 cells
        # Then we use dynamic_rnn to unroll the cells for recurrence
        stacked_decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_cell_layer_1, decoder_cell_layer_2])

        # This finishes the complicated business of creating the decoder
        # One thing we missed is the architecture to feed data to decoder
        # During training, the decoder can take entire input and target sequence, but during inference,
        # output at every timestep has to be passed to the next timestep sequentially

        # TF provides handy helper classes to do this -
        # tf.contrib.seq2seq.TrainingHelper during training
        # tf.contrib.seq2seq.GreedyEmbeddingHelper during inference
        # We will be using a slightly modified version of both since we require different functionalities
        # Task - Check the code for tf.contrib.seq2seq.TrainingHelper in TF source
        # What's the thing that we need but is not in that code?
        # We will write a CustomTrainingHelper on the same lines that does our job

        if is_training:
            helper = CustomTrainingHelper(inputs=target_inputs,
                                          sequence_length=target_sequence_lengths,
                                          target_inputs=target_inputs,
                                          batch_size=batch_size,
                                          teacher_forcing_p=1.0)
        else:
            helper = CustomTestHelper(inputs=target_inputs,
                                      sequence_length=target_sequence_lengths,
                                      target_inputs=target_inputs,
                                      batch_size=batch_size)

        # Note that zero_state method is available on MultiRNNCell instance.
        # It basically calls zero_state of every cell in stack
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=stacked_decoder_cell,
                                                  helper=helper,
                                                  initial_state=stacked_decoder_cell.zero_state(batch_size, tf.float32))

        # Now we are ready to call dynamic_decode to unroll our decoder cells
        (mel_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

        # Postnet
        # The output before postnet are mels
        # Output after postnet are residuals (same dimension as mels) that are to be added to mels
        # Intuition -
        # Output before postnet is a crude version of mels
        # Residuals are "smoothening" them to match the profile of actual mels produced by human voice
        residual_mels = postnet(mel_outputs)

        # Postnet outputs 512 channels of residuals
        # We take mean of all channels and add it to our mels
        residual_mels = tf.reduce_mean(residual_mels, axis=-1)

        residual_added_mels = tf.add(mel_outputs, residual_mels)

        return mel_outputs, residual_added_mels
