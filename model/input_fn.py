import tensorflow as tf
from tensorflow.contrib import ffmpeg

# NOTE: Librosa is another good library for processing audio but
# I have been unable to use it with tf.data structure


def parse_csv_line(line, vocabulary, config):
    # tf.decode_csv converts CSV records to tensors. Not read CSV files!
    # Standard procedure to read any file is with tf.data.TextLineDataset
    # After reading the file into a tensor (NUM_LINES x 1), we interpret the tensor as being in CSV format
    # Each line in that tensor is a scalar string
    # Which means we assume every row of tensor (corresponding to every line in file) has
    # multiple columns delimited by the specified delimiter
    # The output we get is a tensor (NUM_LINES, NUM_COLUMNS)
    fields = tf.decode_csv(line, config['data']['csv_column_defaults'])

    # Note that INPUT_CSV_COLUMNS is (1 x NUM_COLUMNS) while fields is (NUM_LINES, NUM_COLUMNS)
    # So zipping gives NUM_COLUMNS tuples (COLUMN_NAME, (NUM_LINES x 1)), from which we create a dict
    features = dict(zip(config['data']['csv_columns'], fields))

    # Split string into characters
    # IMPORTANT NOTE: tf.string_split returns a SparseTensor of rank 2,
    # the strings split according to the delimiter. Read more about how SparseTensors are represented
    text = tf.string_split([features[config['data']['csv_columns'][0]]], delimiter="")

    # Once we have character SparseTensors, we need to encode the characters as numbers
    # Traditional way is to have one hot encoding or a one hot encoding + embedding matrix
    # When you use one hot encoding + embedding matrix, you are basically choosing a row of embedding matrix
    # So to make it faster, tensorflow expects input to embedding layer as the index of the row,
    # instead of having one hot vectors to be multiplied with embedding matrix
    # So we will maintain a Vocabulary where every character we care about has an associated number as 1-to-1
    # This looks like a map operation for which tensorflow has tf.map_fn

    # Now note that SparseTensors do not support all usual Tensor operations
    # To use tf.map_fn on a SparseTensor, we have to create a new SparseTensor in the following way

    # Also note that embedding layer will expect indexes of dtype tf.int64
    # Also, the vocabulary dict stores values as int64

    text_idx = tf.SparseTensor(text.indices,
                               tf.map_fn(vocabulary.text2idx, text.values, dtype=tf.int64),
                               text.dense_shape)

    # We have to convert this SparseTensor back to dense to support future operations
    text_idx = tf.sparse_tensor_to_dense(text_idx) # Shape - (1, T)
    text_idx = tf.squeeze(text_idx) # Shape - (T,)

    # We also require lengths of every input sequence as inputs to model
    # This ia because we will create batches of variable length input
    # where all sequences are forced to same length by padding at the end with 0s
    # This batch will be passed to an Dynamic RNN which will use sequence lengths
    # to mask the outputs appropriately. The RNN will be unrolled to the common length though
    # This method enables us to do mini batch SGD for variable length inputs

    input_sequence_lengths = tf.size(text_idx) # Scalar

    # We are done with processing text (which is out input to Tacotron)
    # Lets move onto audio (which will be our targets)
    # This part is standard code for obtaining MFCC from audio as given in TF documentation
    # You can read more about what are fourier transform, spectrograms and MFCCs to get an idea

    audio_binary = tf.read_file(features[config['data']['csv_columns'][1]])

    # Sample rate used in paper is 16000, channel count should be 1 for tacotron 2
    # STFT configuration values specified in paper
    waveform = ffmpeg.decode_audio(audio_binary, file_format='wav',
                                   samples_per_second=config['data']['wav_sample_rate'],
                                   channel_count=1)

    stfts = tf.contrib.signal.stft(tf.transpose(waveform),
                                   frame_length=config['data']['frame_length'],
                                   frame_step=config['data']['frame_step'],
                                   fft_length=config['data']['fft_length'])
    magnitude_spectrograms = tf.abs(stfts)
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    # These are to be set according to human speech. Values specified in the paper
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = config['data']['lower_edge_hertz'], \
                                                       config['data']['upper_edge_hertz'], \
                                                       config['data']['num_mel_bins']

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, config['data']['wav_sample_rate'], lower_edge_hertz,upper_edge_hertz)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms = tf.squeeze(mel_spectrograms)  # Removes all dimensions that are 1

    # This finishes processing of audio
    # Now we build the targets and inputs to the decoder

    # We append a frame of 0s at the end of targets to signal end of target
    end_tensor = tf.tile([[0.0]], multiples=[1, tf.shape(mel_spectrograms)[-1]])
    targets = tf.concat([mel_spectrograms, end_tensor], axis=0)

    # We append a frame of 0s at the start of decoder_inputs to set input at t=1
    start_tensor = tf.tile([[0.0]], multiples=[1, tf.shape(mel_spectrograms)[-1]])
    target_inputs = tf.concat([start_tensor, mel_spectrograms], axis=0)

    # Again, we require lengths of every target sequence as inputs to model
    # This ia because we will create batches of variable length input
    # where all sequences are forced to same length by padding at the end with 0s
    # This batch will be passed to an Dynamic RNN which will use sequence lengths
    # to mask the outputs appropriately. The RNN will be unrolled to the common length though
    # This method enables us to do mini batch SGD for variable length inputs
    target_sequence_lengths = tf.shape(targets)[0]

    # Now we return the values that our model requires as a dict (just like old feed_dict structure)
    return {'inputs': text_idx,
            'targets': targets,
            'input_sequence_lengths': input_sequence_lengths,
            'target_sequence_lengths': target_sequence_lengths,
            'target_inputs': target_inputs,
            'debug_data': waveform}


def train_input_fn(vocabulary, config):
    # Note that all these operations are added to the graph.
    # Its just that this part of graph does not contain any trainable variables

    dataset = tf.data.TextLineDataset(config['general']['input_csv'])
    dataset = dataset.skip(config['data']['num_csv_header_lines'])
    dataset = dataset.map(lambda line: parse_csv_line(line, vocabulary, config))
    dataset = dataset.repeat(config['hyper_params']['epochs'])
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.padded_batch(config['hyper_params']['batch_size'], padded_shapes={
        'inputs': [None],
        'targets': [None, config['data']['num_mel_bins']],
        'input_sequence_lengths': [],
        'target_sequence_lengths': [],
        'target_inputs': [None, config['data']['num_mel_bins']],
        'debug_data': [None, None]
    })
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
