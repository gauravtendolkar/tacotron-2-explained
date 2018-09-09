# Tacotron 2 Explained

This repository is meant to teach the intricacies of writing advanced Recurrent Neural Networks in Tensorflow. The code is used as a guide, in weekly Deep Learning meetings at Ohio State University, for teaching -
1. How to read a paper
2. How to implement it in Tensorflow

The paper followed in this repository is - [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884). The repository only implements the Text to Mel Spectrogram part (called Tacotron 2). The repository does not include the vocoder used to synthesize audio.

This is a production grade code which can be used as state of the art TTS frontend. The blog post \[TODO\] shows some audio samples synthesized with a Griffin Lin vocoder. But the code has excess comments to aid a novice Tensorflow user which could be a hindrance. 

The repository also uses Tensorflow's tf.data API for pre-processing and Estimator API for modularity
 
## Directory Structure
The directory structure followed is as specified in [Stanford's CS230 Notes on Tensorflow](https://cs230-stanford.github.io/tensorflow-getting-started.html). We modify the structure a bit to suite our needs.
```
data/ (Contains all data)
model/ (Contains model architecture)
    input_fn.py (Input data pipeline)
    model_fn.py (Main model)
    utils.py (Utility functions)
    loss.py (Model loss)
    wrappers.py (Wrappers for RNN cells)
    helpers.py (Decoder helpers)
    external/ (Code adapted from other repositories)
        attention.py (Location sensitive attention)
        zoneout_wrapper.py (Zoneout)
train.py (Run training)
config.json (Hyper parameters)
synthesize_results.py (Generate Mels from text)
```

## Requirements
The repository uses Tensorflow 1.8.0. Some code may be incompatible with older versions of Tensorflow (specifically the Location Sensitive Attention Wrapper).

## Setup
1. Setup python 3 virtual environment. If you dont have ```virtualenv```, install it with

```
pip install virtualenv
```

2. Then create the environment with

```
virtualenv -p $(which python3) env
```

3. Activate the environment

```
source env/bin/activate
```

4. Install tensorflow

```
pip install tensorflow==1.8.0
```

5. Clone the repository

```
git clone https://gitlab.com/codetendolkar/tacotron-2-explained.git
```

6. Run the training script

```
cd tacotron2
python train.py
```

## Generate Mels from Text

## Synthesize Audio from Mels

## Credits and References
1. "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"
Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu
[arXiv:1712.05884]()
2. Location Sensitive Attention adapted from Tacotron 2 implementation by [Keith Ito](https://github.com/keithito) - [GitHub link](https://github.com/keithito/tacotron/tree/c94ab2757d52e4294dcd6a8da03f49d251b2dec4)
3. Zoneout Wrapper for RNNCell adapted from Tensorflow's official repository for [MaskGan](https://github.com/tensorflow/models/tree/master/research/maskgan). The code contributed by [A Dai](https://github.com/a-dai) - [GitHub link](https://github.com/tensorflow/models/blob/master/research/maskgan/regularization/zoneout.py)
4. And obviously - all the contributors of [Tensorflow](https://github.com/tensorflow)
5. Internet
