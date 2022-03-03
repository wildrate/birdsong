# birdsong
Deep learning models for classifying birdsong.

Includes creation of tensorflow datasets.

This repo is all about producing and assessing TensorFlow models (not yet TFLite!).

NOTE: the output of this - as various trained models - are stored it separate `tinymodels` repository - from which location they are converted, optimised and compiled into headers and stored by release for inclusion into [tinyspot(s)](https://github.com/tinyspot).

## Install

You can run this two ways:

1. On Colab remotely on a cloud instance
2. On a local machine running Juypter notebooks directly

### Local running 

For option 2, the tested approach is

* Install `conda` python package manager
* Do the following to create a new environment called `birdsong` from the packages :
```commandline
conda config --add channels conda-forge
conda config --set channel_priority strict
conda -n birdsong python=3.8
conda activate birdsong
conda install --file requirements.txt
```
* You can then activate this environment again anytime to use it.
* Run 

## Datasets

These were created using the info here: https://www.tensorflow.org/datasets/add_dataset

pip install -q tfds-nightly
tfds --version


