# birdsong
Deep learning models for classifying birdsong.

Includes creation of tensorflow datasets.

This repo is all about producing and assessing TensorFlow models (not yet TFLite!).

NOTE: the output of this - as various trained models - are stored it separate `tinymodels` repository - from which location they are converted, optimised and compiled into headers and stored by release for inclusion into [tinyspot(s)](https://github.com/tinyspot).

## Datasets

These were created using the info here: https://www.tensorflow.org/datasets/add_dataset

pip install -q tfds-nightly
tfds --version


