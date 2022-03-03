# coding=utf-8
# BSD 3-Clause License
#
# Copyright (c) 2022, Wildrate
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""britishbirds dataset."""

import os
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(britishbirds): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
This is a dataset of bird sound recordings, a specific subset gathered from the Xeno Canto collection to form a balanced dataset across 88 species commonly heard in the United Kingdom. It was originally compiled by Dan Stowell and shared on Archive.org.

The copyright in each audio file is owned by the user who donated the file to Xeno Canto. Please see "birdsong_metadata.tsv" for the full listing, which gives the authors' names and the CC licences applicable for each file. The audio files are encoded as .flac files.

Acknowledgements:
These recordings were collected by 68 separate birding enthusiasts and uploaded to and stored by xeno-canto: www.xeno-canto.org. If you make use of these recordings in your work, please cite the specific recording and include acknowledgement of and a link to the xeno-canto website.
"""

# TODO(britishbirds): BibTeX citation
_CITATION = """
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4106198/
"""

_HOMEPAGE_URL = "https://archive.org/details/xccoverbl_2014"
_DOWNLOAD_URL = "https://archive.org/compress/xccoverbl_2014"
# e.g. to use curl above would be:
# curl -Lo data.zip https://archive.org/compress/xccoverbl_2014

class Britishbirds(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for britishbirds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(britishbirds): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "audio":
                tfds.features.Audio(file_format="wav", sample_rate=8000),
            "label":
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=["no", "yes"])),
            "audio/filename":
                tfds.features.Text()
        }),
        supervised_keys=('audio', 'label'),
        homepage=_HOMEPAGE_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    #path = dl_manager.download_and_extract("britishbirds",_DOWNLOAD_URL)
    # To test/not download again... use this:
    path = dl_manager.extract("/Users/tim/Data/britishbirds/archive_org_compress_xccoverbl_2014.zip")
    print(path)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN, gen_kwargs={"path": path}),
    ]

  def _generate_examples(self, path):
    """Yields examples."""
    for root, _, file_name in tf.io.gfile.walk(path):
      for fname in file_name:
        if fname.endswith(".wav"):  # select only .wav files
          # Example of audio file name: 0_0_1_1_0_1_0_0.wav
          labels = fname.split(".")[0].split("_")
          labels = list(map(int, labels))
          key = fname
          example = {
              "audio": os.path.join(root, fname),
              "label": labels,
              "audio/filename": fname,
          }
          yield key, example
