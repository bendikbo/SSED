# SSED - Sliding-window Sound Event Detection

This repository contains a sliding window sound event detection system developed for a master thesis at NTNU IES, with contributions by NINA - Norsk Institutt for NAturfroskning.

The codebase contains multiple convolutional backbones, and is (almost) fully configurable in both training and inference through yaml specified configurations.
This codebase should be easily extendable and reusable for most sound event detection tasks, and I hope you get good use of it, just remember I've licensed it under the MIT License.

# Example application with existing backbones and datasets

I've already added a dataset, and the standard cfg-object in classifier/config/defaults.py should lead to the main training script automatically downloading this dataset and starting a training session once downloaded and extracted. So the full list of bash terminal commands to train a (somewhat) state of the art sound event detection system for the bird sounds in the dataset should be as simple as:

```bash
git clone https://github.com/bendikbo/SSED.git
cd SSED
mkdir env
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
python train.py configs/default.yaml
```
Assuming you have a somewhat new version of python3 already installed, and has installed the virtualenv package.

**In case you're new to python virtual environments**

These commands create an environment for python just for this project in SSED/env, normally this is done so that you don't need to worry about messing up your baseline python environment to run the code and fuck up all your dependencies. This is the smartest move, I promise. You also could set up an anaconda environment, but there's some stuff in requirements.txt that isn't supported by the package manager for anaconda, and I found it a hassle to deal with, so I'm sticking to pip.

# Howto create "state of the art model" with your own data

Okay, so you have a problem where you actually need to create your own * *state of the art* * model for sound event detection, because of reasons. No worries! I'll take you through the steps of doing it right here.

**1. Creating annotations for a dataset**

First of all you need to annotate your dataset, I recommend using audacity for this, as it has built-in spectrogram support, hotkeys for labeling (ctrl+B), in addition to a ton of other cool functionalities. This part is the most boring part of your job, and it's probably not weird that * *data scientists* * (whatever that means) never do this job themselves. After you're done with labeling an audio file, you need to split it into chunks more suitable for training a classifier. Use the script in scripts/create_dataset.py for exactly this purpose, it splits your source audio files into .wav files and .csv files containing descriptions of your sound events.

**2. Write dataset for your data**

Writing datasets in pytorch is (somewhat) easy, you need to make an inherited class of torch.utils.data.Dataset, overwriting two functions, \_\_len\_\_ and \_\_getitem\_\_. Below is an example.
```python
from torch.utils.data import Dataset as dataset
import torchaudio
import os
import pandas as pd
from classifier.data.transform.transforms import AudioTransformer as transformer
from classifier.data.transform.target_transform import build_target_transform


class mydataset(dataset):
"""
  Super simple example implementation of dataset for this codebase
"""
  def __init__(self, cfg, is_train=True):
    #Get a list of all source files here
    #e.g. (given that you've made a cfgnode as cfg.INPUT.SOURCE_PATH)
    self.source_files = os.listdir(cfg.INPUT.SOURCE_DIR)
    self.audio_files = []
    #Probably best to just keep annotations in memory
    self.annotation_dict = {}
    for filename in self.source_files:
      file_extension = filename.split(".")[-1].lower()
      if file_extension == "csv":
        wav_name = filename.splie(".")[0].lower() + ".wav"
        self.annotation_dict[wav_name] = self.read_csv(filename)
      else:
        self.audio_files.append(filename)
    self.transform = transformer(cfg, is_train = is_train)
    self.target_transform = build_target_transform(cfg)
  def read_csv(self, csv_file):
    df = pd.read_csv(csv_file)
    onsets = df["onset"].to_list()
    offsets = df["offset"].to_list()
    lines = zip(onsets, offsets)
    labels = df["class"].to_list()
    return lines, labels
  def __len__(self):
    return len(self.audio_files)
  def __getitem__(self,idx):
    filename = self.audio_files[idx]
    x, fs = torchaudio.load(filename)
    lines, labels = self.annotation_dict[filename]
    x, lines, labels = self.transform(x, lines, labels)
    x, target = self.target_transform(x, lines, labels)
    return x, target, idx
    
```


# Project description

Yeah, so this project started as my master thesis in, get this, *Electronic System Design*, oh the places you'll go. I really don't think I can make a better explanation of the theory stuff than I've done in the thesis theory section and methodology section, so I'll just recommend reading that if you need some background theory on this stuff or some nice figures. Don't worry, these sections are mostly pictures, as I'm not good with getting ideas in my head through words either.


# Docker setup

TODO: add description of how to set up docker

# Data augmentation and feature engineering

The AudioTransformer class, found within classifier/data/transform/transforms.py is an extendable, reusable, automatically configurable tool that comes with some limitations and expectations of added functionality. To set the stage for possible reusability and/or extendability, it's useful to mention some of the assumptions the class makes if new transforms are to be added to it.

**Transformation object forward arguments**

The forwarding arguments for every transformation class is assumed to be x, lines, and labels. X is currently always assumed to be a single channel 1D tensor, which is the reason that the spectrogram transformation is listed last in both transform_dicts, since the spectrify transform converts returns a single or multiple channel 2D tensors (which in all fairness means the tensor is 3D in the case of mutliple channels, but that's just a pedantic remark). Lines and labels are lists that, given an indice i, lines[i] contains a tuple with (onset, offset) of an audio event, and labels[i] contain the label for the aforementioned audio event.

**Support for pure classification problems**

In the interest of supporting pure classification problems, especially mutual exclusive classification problems, where neither lines nor labels probably need to be forwarded, as in most cases of such problems the class of the underlying data remains constant no matter how it's augmented, there is an option to only forward the x-tensor through any instance of the AudioTransformer-class. Therefore, each class currently needs to have a special case where lines=None and labels=None, where the module does not do any line or label readjustment that would otherwise be necessary with transforms like the time shifting transform. This means that if lines and labels are to be augmented, it's best to leave an if statement in the style of the following snippet.
