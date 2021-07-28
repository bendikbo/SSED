# SSED - Sliding-window Sound Event Detection

This repository contains a sliding window sound event detection system developed for a master thesis at NTNU IES, with contributions by NINA - Norsk Institutt for NAturfroskning.

The codebase contains multiple convolutional backbones, and is (almost) fully configurable in both training and inference through yaml specified configurations.
This codebase should be easily extendable and reusable for most sound event detection tasks, and I hope you get good use of it, just remember I've licensed it under the MIT License, and a lot of the other stuff that I've used used is licensed under several other licenses as well (can be found in the LICENSES subdirectory), read up on those so you don't get hit with a C&D.

Send me an email if you have any questions I could probably help answer: bendik.bogfjellmo@gmail.com

An acknowledgement and huge thanks has to be given to Håkon Hukkelås, as this had been developed from different parts of the skeleton code he gives out for the subject he partly runs runs at NTNU "TDT4265 - Computer Vision and Deep Learning". We're all standing on the shoulders of giants, and this is a shoulder I've been standing on, really reccommend checking his stuff out: https://github.com/hukkelas, guy is one of the more talented programmers I've had the pleasures of seeing the works of.

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

Then to run inferrence with the model you've trained:

```bash
python infer.py configs/default.yaml path/to/audio_file.wav
```

This trains an EfficientNet-b7 based model on the dataset I've added, with the basic config file, and runs inference based on the model you've trained.

**In case you're new to python virtual environments**

These commands create an environment for python just for this project in SSED/env, normally this is done so that you don't need to worry about messing up your baseline python environment to run the code and fuck up all your dependencies. This is the smartest move, I promise. You also could set up an anaconda environment, but there's some stuff in requirements.txt that isn't supported by the package manager for anaconda, and I found it a hassle to deal with, so I'm sticking to pip.

# Howto create "state of the art model" with your own data

Okay, so you have a problem where you actually need to create your own * *state of the art* * model for sound event detection, because of reasons. No worries! I'll take you through the steps of doing it right here.

**1. Creating annotations for a dataset**

First of all you need to annotate your dataset, I recommend using audacity for this, as it has built-in spectrogram support, hotkeys for labeling (ctrl+B), in addition to a ton of other cool functionalities. This part is the most boring part of your job, and it's the reason that * *data scientists* * (whatever that means) never do this job themselves. After you're done with labeling an audio file, you need to split it into chunks more suitable for training a classifier. Use the script in scripts/create_dataset.py (TODO:Generalize usability) for exactly this purpose, it splits your source audio files into .wav files and .csv files containing descriptions of your sound events.

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
    self.target_transform = build_target_transform(cfg.INPUT.TRANSFORM)
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

**Add stuff in the codebase for your dataset**
Some stuff in the codebase requires that you add some new functionality, e.g. referencing and dereferencing class labels, so that the model can read numbers and the humans on the other side can read names.


**4. Use your *special* source domain knowledge to make configurations that make sense for the problem you're solving**

So, when I was doing my thesis on creating a model for bird sounds, I got a general sense of how a model should be configured to solve the problem, here are some key aspects that you'll probably need to take into account for your

# Project description

Yeah, so this project started as my master thesis in, get this, *Electronic System Design*, oh the places you'll go. I really don't think I can make a better explanation of the theory stuff than I've done in the thesis theory section and methodology section, so I'll just recommend reading that if you need some background theory on this stuff or some nice figures. Don't worry, these sections are mostly pictures, as I'm not good with getting ideas in my head through words either.


# Data augmentation and feature engineering

The AudioTransformer class, found within classifier/data/transform/transforms.py is an extendable, reusable, automatically configurable tool that comes with some limitations and expectations of added functionality. To set the stage for possible reusability and/or extendability, it's useful to mention some of the assumptions the class makes if new transforms are to be added to it.

**Transformation object forward arguments**

The forwarding arguments for every transformation class is assumed to be x, lines, and labels. X is currently always assumed to be a single channel 1D tensor, which is the reason that the spectrogram transformation is listed last in both transform_dicts, since the spectrify transform converts returns a single or multiple channel 2D tensors (which in all fairness means the tensor is 3D in the case of mutliple channels, but that's just a pedantic remark). Lines and labels are lists that, given an indice i, lines[i] contains a tuple with (onset, offset) of an audio event, and labels[i] contain the label for the aforementioned audio event.

**Support for pure classification problems**

In the interest of supporting pure classification problems, especially mutual exclusive classification problems, where neither lines nor labels probably need to be forwarded, as in most cases of such problems the class of the underlying data remains constant no matter how it's augmented, there is an option to only forward the x-tensor through any instance of the AudioTransformer-class. Therefore, each class currently needs to have a special case where lines=None and labels=None, where the module does not do any line or label readjustment that would otherwise be necessary with transforms like the time shifting transform. This means that if lines and labels are to be augmented, it's best to leave an if statement in the style of the following snippet.

# TODOs

**Add more datasets!**

I'm always eager to expand this project! HMU if you've got a dataset you want to add, I'll even host it on my own google drive and add a download script to automate the process for newcomers.

**Add EfficientNetV2 to backbones**

EfficientNetV2 yielded some improvements on the original, so it should probably be added as a model.

**Develop support for time and date in the inferrence script**

The infer.py script does not take starting time into account at the moment.

**Make website for user interface**

Most end users of this system are probably not extremely tech savvy folks, so I think it would benefit the projet to add code for hosting a webservice where a user can drag and drop audio files and get their annotations sent by email or something like that.

**Add some docker stuff**

Sadly, not everyone runs linux; nah just kidding, but I do, and the codebase in this project assumes you do to, so I should probably add some docker stuff for this project so everyone can use it, despite using other operating systems.

**Optimize for inferrence speed**

The inferrence speed is, putting it lightly, somewhat abysmal (8 mins for 24h audio when inferring on a RTX3090). Most of the inferrence timing is used for creating spectrograms (educated guess), making huge spectrograms, slicing these for inferrence, is probably a better approach than creating num_hops spectrograms for every time sequence. Currently, model complexity is not the major bottleneck when inferring, it'll probably be a good idea to change this before making industrial sized practical applications of this piece of steaming hot garbage.
