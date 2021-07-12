# SSED - Sliding-window Sound Event Detection

This repository contains a sliding window sound event detection system developed for a master thesis at NTNU IES, with contributions by NINA - Norsk Institutt for NAturfroskning.

The codebase contains multiple backbones, and is (almost) fully configurable in both training and inference through yaml specified configurations.
This codebase should be easily extendable and reusable for most sound event detection tasks, and I hope you get good use of it, just remember I've licensed it under the MIT License.

# Example application

I've already added a dataset, and the standard cfg-object in classifier/config/defaults.py should lead to the main training script automatically downloading this dataset and starting a training session once downloaded and extracted. So the full list of bash terminal commands to train a (somewhat) state of the art sound event detection system for the bird sounds in the dataset should be as simple as:

```bash
git clone https://github.com/bendikbo/SSED.git
mkdir env
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
python train.py configs/default.yaml
```
assuming you have a somewhat new version of python3 already installed.


# Project description

TODO: add pdf of project thesis


# Docker setup

TODO: add description of how to set up docker

# Data augmentation and feature engineering

The AudioTransformer class, found within classifier/data/transform/transforms.py is an extendable, reusable, automatically configurable tool that comes with some limitations and expectations of added functionality. To set the stage for possible reusability and/or extendability, it's useful to mention some of the assumptions the class makes if new transforms are to be added to it.

**Transformation object forward arguments**

The forwarding arguments for every transformation class is assumed to be x, lines, and labels. X is currently always assumed to be a single channel 1D tensor, which is the reason that the spectrogram transformation is listed last in both transform_dicts, since the spectrify transform converts returns a single or multiple channel 2D tensors (which in all fairness means the tensor is 3D in the case of mutliple channels, but that's just a pedantic remark). Lines and labels are lists that, given an indice i, lines[i] contains a tuple with (onset, offset) of an audio event, and labels[i] contain the label for the aforementioned audio event.

**Support for pure classification problems**

In the interest of supporting pure classification problems, especially mutual exclusive classification problems, where neither lines nor labels probably need to be forwarded, as in most cases of such problems the class of the underlying data remains constant no matter how it's augmented, there is an option to only forward the x-tensor through any instance of the AudioTransformer-class. Therefore, each class currently needs to have a special case where lines=None and labels=None, where the module does not do any line or label readjustment that would otherwise be necessary with transforms like the time shifting transform. This means that if lines and labels are to be augmented, it's best to leave an if statement in the style of the following snippet.
