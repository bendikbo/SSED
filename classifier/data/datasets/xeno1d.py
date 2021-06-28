import os
import torch
import torchaudio
import numpy as np
import time
from torchaudio import transforms
from torch.utils.data import Dataset
from classifier.utils import to_cuda


class XenoDataset1D(Dataset):
    """
    Dataset class for Xeno Canto dataset
    """
    def __init__(
            self,
            data_dir,
            norm = True,
            stoch_factor = 1.0,
            rand_crop = 0.0,
            rand_flip = 0.0,
            rand_noise = 0.0,
            rand_eq = 0.0,
            rand_amp = 0.0,
            rand_contrast = 0.0,
            rand_overdrive = 0.0
            ):
        self.stoch_factor = stoch_factor
        self.norm = norm
        self.rand_crop = rand_crop
        self.rand_flip = rand_flip
        self.rand_noise = rand_noise
        self.rand_eq = rand_eq
        self.rand_amp = rand_amp
        self.rand_contrast = rand_contrast
        self.rand_overdrive = rand_overdrive
        self.files = sorted(os.listdir(data_dir))
        self.data_dir = data_dir
        i = 0
        self.label_dict = {}
        for file_name in self.files:
            if file_name.split("_")[0] not in self.label_dict.keys():
                self.label_dict[file_name.split("_")[0]] = i
                i += 1
        print(self.label_dict)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.files[idx]
        data, _ = torchaudio.load(self.data_dir + file_name)
        data = to_cuda(data)
        #Cropping tensor to middle or random
        #start_time = time.time_ns()
        if data.size(1) > 65536:
            start = int(np.floor((data.size(1)-65536)/2))
            if self.rand_crop:
                start = np.random.randint(low=0, high=data.size(1)-65536)
            data = data.narrow(1, start, 65536)
        #end_time = time.time_ns()
        #print("ns to crop: {}".format(start_time - end_time))
        #start_time = time.time_ns()
        #Random flip
        if np.random.uniform() < self.rand_flip:
            data.mul(-1)
        if np.random.uniform() < self.rand_noise:
            std = to_cuda(data.std())
            data += to_cuda(torch.randn(data.size())) * std.mul(self.stoch_factor)
        if np.random.uniform() < self.rand_eq:
            data = torchaudio.functional.equalizer_biquad(
                waveform = data,
                sample_rate = 16000,
                center_freq = np.random.uniform(low=250, high=7000),
                gain = np.random.uniform(low=-5*self.stoch_factor,high=5*self.stoch_factor)
            )
        if np.random.uniform() < self.rand_contrast:
            data = torchaudio.functional.contrast(
                waveform = data,
                enhancement_amount = np.random.uniform(low=75, high=75)
            )
        if np.random.uniform() < self.rand_overdrive:
            data = torchaudio.functional.overdrive(
                waveform = data,
                gain = np.random.uniform(low = 0, high = 10),
                colour = np.random.uniform(low = 0, high = 10)
            )
        if np.random.uniform() < self.rand_amp:
            data.mul(np.random.uniform(
                low=np.power(0.8, self.stoch_factor),
                high = np.power(1.2, self.stoch_factor))
                )
        if self.norm:
            data = data.unsqueeze(0)
            std = torch.std(data)
            normalize = transforms.Normalize(0, (std))
            normalize(data)
            data = data.squeeze()
            data = data.unsqueeze(0)
        label = self.label_dict[file_name.split("_")[0]]
        sample = (data, label)
        return sample