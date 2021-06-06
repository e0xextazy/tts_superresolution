import os
import random
from pathlib import Path
import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram, MelSpectrogram, GriffinLim, AmplitudeToDB
from torch.utils.data import Dataset, DataLoader

def audio_extensions():
  return {'.m3u', '.ram', '.au', '.snd', '.mp3', '.mp2', '.aif', '.aifc', '.aiff', '.ra', '.wav', '.amr',
          '.awb', '.axa', '.csd', '.orc', '.sco', '.flac', '.mid', '.midi', '.kar', '.mpga', '.mpega', '.m4a',
          '.oga', '.ogg', '.opus', '.spx', '.sid', '.gsm', '.wma', '.wax', '.rm', '.pls', '.sd2'}

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_audio_files(path, recurse=True, followlinks=True):
    path = Path(path)
    extensions = audio_extensions()
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return res

class MonoAudioDataset(Dataset):
    def __init__(self, root_dir, duration=1000, freq=16000, transform=None):
        self.root_dir = root_dir
        self.audio_len = duration * freq // 1000 # Переводим их в длину тензора
        self.freq = freq
        self.audio_list = get_audio_files(root_dir)
        self.resample = Resample(new_freq=freq)
        self.transform = transform

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_path = self.audio_list[idx]
        audio, freq = torchaudio.load(audio_path)
        
        # Переводим в mono
        audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Подгоняем под одну частоту
        self.resample.orig_freq = freq
        audio = self.resample(audio)

        # Подгоняем под одну длительность, чтобы можно было склеить в батч
        if audio.size(-1) < self.audio_len:
            zeros_front = random.randint(0, self.audio_len - audio.size(-1))
            pad_front = torch.zeros((1, zeros_front))
            pad_back = torch.zeros((1, self.audio_len - audio.size(-1) - zeros_front))
            audio = torch.cat((pad_front, audio, pad_back), 1)
        elif audio.size(-1) > self.audio_len:
            crop_start = random.randint(0, int(audio.size(-1) - self.audio_len))
            audio = audio[:, crop_start : crop_start + self.audio_len]

        sample = {'audio': audio}

        if self.transform:
            sample = self.transform(sample)
        return sample

class AudioTransform(torch.nn.Module):
    # Не включать power в input_par, target_par т.к. AmplitudeToDB один power на всех!!! Или 2 AmplitudeToDB делать надо
    def __init__(self, db_rage=(-100, 80), input_par=dict(), target_par=dict(), griffin_lim_par=None, mel=False, device='cpu'):
        super(AudioTransform, self).__init__()
        self.min_db = db_rage[0]
        self.device = device
        if mel:
            self.input_spec = MelSpectrogram(**input_par).to(device)
            self.target_spec = MelSpectrogram(**target_par).to(device)
        else:
            self.input_spec = Spectrogram(**input_par).to(device)
            self.target_spec = Spectrogram(**target_par).to(device)

        if griffin_lim_par:
            self.griffin_lim = GriffinLim(**griffin_lim_par).to(device)
            self.g_spec = Spectrogram(**griffin_lim_par).to(device)
        else:
            self.griffin_lim = None
        
        self.amp2db = AmplitudeToDB(db_rage[1]).to(device)

    def forward(self, sample):
        audio = sample['audio'].to(self.device )
        input = audio
        if self.griffin_lim:
            input = self.g_spec(input)
            input = self.griffin_lim(input)

        input = self.amp2db(self.input_spec(input))
        target = self.amp2db(self.target_spec(audio))

        input = torch.clamp(input, min=self.min_db)
        target = torch.clamp(target, min=self.min_db)

        return {'input':input, 'target': target}
