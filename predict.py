import argparse
import json
import os
import numpy
import matplotlib.pyplot as plt
from librosa.display import specshow
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import torchaudio
from torchaudio.transforms import GriffinLim

from src.dynamic_unet import prepare_resnet_encoder, create_my_dynamic_unet_from_resnet
from src.data import MonoAudioDataset, AudioTransform

sr = 16000
figsize=(8, 6)
img_ext = '.png'

def db2pow(S_db):
    ref = 1.0
    return ref * torch.pow(10.0, S_db / 10)

def get_args():
    parser = argparse.ArgumentParser(description='Predict the dynamic UNet on spectrograms',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--workers', metavar='W', type=int, default=1,
                        help='Number of workers', dest='workers')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.15,
                        help='Percent of the data that was used as validation (0-1)')
    parser.add_argument('-t', '--test', dest='test', type=float, default=0.05,
                        help='Percent of the data that was used as test (0-1)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=42,
                        help='Seed used for train/valis/test split)')
    parser.add_argument('-c', '--config', dest='conf', type=str, default='config/spectrogram_config.json',
                        help='Model config path')
    parser.add_argument('-d', '--data', dest='data', type=str, default='data/buriy_audiobooks_2_val',
                        help='Data path')
    parser.add_argument('-cp', '--checkpoint', dest='checkpoint', type=str, default='checkpoints/CP_best.pth',
                        help='Checkpoint path')
    parser.add_argument('-p', '--predicts', dest='predicts', type=str, default='data/predict',
                        help='Predicts path')
    parser.add_argument('-e', '--extension', dest='extension', type=str, default='.wav',
                        help='Sound extension')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conf = args.conf
    with open(conf) as json_file:
        conf = json.load(json_file)
    mel = conf['mel']
    min_db = conf['min_db']
    top_db = conf['top_db']
    duration = conf['duration']
    griffin_lim_par = conf['griffin_lim_par']
    input_par = conf['input_par']
    target_par = conf['target_par']

    transform=AudioTransform(mel=mel, db_rage=(min_db, top_db), input_par=input_par, target_par=target_par, griffin_lim_par=griffin_lim_par, device=device)
    dataset = MonoAudioDataset(root_dir=args.data, duration=duration, transform=None)

    n_val = int(len(dataset) * args.val)
    n_test = int(len(dataset) * args.test)
    n_train = len(dataset) - n_val - n_test
    _, _, test = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed))

    loader = DataLoader(test, batch_size=args.batchsize, num_workers=args.workers, pin_memory=True)

    model = models.resnet34(pretrained=False)
    model = prepare_resnet_encoder(model, split_rec=False)

    for audio_batch in loader:
        b = audio_batch
        break

    l = b['audio'].size(-1)
    gl_small = GriffinLim(length=l, n_fft=201, win_length=201, hop_length=50).to(device)
    gl_big = GriffinLim(length=l, **target_par).to(device)

    b = transform(b)
    input = b['input']
    target = b['target']

    size_input = input.shape[-2:]
    size_target = target.shape[-2:]

    net = create_my_dynamic_unet_from_resnet(model, (min_db, top_db), size_target, size_target)
    net = nn.Sequential(nn.Upsample(size=size_target, mode='bilinear'), net)

    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()
    net.to(device=device)

    try:
        os.mkdir(args.predicts)
    except OSError:
      pass

    count1 = count2 = 0
    for audio_batch in loader:
        batch = transform(audio_batch)
        orig = audio_batch['audio']
        input = batch['input']
        target = batch['target']
        
        with torch.no_grad():
            pred = net(input)

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=figsize)
        for i in range(pred.size(0)):
            # Spectrogram (Mel-spectrogram)
            ax1.set_title("Input")
            specshow(input[i, 0].cpu().numpy(), ax=ax1, hop_length=transform.input_spec.hop_length, sr=sr)
            ax2.set_title("Target")
            specshow(target[i, 0].cpu().numpy(), ax=ax2, hop_length=transform.target_spec.hop_length, sr=sr)
            ax3.set_title("Predict")
            specshow(pred[i, 0].cpu().numpy(), ax=ax3, hop_length=transform.target_spec.hop_length, sr=sr)
            plt.savefig(os.path.join(args.predicts, str(count1) + img_ext), dpi=150)
            count1 += 1

        # https://github.com/librosa/librosa/blob/a3103e025794e0df9794f34f461f1b401f0a1ab7/librosa/feature/inverse.py#L20
        #if not mel:
        input = db2pow(input)
        target = db2pow(target)
        pred = db2pow(pred)
        
        m = nn.Upsample(size=(201, 161), mode='nearest')
        input = m(input)

        input = gl_big(input).cpu()
        pred = gl_big(pred).cpu()
        target = gl_big(target).cpu()

        for i in range(pred.size(0)):
            orig_mean = torch.mean(orig[i])
            input_mean = torch.mean(input[i])
            target_mean = torch.mean(target[i])
            pred_mean = torch.mean(pred[i])

            orig[i] = orig[i] - orig_mean
            input[i] = input[i] - input_mean
            target[i] = target[i] - target_mean
            pred[i] = pred[i] - pred_mean

            orig_max = torch.max(orig[i])
            input_max = torch.max(input[i])
            target_max = torch.max(target[i])
            pred_max = torch.max(pred[i])

            orig[i] = orig[i] / orig_max
            input[i] = input[i] / input_max
            target[i] = target[i] / target_max
            pred[i] = pred[i] / pred_max

            torchaudio.save(os.path.join(args.predicts, str(count2) + '_orig' + args.extension), src=orig[i], sample_rate=sr)
            torchaudio.save(os.path.join(args.predicts, str(count2) + '_input' + args.extension), src=input[i], sample_rate=sr)
            torchaudio.save(os.path.join(args.predicts, str(count2) + '_target' + args.extension), src=target[i], sample_rate=sr)
            torchaudio.save(os.path.join(args.predicts, str(count2) + '_pred' + args.extension), src=pred[i], sample_rate=sr)
            count2 += 1
