import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd

from train import train_net
from src.data import MonoAudioDataset, AudioTransform
from src.dynamic_unet import prepare_resnet_encoder, create_my_dynamic_unet_from_resnet

l_epochs = [25, 50, 75, 100]
l_batch_size = [32, 64, 128]
l_lr = [1e-1, 1e-2, 1e-3]
l_min_db = [-70, -50, -30]
l_top_db = [80, 100, 150]

duration = 1000
mel = False
griffin_lim_par = {}
input_par = {
        "n_fft": 256,
        "hop_length": 256,
        "win_length": 256
    }
target_par = {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512
    }
data = 'data/buriy_audiobooks_2_val'

results = []
col_names = ['min_db', 'top_db', 'epochs', 'bs', 'lr', 'loss']

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info('Using device {}'.format(device))
    print('Using device {}'.format(device))
    count = 0
    for min_db in l_min_db:
        for top_db in l_top_db:
            for epochs in l_epochs:
                for batch_size in l_batch_size:
                    for lr in l_lr:
                        transform=transform=AudioTransform(mel=mel, db_rage=(min_db, top_db), input_par=input_par, target_par=target_par, griffin_lim_par=griffin_lim_par)
                        dataset = MonoAudioDataset(root_dir=data, duration=duration, transform=transform)

                        model = models.resnet34(pretrained=False)
                        model = prepare_resnet_encoder(model, split_rec=False)

                        size_input = dataset[0]['input'].shape[-2:]
                        size_target = dataset[0]['target'].shape[-2:]

                        net = create_my_dynamic_unet_from_resnet(model, (min_db, top_db), size_target, size_target)
                        net = nn.Sequential(nn.Upsample(size=size_target, mode='bilinear'), net)
                        net.to(device=device)

                        cell = []
                        loss = train_net(dataset=dataset,
                                    net=net,
                                    num_workers=11,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    lr=lr,
                                    device=device,
                                    val_percent=0.15,
                                    test_percent=0.05,
                                    split_seed=55,
                                    count=count,
                                    cp_path='checkpoints')
                        results.append([min_db, top_db, epochs, batch_size, lr, loss])
                        count += 1
    df = pd.DataFrame(results, columns=col_names)
    df.to_csv('result.csv')
                        