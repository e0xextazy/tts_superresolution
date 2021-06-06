import argparse
import logging
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data import MonoAudioDataset, AudioTransform
from src.dynamic_unet import prepare_resnet_encoder, create_my_dynamic_unet_from_resnet

save_cp_policy = 'best' # Or 'all'

def train_net(dataset, net, device, epochs, num_workers, batch_size, lr, val_percent, test_percent, split_seed, cp_path):
    n_val = int(len(dataset) * val_percent)
    n_test = int(len(dataset) * test_percent)
    n_train = len(dataset) - n_val - n_test
    train, val, test = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(split_seed))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loss = []
    train_loss =[]
    
    writer = SummaryWriter(comment='LR_{}_BS_{}'.format(lr, batch_size))

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Training size:   {}
        Validation size: {}
        Test size:       {}
        Checkpoints:     {}
        Device:          {}
    '''.format(epochs, batch_size, lr, n_train, n_val, n_test, save_cp_policy, device.type))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = nn.MSELoss()

    global_step = 0
    best_val_epoch = -1
    best_val_score = 1e10

    for epoch in range(epochs):
        net.train()
        t_loss = 0

        # Train
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_loader:
                input = batch['input'].to(device=device)
                target = batch['target'].to(device=device)

                pred = net(input)
                loss = criterion(pred, target)
                t_loss = loss.item()
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(input.shape[0])
                global_step += 1
        train_loss.append(t_loss)
        scheduler.step()

        # Valid
        val_score = eval_net(net, val_loader, device)
        val_loss.append(val_score.cpu())
        if val_score < best_val_score:
            best_val_score = val_score
            best_val_epoch = epoch

        logging.info('Validation MSELoss: {}'.format(val_score))
        writer.add_scalar('Loss/valid', val_score)

        try:
            os.mkdir(cp_path)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

        if best_val_epoch == epoch:
            logging.info('New best checkpoint saved at epoch {} !'.format(epoch + 1))
            torch.save(net.state_dict(), os.path.join(cp_path, 'CP_best.pth'))
        if save_cp_policy == 'all':
            logging.info('Checkpoint {} saved !'.format(epoch + 1))
            torch.save(net.state_dict(), os.path.join(cp_path, 'CP_epoch{}.pth'.format(epoch + 1)))
        elif save_cp_policy == 'best':
            logging.info('Last checkpoint saved at epoch {} !'.format(epoch + 1))
            torch.save(net.state_dict(), os.path.join(cp_path, 'CP_last.pth'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(
        xlim = [-1, epochs],
        title = 'Функция потерь на тренировочном и валидационном наборах данных',
        xlabel = 'Эпохи',
        ylabel = 'Функция потерь')
    ax.plot(range(epochs), train_loss, label = 'Тренировочные данные')
    ax.plot(range(epochs), val_loss, label = 'Валидационные данные')
    ax.legend()
    plt.savefig('loss.png', dpi=150)

    # Test
    test_score = eval_net(net, test_loader, device, test=True)

    logging.info('Test MSELoss (last): {}'.format(test_score))
    writer.add_scalar('Loss/test (last)', test_score)

    # Test best epoch
    if best_val_epoch + 1 != epochs:
        net.load_state_dict(torch.load(os.path.join(cp_path, 'CP_best.pth')))
        test_score = eval_net(net, test_loader, device, test=True)

        logging.info('Test MSELoss (best) at epoch {}: {}'.format(best_val_epoch + 1, test_score))
        writer.add_scalar('Loss/test (best) at epoch', best_val_epoch + 1, test_score)

    writer.close()

def eval_net(net, loader, device, test=False):
    net.eval()
    criterion = nn.MSELoss()
    n_val = len(loader)  # the number of batch
    tot = 0
    desc = 'Test round' if test else 'Validation round'
    with tqdm(total=n_val, desc=desc, unit='batch', leave=False) as pbar:
        for batch in loader:
            input = batch['input'].to(device=device)
            target = batch['target'].to(device=device)

            with torch.no_grad():
                pred = net(input)

            tot += criterion(pred, target)

            pbar.update()
    net.train()
    return tot / n_val

def get_args():
    parser = argparse.ArgumentParser(description='Train the dynamic UNet on spectrograms',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-w', '--workers', metavar='W', type=int, default=1,
                        help='Number of workers', dest='workers')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.15,
                        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-t', '--test', dest='test', type=float, default=0.05,
                        help='Percent of the data that is used as test (0-1)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=42,
                        help='Seed for train/valis/test split')
    parser.add_argument('-c', '--config', dest='conf', type=str, default='config/spectrogram_config.json',
                        help='Model config path')
    parser.add_argument('-d', '--data', dest='data', type=str, default='data/buriy_audiobooks_2_val',
                        help='Data path')
    parser.add_argument('-cp', '--checkpoints', dest='checkpoints', type=str, default='checkpoints',
                        help='Checkpoints path')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

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

    transform=transform=AudioTransform(mel=mel, db_rage=(min_db, top_db), input_par=input_par, target_par=target_par, griffin_lim_par=griffin_lim_par)
    dataset = MonoAudioDataset(root_dir=args.data, duration=duration, transform=transform)

    model = models.resnet34(pretrained=False)
    model = prepare_resnet_encoder(model, split_rec=False)

    size_input = dataset[0]['input'].shape[-2:]
    size_target = dataset[0]['target'].shape[-2:]

    logging.info('''Model info:
        Input size:  {}
        Target size: {}
    '''.format(size_input, size_target))

    net = create_my_dynamic_unet_from_resnet(model, (min_db, top_db), size_target, size_target)
    net = nn.Sequential(nn.Upsample(size=size_target, mode='bilinear'), net)
    net.to(device=device)

    try:
        train_net(dataset=dataset,
                  net=net,
                  num_workers=args.workers,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val,
                  test_percent=args.test,
                  split_seed=args.seed,
                  cp_path=args.checkpoints)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
