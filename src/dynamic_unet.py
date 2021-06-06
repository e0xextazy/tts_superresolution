import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, ch):
        super(Bottleneck, self).__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(ch), # На случай если в конце encoder'а не было BatchNorm2d, а если был, то ничего не испортит
            nn.ReLU(),
            # Middle conv1
            nn.Conv2d(ch, ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),
            # Middle conv2
            nn.Conv2d(ch*2, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.body(x)

class ExtraUp(nn.Module):
    def __init__(self, up_ch):
        super(ExtraUp, self).__init__()
        if up_ch % 2 !=0:
            raise 'PixelShuffle сломается, т.к. уменьшает число каналов в 4 раза'
        self.body = nn.Sequential(
            nn.Conv2d(up_ch, up_ch*2, kernel_size=1),
            nn.BatchNorm2d(up_ch*2),
            nn.ReLU(),
            nn.PixelShuffle(2)
            )

    def forward(self, x):
        return self.body(x)

class DecoderLayer(nn.Module):
    def __init__(self, up_ch, skip_ch, mode, last=False):
        super(DecoderLayer, self).__init__()
        self.mode = mode
        ni = up_ch//2 + skip_ch
        nf = ni//2
        self.up = ExtraUp(up_ch)

        self.body = nn.Sequential(
            nn.BatchNorm2d(ni),
            nn.Conv2d(ni, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.Conv2d(nf, nf if not last else nf-1, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf if not last else nf-1),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.size()[-2:], mode='nearest')
        x = torch.cat([x, skip], 1)
        x = self.body(x)
        return x

class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super(SigmoidRange, self).__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return x.sigmoid() * (self.high - self.low) + self.low

class DynamicUnet(nn.Module):
    def __init__(self, range_db, encoder, bottleneck_arch, decoder_arch, final_up, input_size, target_size):
        super(DynamicUnet, self).__init__()
        self.encoder = encoder
        self.size_change_ids = list()
        sizes = list()
        tensor = torch.zeros(1, 1, *input_size)
        for i, layer in enumerate(encoder.children()):
            new_tensor = layer(tensor)
            if tensor.size()[-2:] != new_tensor.size()[-2:]:
                self.size_change_ids.append(i)
                sizes.append(tensor.size())
            tensor = new_tensor

        self.middle = bottleneck_arch(tensor.size(1))
        tensor = self.middle(tensor)

        self.size_change_ids.reverse()
        sizes.reverse()

        dec_layers = list()
        for i, id in enumerate(self.size_change_ids):
            last = i==len(self.size_change_ids)-1
            dec_l = decoder_arch(tensor.size(1), sizes[i][1], last)
            tensor = dec_l(tensor, torch.zeros(sizes[i]))
            dec_layers.append(dec_l)
        self.dec_layers = nn.ModuleList(dec_layers)

        tensor = torch.cat([tensor, torch.zeros(1, 1, *input_size)], dim=1) # TODO надо отсюда

        ups = list()
        while tensor.size(-2)<target_size[-2] or tensor.size(-1)<target_size[-1]:
            up = final_up(tensor.size(1))
            ups.append(up)
            tensor = up(tensor)

        ups.append(nn.Upsample(target_size[-2:]))
        self.ups = nn.Sequential(*ups)
        # TODO вот сюда!
        nf = tensor.size(1)
        self.final = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.Conv2d(nf, nf , kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.Conv2d(tensor.size(1), 1, kernel_size=1),
            SigmoidRange(*range_db)
        )

    def forward(self, input):
        x = input
        skips = list()
        for i, layer in enumerate(self.encoder.children()):
            if i in self.size_change_ids:
                skips.append(x)
            x = layer(x)

        x = self.middle(x)

        for layer in self.dec_layers:
            skip = skips.pop()
            x = layer(x, skip)

        x = torch.cat([x, input], dim=1)
        x = self.ups(x)

        return self.final(x)

def prepare_resnet_encoder(resnet_model, n_channels=1, split_rec=False):
    model = resnet_model

    model = nn.Sequential(*list(model.children())[:-2])

    # Делаем 1 канальные входные данные
    layer = model[0]
    layer.in_channels=n_channels
    layer.weight = nn.Parameter(layer.weight[:,1,:,:].unsqueeze(1))
    model[0] = layer

    if split_rec:
        l = list()
        for layer in model:
            if type(layer) == nn.Sequential:
                l += list(layer)
            else:
                l.append(layer)
        model = nn.Sequential(*l)

    return model

def create_my_dynamic_unet_from_resnet(encoder, range_db, input_size, target_size):
    return DynamicUnet(range_db, encoder, Bottleneck, DecoderLayer, ExtraUp, input_size, target_size)
