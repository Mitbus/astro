from Encoder import Encoder
from Decoder import Decoder
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from scipy import signal
import math
import numpy as np
manual_seed = 12
random.seed(manual_seed)
torch.manual_seed(manual_seed)
import json
import time
import os
import wandb
import LearningRateTest


batch_size = 16
test_start = False
lr = 1e-5
deg_x = 8
deg_y = 7
shape_x = 2**deg_x
shape_y = 2**deg_y
beta1 = 0.5
inter_size = 500
loss_step = 10
test_step = 100
save_step = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training with', device)
wandb.init(project="astro", entity="ozil")


def data_loader(begin, end):
    epoch = 0
    while True:
        step = 0
        print(f'Epoch: {epoch}     ')
        for i in range(begin, end):
            wait = False
            while not (os.path.isfile(f'train/x_{i}.npy') and os.path.isfile(f'train/y_{i}.npy')):
                if not wait:
                    print(f'File y_{i}.npy or x_{i}.npy not exists. Waiting...')
                    wait = True
                time.sleep(5)
            if wait:
                print('Reading data:', i)
            xs = np.load(f'train/x_{i}.npy')
            ys = np.load(f'train/y_{i}.npy')
            for j in range(0, xs.shape[0], batch_size):
                x = torch.tensor(xs[j:j+batch_size], device=device, dtype=torch.float32)
                y = torch.tensor(ys[j:j+batch_size], device=device, dtype=torch.float32)
                mask = (torch.rand(size=x.shape, device=device, dtype=torch.float32) < 0.8).float()
                shift = torch.normal(mean=0, std=0.01, size=x.shape, device=device, dtype=torch.float32)
                yield (x + shift) * mask, y, step, epoch
                step += 1
        epoch += 1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.1))
        # nn.init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.1))
        # nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
        nn.init.constant_(m.bias.data, 0)


netE = torch.load('tmp/encoder_0_5000')
netD = torch.load('tmp/decoder_0_5000')

# netE = Encoder(dim=inter_size).to(device)
# netE.apply(weights_init)
# print(netE)
# netD = Decoder(dim=inter_size).to(device)
# netD.apply(weights_init)
# print(netD)

# test lr


def test():
    class Model(nn.Module):
        def __init__(self, encoder, decoder):
            super(Model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, input):
            # y, ind = self.encoder._forward_impl(input)
            # return self.decoder._forward_impl(y, ind)
            vec = self.encoder(input)
            return self.decoder(vec)
    def data_iterator():
        for x, y, step, epoch in data_loader(0, 1000):
            yield x,y

    model = Model(netE, netD)
    data = LearningRateTest.test(
        model,
        data_iterator(),
        lambda lr: optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3),
        nn.HuberLoss(),
        lr_low=1e-6,
        lr_max=1e3,
        mult=1.1
    )
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    smooth_loss = data[:,1] #smooth(data[:,1],15)
    ax.plot(data[:,0], smooth_loss)
    ax.set_xscale('log')
    fig.savefig(f'test_lr.png')
    plt.close(fig)
    for i in range(len(smooth_loss)):
        wandb.log({'loss': smooth_loss[i], 'lr': data[i,0]})

if test_start:
    print('Testing best learning rate')
    test()
    exit()


criterion1 = nn.HuberLoss()
criterion2 = nn.KLDivLoss()
alpha = 5
optimizerE = optim.Adam(netE.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
first_name = str(criterion1.__class__).split(".")[-1].split("'")[0]
second_name = str(criterion2.__class__).split(".")[-1].split("'")[0]


print('Processing test data...')
tx, ty, i, epoch = next(data_loader(0, 1))
losses = []
ma_param = loss_step * 5
ma_losses = []
test_first_losses = []
test_second_losses = []
test_total_losses = []

print('Testing nn topology...')
vec = netE(tx)
y_pred = netD(vec)

print("Starting Training Loop...")
for x,y, i, epoch in data_loader(1, 1000):
    
    netE.zero_grad()
    netD.zero_grad()
    
    vec = netE(x)
    y_pred = netD(vec)
    
    err = criterion1(y_pred, y)*alpha + criterion2(y_pred, y)
    err.backward()
    losses.append(err.item())
    
    optimizerE.step()
    optimizerD.step()

    wandb_log = {}
    if i % loss_step == 0:
        ma_losses.append(sum(losses) / len(losses))
        print('step', i, ma_losses[-1], end='\r')
        losses = losses[-ma_param:]
        wandb_log["train_losses"] = ma_losses[-1]
    if i % test_step == 0:
        with torch.no_grad():
            y_test = netD(netE(tx))
            err1 = criterion1(y_test, ty)
            err2 = criterion2(y_test, ty)
            test_first_losses.append(err1.item())
            test_second_losses.append(err2.item())
            test_total_losses.append((err1*alpha+err2).item())
        wandb_log[f'test_{first_name}_losses'] = test_first_losses[-1]
        wandb_log[f'test_{second_name}_losses'] = test_second_losses[-1]
        wandb_log[f'test_total_losses'] = test_total_losses[-1]
        
        print(f'step {i} test {first_name}:', test_first_losses[-1], f'\t test {second_name}:', test_second_losses[-1], '\t test total:', test_total_losses[-1])
    if i % save_step == 0:
        # alpha = - e2.item() / e1.item()
        # print(f'alpha set to {alpha}')
        torch.save(netE, f'tmp/encoder_{epoch}_{i}')
        torch.save(netD, f'tmp/decoder_{epoch}_{i}')

        fig = plt.figure(figsize=(16, 8))
        fig.add_subplot(2, 2, 1)  
        plt.plot(list(range(len(ma_losses))), ma_losses)
        plt.title('train_losses')

        fig.add_subplot(2, 2, 2)  
        plt.plot(list(range(len(test_first_losses))), test_first_losses)
        plt.title(f'test_{first_name}_losses')

        fig.add_subplot(2, 2, 3)  
        plt.plot(list(range(len(test_second_losses))), test_second_losses)
        plt.title(f'test_{second_name}_losses')

        fig.add_subplot(2, 2, 4)  
        plt.plot(list(range(len(test_total_losses))), test_total_losses)
        plt.title('test_total_losses')

        fig.savefig(f'tmp/report_{epoch}_{i}.png')
        plt.close(fig)
    if (epoch > 0 or i > 100) and len(wandb_log) != 0:
        wandb.log(wandb_log)
