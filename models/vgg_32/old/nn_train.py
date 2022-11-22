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


batch_size = 16
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


def data_loader(begin, end):
    epoch = 0
    while True:
        step = 0
        print(f'Epoch: {epoch}     ')
        time.sleep(15)
        for i in range(begin, end):
            wait = False
            while not (os.path.isfile(f'train/x_{i}.txt') and os.path.isfile(f'train/y_{i}.txt')):
                if not wait:
                    print(f'File y_{i}.txt or x_{i}.txt not exists. Waiting...')
                    wait = True
                time.sleep(5)
            if wait:
                print('Reading data:', i)
            with open(f'train/x_{i}.txt') as f:
                xs = json.loads(f.read())
            with open(f'train/y_{i}.txt') as f:
                ys = json.loads(f.read())
            while len(xs) != 0:
                x = torch.tensor(xs.pop(), device=device, dtype=torch.float32)
                y = torch.tensor(ys.pop(), device=device, dtype=torch.float32)
                noise = torch.normal(mean=0, std=0.5, size=x.shape, device=device, dtype=torch.float32)
                noise2 = (torch.rand(size=x.shape, device=device, dtype=torch.float32) < 0.9).float()
                yield x+noise+noise2, y, step, epoch
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


# netE = torch.load('tmp/encoder_19_0')
# netD = torch.load('tmp/decoder_19_0')

netE = Encoder(dim=inter_size).to(device)
netE.apply(weights_init)
print(netE)
netD = Decoder(dim=inter_size).to(device)
netD.apply(weights_init)
print(netD)


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
for x,y, i, epoch in data_loader(1, 500):
    
    netE.zero_grad()
    netD.zero_grad()
    
    vec = netE(x)
    y_pred = netD(vec)
    
    err = criterion1(y_pred, y)*alpha + criterion2(y_pred, y)
    err.backward()
    losses.append(err.item())
    
    optimizerE.step()
    optimizerD.step()

    if i % loss_step == 0:
        ma_losses.append(sum(losses) / len(losses))
        print('step', i, ma_losses[-1], end='\r')
        losses = losses[-ma_param:]
    if i % test_step == 0:
        with torch.no_grad():
            y_test = netD(netE(tx))
            err1 = criterion1(y_test, ty)
            err2 = criterion2(y_test, ty)
            test_first_losses.append(err1.item())
            test_second_losses.append(err2.item())
            test_total_losses.append((err1*alpha+err2).item())
            
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
