from models.vgg_32.Encoder import Encoder
from models.vgg_32.Decoder import Decoder
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import time
import os
import wandb
import LearningRateTest
import multiprocessing
from DataLoader import data_loader
from Config import *


# init

random.seed(manual_seed)
torch.manual_seed(manual_seed)
wandb.init(project="astro", entity="ozil")
print('Training with', device)


# build model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# netE = torch.load('tmp/m4_encoder_19_0')
# netD = torch.load('tmp/m4_decoder_19_0')

netE = Encoder().to(device)
netE.apply(weights_init)
print(netE)
netD = Decoder().to(device)
netD.apply(weights_init)
print(netD)



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
        for x, y, step, total_step, epoch in data_loader(0, max_data):
            yield x,y
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    model = Model(netE, netD)
    data = LearningRateTest.test(
        model,
        data_iterator(),
        lambda lr: optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
        nn.HuberLoss(),
        lr_low=1e-12,
        lr_max=1e8,
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
    # test()



# test model before start

criterion1 = nn.HuberLoss()
if cirterion2_mult != 0:
    criterion2 = nn.KLDivLoss()
optimizerE = optim.Adam(netE.parameters(), lr=lr, weight_decay=5e-4, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, weight_decay=5e-4, betas=(0.5, 0.999))
first_name = str(criterion1.__class__).split(".")[-1].split("'")[0]
if cirterion2_mult != 0:
    second_name = str(criterion2.__class__).split(".")[-1].split("'")[0]


print('Processing test data...')
tx, ty, i, i_total, epoch = next(data_loader(0, 1))
losses = []
ma_param = loss_step * 5
ma_losses = []
test_first_losses = []
test_second_losses = []
test_total_losses = []
last_lr_median_err = np.inf

print('Testing nn topology...')
# vec, ind = netE._forward_impl(tx)
# y_pred = netD._forward_impl(vec, ind)
y_pred = netD(netE(tx))
wandb.config = {
  "learning_rate": lr,
  "batch_size": tx.shape[0]
}



# main training loop

# y_old = ty
print("Starting Training Loop...")
data_iter = data_loader(1, max_data, noise=True)
for x,y,i,i_total, epoch in data_iter:
    if test_start:
        loss_step = 1
        test_step = 1
        if i > 10:
            break
        
    netE.zero_grad()
    netD.zero_grad()

    # vec, ind = netE._forward_impl(x)
    # y_pred = netD._forward_impl(vec, ind)
    vec = netE(x)
    y_pred = netD(vec)

    if cirterion2_mult != 0:
        err = criterion1(y_pred, y) + criterion2(y_pred, y)*cirterion2_mult
    else:
        err = criterion1(y_pred, y)
    err.backward()
    losses.append(err.item())

    optimizerE.step()
    optimizerD.step()

    wandb_log = {}
    if i % loss_step == 0:
        ma_losses.append(np.median(losses))
        print('step', i, ma_losses[-1], end='\r')
        losses = losses[-ma_param:]
        wandb_log["train_losses"] = ma_losses[-1]
    if i % test_step == 0:
        with torch.no_grad():
            # vec, ind = netE._forward_impl(tx)
            # y_test = netD._forward_impl(vec, ind)
            vec = netE(tx)
            y_test = netD(vec)
            err1 = criterion1(y_test, ty)
            if cirterion2_mult != 0:
                err2 = criterion2(y_test, ty)
            test_first_losses.append(err1.item())
            if cirterion2_mult != 0:
                test_second_losses.append(err2.item())
                test_total_losses.append((err1 + err2 * cirterion2_mult).item())
        wandb_log[f'test_{first_name}_losses'] = test_first_losses[-1]
        if cirterion2_mult != 0:
            wandb_log[f'test_{second_name}_losses'] = test_second_losses[-1]
            wandb_log['test_total_losses'] = test_total_losses[-1]
        print('step', i, 'total', i_total, wandb_log)
    if i % save_step == 0:
        torch.save(netE, f'tmp/encoder_{epoch}_{i}')
        torch.save(netD, f'tmp/decoder_{epoch}_{i}')

        fig = plt.figure(figsize=(16, 8))
        fig.add_subplot(2, 2, 1)  
        plt.plot(list(range(len(ma_losses))), ma_losses)
        plt.title('train_losses')

        fig.add_subplot(2, 2, 2)  
        plt.plot(list(range(len(test_first_losses))), test_first_losses)
        plt.title(f'test_{first_name}_losses')

        if cirterion2_mult != 0:
            fig.add_subplot(2, 2, 3)
            plt.plot(list(range(len(test_second_losses))), test_second_losses)
            plt.title(f'test_{second_name}_losses')

            fig.add_subplot(2, 2, 4)
            plt.plot(list(range(len(test_total_losses))), test_total_losses)
            plt.title('test_total_losses')

        fig.savefig(f'tmp/report_{epoch}_{i}.png')
        plt.close(fig)
    if i_total % edit_lr_step == 0:
        wandb_log['lr'] = lr
        cur_median_err = np.median(ma_losses)
        if last_lr_median_err * 0.95 <= cur_median_err:
            lr = lr * lr_mult
            print('new lr:', lr, 'medians:', last_lr_median_err, cur_median_err)
            optimizerE = optim.Adam(netE.parameters(), lr=lr, weight_decay=5e-4)
            optimizerD = optim.Adam(netD.parameters(), lr=lr, weight_decay=5e-4)
        last_lr_median_err = cur_median_err
    if (epoch > 0 or i > 100) and len(wandb_log) != 0:
        wandb.log(wandb_log)
    # y_old = y
