from models.resnet50.Encoder import Encoder
from models.resnet50.Decoder import Decoder
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import math
import numpy as np
manual_seed = 12
random.seed(manual_seed)
torch.manual_seed(manual_seed)
import json
import time
import os
import wandb
import test_lr
import multiprocessing



# config

wandb.init(project="astro", entity="ozil")
lr = 1e-3
max_data = 1000
inter_size = 1000
loss_step = 50
test_step = 500
save_step = 2500
edit_lr_step = 10_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training with', device)



# dataset

def data_loader(begin, end):
    epoch = 0
    total_step = 0
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
                yield x+noise+noise2, y, step, total_step, epoch
                step += 1
                total_step += 1
        epoch += 1



# build model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# netE = torch.load('tmp/m4_encoder_19_0')
# netD = torch.load('tmp/m4_decoder_19_0')

netE = Encoder(dim=inter_size).to(device)
# netE.apply(weights_init)
print(netE)
netD = Decoder(dim=inter_size).to(device)
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
            y, ind = self.encoder._forward_impl(input)
            return self.decoder._forward_impl(y, ind)
    def data_iterator():
        for x, y, step, total_step, epoch in data_loader(0, max_data):
            yield x,y
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    model = Model(netE, netD)
    data = test_lr.test(
        model,
        data_iterator(),
        lambda lr: optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4),
        nn.MSELoss(),
        lr_low=1e-6,
        lr_max=1e-3,
        mult=1.05
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

print('Testing best learning rate')
test()



# test model before start

criterion1 = nn.MSELoss()
# criterion2 = nn.KLDivLoss()
# alpha = 5
optimizerE = optim.Adam(netE.parameters(), lr=lr, weight_decay=5e-4)
optimizerD = optim.Adam(netD.parameters(), lr=lr, weight_decay=5e-4)
first_name = str(criterion1.__class__).split(".")[-1].split("'")[0]
# second_name = str(criterion2.__class__).split(".")[-1].split("'")[0]


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
vec, ind = netE._forward_impl(tx)
y_pred = netD._forward_impl(vec, ind)
wandb.config = {
  "learning_rate": lr,
  "batch_size": tx.shape[0]
}



# main training loop

print("Starting Training Loop...")
data_iter = data_loader(1, max_data)
for x,y,i,i_total, epoch in data_iter:
    loss_step = 1
    test_step = 1
    if i > 10:
        break
        
    netE.zero_grad()
    netD.zero_grad()

    vec, ind = netE._forward_impl(x)
    y_pred = netD._forward_impl(vec, ind)

    err = criterion1(y_pred, y)
    # e2 = criterion2(y_pred, y)
    # err = e1*alpha + e2
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
            vec, ind = netE._forward_impl(tx)
            y_test = netD._forward_impl(vec, ind)
            err1 = criterion1(y_test, ty)
            # err2 = criterion2(y_test, ty)
            test_first_losses.append(err1.item())
            # test_second_losses.append(err2.item())
            # test_total_losses.append((err1*alpha+err2).item())
        wandb_log[f'test_{first_name}_losses'] = test_first_losses[-1]
        # wandb_log[f'test_{second_name}_losses'] = test_second_losses[-1]
        # wandb_log['test_total_losses'] = test_total_losses[-1]
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

        # fig.add_subplot(2, 2, 3)  
        # plt.plot(list(range(len(test_second_losses))), test_second_losses)
        # plt.title(f'test_{second_name}_losses')

        # fig.add_subplot(2, 2, 4)  
        # plt.plot(list(range(len(test_total_losses))), test_total_losses)
        # plt.title('test_total_losses')

        fig.savefig(f'tmp/report_{epoch}_{i}.png')
        plt.close(fig)
    if i_total % edit_lr_step == 0:
        wandb_log['lr'] = lr
        cur_median_err = np.median(ma_losses)
        if last_lr_median_err * 0.95 <= cur_median_err:
            lr = lr / 2
            print('new lr:', lr, 'medians:', last_lr_median_err, cur_median_err)
            optimizerE = optim.Adam(netE.parameters(), lr=lr, weight_decay=5e-4)
            optimizerD = optim.Adam(netD.parameters(), lr=lr, weight_decay=5e-4)
        last_lr_median_err = cur_median_err
    if (epoch > 0 or i > 100) and len(wandb_log) != 0:
        wandb.log(wandb_log)
