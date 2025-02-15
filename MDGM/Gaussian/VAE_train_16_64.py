import argparse
import os
import numpy as np
import itertools
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch
import h5py
from load_data import load_data_2scales
from plot import plot_generation,plot_reconstruction
from  CONV_VAE_model import Encoder, Decoder

parser = argparse.ArgumentParser()
parser.add_argument('--exp',type = str ,default = 'Channel_16_64',help = 'dataset')
parser.add_argument("--n-epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument('--n-train', type=int, default=25000, help='number of training data')
parser.add_argument('--n-test', type=int, default=1000, help='number of test data')
parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample-interval", type=int, default=1, help="interval between image sampling")
parser.add_argument("--beta_vae", type=float, default=2, help="beta hyperparameter")
args = parser.parse_args()

dir = os.getcwd()
directory = f'/Gaussian/experiments/experiments_16_64/latent256/beta_0.5_{args.beta_vae}'
exp_dir = dir + directory + "/N{}_Bts{}_Eps{}_lr{}".\
    format(args.n_train, args.batch_size, args.n_epochs, args.lr)
output_dir = exp_dir + "/predictions"
model_dir = exp_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
encoder = Encoder() 
decoder = Decoder(input_channels=2)
encoder.to(device)
decoder.to(device)
print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))


train_hdf5_file = os.getcwd() + \
    f'/Gaussian/data/training_set_64_gaussian.hdf5'
test_hdf5_file = os.getcwd() + \
    f'/Gaussian/data/test_set_64_gaussian.hdf5'
train_hdf5_file1 = os.getcwd() + \
    f'/Gaussian/data/training_set_16_gaussian.hdf5'
test_hdf5_file1 = os.getcwd() + \
    f'/Gaussian/data/test_set_16_gaussian.hdf5'
train_loader = load_data_2scales(train_hdf5_file,train_hdf5_file1, args.n_train, args.batch_size,singlescale=False)
with h5py.File(test_hdf5_file, 'r') as f:
    x_test = f['test'][()]
    x64_test =x_test 
with h5py.File(test_hdf5_file1, 'r') as f:
    x_test = f['test'][()]
    x16_test =x_test 

optimizer= torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))


def test(encoder_model,epoch,x64_test,x16_test):
    encoder.eval()
    decoder.eval()

    z = torch.randn(9, 1, 16, 16).to(device)
    _,z16,_ = X_l_encoder(encoder_model,(torch.FloatTensor(x16_test[150:151])).to(device))
    z16 = (torch.ones([9,1,4,4])).to(device)*z16
    z16 = upsample(z16,16)
    z_h = torch.cat((z16,z),1)
    gen_imgs = decoder(z_h)
    samples = np.squeeze(gen_imgs.data.cpu().numpy())
    plot_generation(samples,epoch,output_dir,150)
   
   
    real_imgs = x64_test[[10,30,50,100]]
    real_x16 = x16_test[[10,30,50,100]]
    z16,_,_ = X_l_encoder(encoder_model,(torch.FloatTensor(real_x16)).to(device))
    z16 = upsample(z16,16)
    real_imgs = (torch.FloatTensor(real_imgs)).to(device)
    encoded_imgs,_,_ = encoder(real_imgs)
    z_h = torch.cat((z16,encoded_imgs),1)
    decoded_imgs = decoder(z_h)
    samples_gen  = np.squeeze(decoded_imgs.data.cpu().numpy())
    samples_real = np.squeeze(real_imgs.data.cpu().numpy())
    samples = np.vstack((samples_real,samples_gen))
    plot_reconstruction(samples,epoch,output_dir)



def load_model():
        dir = os.getcwd()
        model_dir = dir+f'/Gaussian/encoder_16_VAE_0.5_epoch30.pth'
        model = Encoder()
        model.load_state_dict(torch.load(model_dir, map_location=device))
        encoder_model =model.to(device)
        return encoder_model

def X_l_encoder(encoder_model,xl_data):
        encoder_model.eval()
        z_l = encoder_model(xl_data)
        return z_l

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x.view(-1,4096), x.view(-1,4096), size_average=False)
    mu=mu.reshape(-1,256)
    logvar=logvar.reshape(-1,256)
    KLD = torch.sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    return BCE + args.beta_vae*KLD, BCE , KLD

class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()
        self.transup = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
    def forward(self,zl,size):
        if size == 16:
            zl = self.transup(zl)
            zl = self.transup(zl)
        return zl
# ----------#
#  Training #
# ----------#
upsample = upsample()
upsample.to(device)
encoder_model = load_model() 

for epoch in range(1,args.n_epochs+1):
    encoder.train()
    decoder.train()
    train_loss = 0   
    for batch_idx, (x64_data,x16_data) in enumerate(train_loader):
        x64_data = x64_data.to(device)
        x16_data = x16_data.to(device)
        optimizer.zero_grad() 
        z, mu, logvar = encoder(x64_data)
        _,z16,_ = X_l_encoder(encoder_model,x16_data)
        z16 = upsample(z16,16)
        z_h = torch.cat((z16,z),1)
        recon_batch = decoder(z_h)
        loss,rec_loss, kl_loss= loss_function(recon_batch, x64_data,  mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} recon_loss:{:.6f} kl_loss:{:.6f}'.format(
                epoch, batch_idx * len(x64_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(x64_data),rec_loss.item() / len(x64_data),kl_loss.item() / len(x64_data)))
    batches_done = epoch * len(train_loader) + batch_idx

    if (epoch) % args.sample_interval == 0:
        test(encoder_model,epoch,x64_test,x16_test)
        torch.save(decoder.state_dict(), model_dir + f'/decoder_16_64_VAE_0.5_{args.beta_vae}_epoch{epoch}.pth')
        torch.save(encoder.state_dict(), model_dir + f'/encoder_16_64_VAE_0.5_{args.beta_vae}_epoch{epoch}.pth')


