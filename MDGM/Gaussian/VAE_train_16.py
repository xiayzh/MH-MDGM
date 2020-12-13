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
from load_data import load_data_1scale
from plot import plot_generation,plot_reconstruction
from  CONV_VAE_model import Encoder, Decoder

parser = argparse.ArgumentParser()
parser.add_argument('--exp',type = str ,default = 'Gaussian_16',help = 'dataset')
parser.add_argument("--n-epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument('--n-train', type=int, default=25000, help='number of training data')
parser.add_argument('--n-test', type=int, default=1000, help='number of test data')
parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample-interval", type=int, default=5, help="interval between image sampling")
parser.add_argument("--beta_vae", type=float, default=1, help="beta hyperparameter")
args = parser.parse_args()

dir = os.getcwd()
directory = f'/Gaussian/experiments/experiments_16/latent16/beta_{args.beta_vae}'
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
decoder = Decoder()
encoder.to(device)
decoder.to(device)
print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))


train_hdf5_file = os.getcwd() + \
    f'/Gaussian/data/training_set_16_gaussian.hdf5'
test_hdf5_file = os.getcwd() + \
    f'/Gaussian/data/test_set_16_gaussian.hdf5'
train_loader = load_data_1scale(train_hdf5_file, args.n_train, args.batch_size,singlescale=True)
with h5py.File(test_hdf5_file, 'r') as f:
    x_test = f['test'][()]
    x_test =x_test

optimizer= torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))

def test(epoch,x_test):
    encoder.eval()
    decoder.eval()
    z = torch.randn(9, 1, 4 ,4).to(device)
    imgs = decoder(z)
    samples = np.squeeze(imgs.data.cpu().numpy())
    plot_generation(samples,epoch,output_dir,1)

    real_imgs = x_test[[10,30,50,100]]
    real_imgs = (torch.FloatTensor(real_imgs)).to(device) 
    encoded_imgs,_,_ = encoder(real_imgs)
    decoded_imgs = decoder(encoded_imgs)
    samples_gen  = np.squeeze(decoded_imgs.data.cpu().numpy())
    samples_real = np.squeeze(real_imgs.data.cpu().numpy())

    samples = np.vstack((samples_real,samples_gen))
    plot_reconstruction(samples,epoch,output_dir)

def loss_function(recon_x, x, mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    BCE = F.mse_loss(recon_x.view(-1,256), x.view(-1,256), size_average=False)
    mu=mu.reshape(-1,16)
    logvar=logvar.reshape(-1,16)
    KLD = torch.sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    return BCE + args.beta_vae*KLD, BCE , KLD
# ----------#
#  Training #
# ----------#
for epoch in range(1,args.n_epochs+1):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch_idx, (data, ) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        recon_batch = decoder(z)
        loss,rec_loss, kl_loss= loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} recon_loss:{:.6f} kl_loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),rec_loss.item() / len(data),kl_loss.item() / len(data)))

    batches_done = epoch * len(train_loader) + batch_idx
    if (epoch) % args.sample_interval == 0:
        test(epoch,x_test)
        torch.save(decoder.state_dict(), model_dir + f'/decoder_16_VAE_{args.beta_vae}_epoch{epoch}.pth')
        torch.save(encoder.state_dict(), model_dir + f'/encoder_16_VAE_{args.beta_vae}_epoch{epoch}.pth')

