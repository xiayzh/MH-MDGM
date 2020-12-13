import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np
plt.switch_backend('agg')


def plot_generation(samples, epoch, output_dir,number):
    fig, _ = plt.subplots(3,3, figsize=(9, 9))
    vmin1 = np.amin(samples)
    vmax1 = np.amax(samples)
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_axis_off()
        cax = ax.imshow(samples[j],  cmap='jet', origin='upper',vmin=vmin1,vmax=vmax1)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
    plt.savefig(output_dir+f'/gen_epoch_{epoch}_{number}.png', bbox_inches='tight',dpi=600)
    plt.close(fig)
    print("epoch {}, done printing".format(epoch))

def plot_reconstruction(samples, epoch, output_dir):
    fig, _ = plt.subplots(2,4, figsize=(12, 6))
    vmin1 = np.amin(samples)
    vmax1 = np.amax(samples)
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_axis_off()
        cax = ax.imshow(samples[j],  cmap='jet', origin='upper',vmin=vmin1,vmax=vmax1)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
    plt.savefig(output_dir+'/recon_epoch_{}.png'.format(epoch), bbox_inches='tight',dpi=600)
    plt.close(fig)
    print("epoch {}, done printing".format(epoch))