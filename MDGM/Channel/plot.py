import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np
plt.switch_backend('agg')


# def plot_generation(samples, epoch, output_dir):
#     Ncol = 3
#     Nrow = samples.shape[0] // Ncol

#     fig, axes = plt.subplots(Nrow, Ncol, figsize=(Ncol*4, Nrow*2.1))
#     fs = 16
#     for j, ax in enumerate(fig.axes):
#         ax.set_aspect('equal')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if j < samples.shape[0]:
#             cax = ax.imshow(samples[j], cmap='jet')
#             cbar = plt.colorbar(cax, ax=ax, fraction=0.025, pad=0.04,
#                             format=ticker.ScalarFormatter(useMathText=True))
#             cbar.formatter.set_powerlimits((0, 0))
#             cbar.ax.yaxis.set_offset_position('left')
#             cbar.update_ticks()
#             cbar.ax.tick_params(axis='both', which='both', length=0)
#             cbar.ax.yaxis.get_offset_text().set_fontsize(fs-3)
#             cbar.ax.tick_params(labelsize=fs-2)
#     plt.savefig(output_dir+'/gen_epoch_{}.png'.format(epoch), bbox_inches='tight',dpi=600)
#     plt.close(fig)
#     print("epoch {}, done printing".format(epoch))

# def plot_recon_pred(samples, epoch, output_dir):
#     Ncol = 3
#     Nrow = samples.shape[0] // Ncol
#     fig, axes = plt.subplots(Nrow, Ncol, figsize=(Ncol*4, Nrow*2.1))
#     fs = 16
#     for j, ax in enumerate(fig.axes):
#         ax.set_aspect('equal')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         if j < samples.shape[0]:
#             cax = ax.imshow(samples[j], cmap='jet')
#             cbar = plt.colorbar(cax, ax=ax, fraction=0.025, pad=0.04,
#                             format=ticker.ScalarFormatter(useMathText=True))
#             cbar.formatter.set_powerlimits((0, 0))
#             cbar.ax.yaxis.set_offset_position('left')
#             cbar.update_ticks()
#             cbar.ax.tick_params(axis='both', which='both', length=0)
#             cbar.ax.yaxis.get_offset_text().set_fontsize(fs-3)
#             cbar.ax.tick_params(labelsize=fs-2)
#     plt.savefig(output_dir+'/recon_epoch_{}.png'.format(epoch), bbox_inches='tight',dpi=600)
#     plt.close(fig)
#     print("epoch {}, done printing".format(epoch))


def plot_generation(samples, epoch, output_dir):
    fig, _ = plt.subplots(3,3, figsize=(9, 9))
    vmin1 = np.amin(samples)
    vmax1 = np.amax(samples)
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_axis_off()
        cax = ax.imshow(samples[j],  cmap='jet', origin='upper',vmin=vmin1,vmax=vmax1)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
    plt.savefig(output_dir+'/gen_epoch_{}.png'.format(epoch), bbox_inches='tight',dpi=600)
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