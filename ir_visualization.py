# functions to save or visualize irs to png et al

import matplotlib.pyplot as plt
import numpy as np
import wandb
from os import path
from PIL import Image as Im

def array_to_figure(npa, title, xlabel, ylabel, cmap=None, fullpath=None):
    fig, ax = plt.subplots(1, 1, dpi=400, figsize=(10, 5))
    ax.set(title=title,
           xlabel=xlabel,
           ylabel=ylabel)
    if npa.ndim == 2:
        pc = ax.pcolormesh(npa, vmin=0, vmax=1, cmap=cmap)
        if cmap is not None:
            fig.colorbar(pc, shrink=0.6, ax=ax, location='right')
    elif npa.ndim == 3:
        npa = np.flip(npa, axis=0)  # flip to match with figures
        ax.imshow(npa, vmin=0, vmax=1)
    else:
        print(f"Error creating figure. Invalid shaped array with dim {npa.ndim}.")
    if fullpath:
        plt.savefig(fullpath)
    return wandb.Image(fig)


def save_ir(irs, recs, filename, savedir, truth=None):
    log_dict = cgrade_ir(irs, filename, savedir)
    log_dict.update(raw_ir(irs, filename, savedir))
    if truth is not None:
        log_dict.update(error_plot(truth, irs, filename, savedir))
    return log_dict

def cgrade_ir(ir, filename, savedir, channel=1):
    filename += '.pdf'
    f = array_to_figure(ir[:, :, channel], 'ground truth impulse responses', 'timestep', 'listener_id', cmap='hot', fullpath=path.join(savedir, filename))
    return {"ir": f}

def raw_ir(ir, filename, savedir):
    filename += '.png'
    filename = 'raw_' + filename
    # flip ir so ir sorting matches matplotlib output!!
    ir = np.flip(ir, axis=0)
    pil_image = Im.fromarray(np.uint8(ir * 255))
    pil_image.save(path.join(savedir, filename))
    return {"raw_ir": wandb.Image(path.join(savedir, filename))}


def error_plot(target, prediction, filename, savedir, channel=1):
    target = target[:, :, channel]
    prediction = prediction[:, :, channel]

    # absolute error
    errorfn = path.join(savedir, 'error_' + filename)
    error = np.abs(target - prediction)
    errorfig = array_to_figure(error, 'absolute error between target and prediction', 'timestep', 'listener_id', cmap='hot', fullpath=errorfn)

    # both in one image, layered in red and green channel
    layerfn = path.join(savedir, 'layer_' + filename)
    layer = np.stack((target, prediction, np.zeros_like(target)), -1)
    layerfig = array_to_figure(layer, 'target (red) and prediction (green)', 'timestep', 'listerner_id', fullpath=layerfn)

    # argmax diff calc
    argmaxdiff = np.argmax(target, axis=1) - np.argmax(prediction, axis=1)
    mean_amd = np.mean(argmaxdiff)

    mse2psnr = lambda x: -10. * np.log(x) / np.log(10.)
    # mask error for ground truth positive (gtp) and ground truth zero (gtz) pixels
    mask = np.greater(target, 0)
    fperr = np.mean((error * mask) ** 2)
    fzerr = np.mean((error * np.invert(mask)) ** 2)
    to_log = {"error": errorfig,
              "layered": layerfig,
              "gtp_error": fperr,
              "gtp_psnr": mse2psnr(fperr),
              "gtz_error": fzerr,
              "gtz_psnr": mse2psnr(fzerr),
              "argmaxdiff_mean": mean_amd,
              "argmaxdiff": argmaxdiff}

    return to_log

