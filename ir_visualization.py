# functions to save or visualize irs to png et al

import matplotlib.pyplot as plt
import numpy as np
import wandb
from os import path
from PIL import Image as Im

def save_ir(irs, recs, filename, savedir, truth=None):

    log_dict = cgrade_ir(irs, path.join(savedir, filename))
    if recs is None:
        log_dict.update(raw_ir(irs, path.join(savedir, 'raw_' + filename)))
    if truth is not None:
        log_dict.update(error_plot(truth, irs, path.join(savedir, 'error_' + filename)))
    return log_dict

def cgrade_ir(ir, filename, channel=1):
    ir = np.uint8(255*ir[:, :, channel])
    plt.imshow(ir, cmap='hot', vmin=0, vmax=255)
    plt.savefig(filename)

    return {"ir": wandb.Image(filename)}

def raw_ir(ir, filename):
    pil_image = Im.fromarray(np.uint8(ir * 255))
    pil_image.save(filename)
    return {"ground_truth": wandb.Image(filename)}


def error_plot(target, prediction, filename, channel=1):
    error = np.abs(target - prediction)
    errorpic = np.uint8(255*error[:, :, channel])
    plt.imshow(errorpic, cmap='hot', vmin=0, vmax=255)
    plt.savefig(filename)
    mse2psnr = lambda x: -10. * np.log(x) / np.log(10.)

    mask = np.greater(target, 0)
    fnerr = np.mean((error * mask) ** 2)
    fperr = np.mean((error * np.invert(mask)))
    return {"error": wandb.Image(filename),
            "gtp_error": fnerr,
            "gtp_psnr": mse2psnr(fnerr),
            "gtn_error": fperr,
            "gtn_psnr": mse2psnr(fperr)}

