# functions to save or visualize irs to png et al

import matplotlib.pyplot as plt
import numpy as np
import wandb
from os import path
from PIL import Image as Im

def save_ir(irs, recs, filename, savedir, truth=None):
    if recs is None:
        pil_image = Im.fromarray(np.uint8(irs * 255))
        pil_image.save(path.join(savedir, 'raw_' + filename))
    log_dict = cgrade_ir(irs, path.join(savedir, filename))
    if truth is not None:
        log_dict_2 = error_plot(truth, irs, path.join(savedir, 'error_' + filename))
        log_dict.update(log_dict_2)
    return log_dict

def cgrade_ir(ir, filename, channel=1):
    ir = np.uint8(255*ir[:, :, channel])
    plt.imshow(ir, cmap='hot', vmin=0, vmax=255)
    plt.savefig(filename)

    return {"ir": wandb.Image(filename)}


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

