# functions to save or visualize irs to png et al

import matplotlib.pyplot as plt
import numpy as np
import wandb
from os import path
from PIL import Image as Im

def array_to_picture(npa, fullpath):
    scaled = np.uint8(255*npa)
    plt.imshow(scaled, cmap='hot', vmin=0, vmax=255)
    plt.savefig(fullpath)
    return fullpath


def save_ir(irs, recs, filename, savedir, truth=None):
    log_dict = cgrade_ir(irs, filename, savedir)
    log_dict.update(raw_ir(irs, filename, savedir))
    if truth is not None:
        log_dict.update(error_plot(truth, irs, filename, savedir))
    return log_dict

def cgrade_ir(ir, filename, savedir, channel=1):
    ir = np.uint8(255*ir[:, :, channel])
    plt.imshow(ir, cmap='hot', vmin=0, vmax=255)
    plt.savefig(path.join(savedir, filename))

    return {"ir": wandb.Image(path.join(savedir, filename))}

def raw_ir(ir, filename, savedir):
    filename = 'raw_' + filename
    pil_image = Im.fromarray(np.uint8(ir * 255))
    pil_image.save(path.join(savedir, filename))
    return {"raw_ir": wandb.Image(path.join(savedir, filename))}


def error_plot(target, prediction, filename, savedir, channel=1):
    target = target[:, :, channel]
    prediction = prediction[:, :, channel]

    # absolute error
    errorfn = path.join(savedir, 'error_' + filename)
    error = np.abs(target - prediction)
    errorpic = np.uint8(255*error)
    plt.imshow(errorpic, cmap='hot', vmin=0, vmax=255)
    plt.savefig(errorfn)

    # both in one image, layered in red and green channel
    layerfn = path.join(savedir, 'layer_' + filename)
    layer = np.stack((target, prediction, np.zeros_like(target)), -1)
    layerpic = np.uint8(255*layer)
    plt.imshow(layerpic, vmin=0, vmax=255)
    plt.savefig(layerfn)
    mse2psnr = lambda x: -10. * np.log(x) / np.log(10.)

    # shifting prediction left
    shifts = range(1, 5)
    shiftdict = dict()
    for s in shifts:
        # shift prediction left to see if we match better
        sprediction = np.concatenate((prediction[:, :-s], np.zeros_like(prediction)[:, :s]), axis=1)
        serror = np.abs(target - sprediction)
        shiftname = array_to_picture(serror, path.join(savedir, f"shift_{s}_error_" + filename))
        shiftdict.update({f"serror_{s}": wandb.Image(shiftname)})

    # argmax diff calc
    argmaxdiff = np.argmax(target, axis=1) - np.argmax(prediction, axis=1)
    mean_amd = np.mean(argmaxdiff)

    # mask error for ground truth positive (gtp) and ground truth zero (gtz) pixels
    mask = np.greater(target, 0)
    fperr = np.mean((error * mask) ** 2)
    fzerr = np.mean((error * np.invert(mask)) ** 2)
    to_log = {"error": wandb.Image(errorfn),
              "layered": wandb.Image(layerfn),
              "gtp_error": fperr,
              "gtp_psnr": mse2psnr(fperr),
              "gtz_error": fzerr,
              "gtz_psnr": mse2psnr(fzerr),
              "argmaxdiff_mean": mean_amd,
              "argmaxdiff": argmaxdiff}
    to_log.update(shiftdict)

    return to_log

