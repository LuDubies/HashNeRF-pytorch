# functions to save or visualize irs to png et al

import matplotlib.pyplot as plt
import numpy as np
import wandb

def cgrade_ir(ir, filename, channel=1, upload=False):
    ir = np.uint8(255*ir[:, :, channel])
    plt.imshow(ir, cmap='hot', vmin=0, vmax=255)
    plt.savefig(filename)
    if upload:
        wandb.log({"ir": wandb.Image(filename)})


def error_plot(target, prediction, filename, channel=1, upload=False):
    error = np.abs(target - prediction)
    errorpic = np.uint8(255*error[:, :, channel])
    plt.imshow(errorpic, cmap='hot', vmin=0, vmax=255)
    plt.savefig(filename)
    mse2psnr = lambda x: -10. * np.log(x) / np.log(10.)
    if upload:
        mask = np.greater(target, 0)
        fnerr = np.mean((error * mask) ** 2)
        fperr = np.mean((error * np.invert(mask)))
        wandb.log({"error": wandb.Image(filename),
                   "gtp_error": fnerr,
                   "gtp_psnr": mse2psnr(fnerr),
                   "gtn_error": fperr,
                   "gtn_psnr": mse2psnr(fperr)})

