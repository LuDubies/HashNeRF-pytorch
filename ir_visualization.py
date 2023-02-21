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
    error = np.uint8(255*error[:, :, channel])
    plt.imshow(error, cmap='hot', vmin=0, vmax=255)
    plt.savefig(filename)
    if upload:
        wandb.log({"error": wandb.Image(filename)})

