# functions to save or visualize irs to png et al

import matplotlib.pyplot as plt
import numpy as np

def cgrade_ir(ir, filename, channel=1):
    ir = np.uint8(255*ir[:, :, channel])
    plt.imshow(ir, cmap='hot', vmin=0, vmax=255)
    plt.savefig(filename)


