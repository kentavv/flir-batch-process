#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def main():
    rv = np.load('emd_results.npz')

    rv = rv['arr_0']
    plt.imshow(rv)
    plt.colorbar()
    plt.show()


main()
