#!/usr/bin/python3

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics.pairwise import euclidean_distances
from pyemd import emd


def main():
    rv = np.load('out.npz')

    rv = rv['arr_0']
    plt.imshow(rv)
    plt.colorbar()
    plt.show()


main()

