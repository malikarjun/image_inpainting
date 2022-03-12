import matplotlib.pyplot as plt
import numpy as np
import random
from os.path import join
from copy import deepcopy


def create_block_noise(img):
    row = img.shape[0]
    col = img.shape[1]

    known = np.ones((row, col))

    for i in range(int(3 * row/10), int(6 * row/10)):
        for j in range(int(3 * col/10), int(6 * col/10)):
            known[i, j] = 0

    return known


if __name__ == "__main__":
    base_dir = "texture_imgs"
    prob = 0.01
    img = plt.imread(join(base_dir, "grayscale.jpg"))[:75, :75]
    # img = plt.imread(join(base_dir, "grayscale.png"))[:50, :50, 0]
    known = create_block_noise(img)
    np.save(join(base_dir, "mask.npy"), known)
    plt.imsave(join(base_dir, "grayscale.png"), img, cmap="gray")

    img_corr = deepcopy(img)
    img_corr[known == 0] = 0
    plt.imsave(join(base_dir, "grayscale_masked.png"), img_corr, cmap="gray")


