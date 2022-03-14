import matplotlib.pyplot as plt
import numpy as np
import random
from os.path import join
from copy import deepcopy

def create_random_noise(img, corrupt_prob=0.4):
    row = img.shape[0]
    col = img.shape[1]

    known = np.ones((row, col))

    border = 5
    for i in range(border, row-border):
        for j in range(border, col-border):
            if random.uniform(0, 1) < corrupt_prob:
                known[i, j] = 0

    return known


if __name__ == "__main__":
    base_dir = "texture2_random_imgs"
    img = plt.imread(join(base_dir, "grayscale.jpg"))[:75, :75]

    if len(img.shape) > 2:
        img = img[:, :, 0]
    known = create_random_noise(img)
    np.save(join(base_dir, "mask.npy"), known)
    plt.imsave(join(base_dir, "grayscale.png"), img, cmap="gray")

    img_corr = deepcopy(img)
    img_corr[known == 0] = 0
    plt.imsave(join(base_dir, "grayscale_masked.png"), img_corr, cmap="gray")

