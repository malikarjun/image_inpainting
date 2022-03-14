import matplotlib.pyplot as plt
import numpy as np
import random
from os.path import join
from copy import deepcopy
import cv2


def create_line_masks(_img):
    img = deepcopy(_img)
    x, y, _ = img.shape
    # Prepare masking matrix
    # White background
    mask = np.full(img.shape, 1, np.uint8)
    for _ in range(np.random.randint(1, 10)):

        # Get random x locations to start line
        x1, x2 = np.random.randint(1, x), np.random.randint(1, x)

        # Get random y locations to start line
        y1, y2 = np.random.randint(1, y), np.random.randint(1, y)

        # Get random thickness of the line drawn
        thickness = np.random.randint(1, 10)

        # Draw black line on the white mask
        cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness)

    img[mask == 0] = 0

    numeric_mask = np.ones((img.shape[0], img.shape[1]))
    numeric_mask[mask[:, :, 0] == 0] = 0

    return numeric_mask


if __name__ == "__main__":
    base_dir = "realworld_linemask_imgs"
    img = plt.imread(join(base_dir, "grayscale.jpg"))
    known = create_line_masks(img)
    np.save(join(base_dir, "mask.npy"), known)
    plt.imsave(join(base_dir, "grayscale.png"), img, cmap="gray")

    img_corr = deepcopy(img)
    img_corr[known == 0] = 0
    plt.imsave(join(base_dir, "grayscale_masked.png"), img_corr, cmap="gray")


