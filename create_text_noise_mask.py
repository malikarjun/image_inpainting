import matplotlib.pyplot as plt
import numpy as np
import random
from os.path import join
from copy import deepcopy
import cv2


def create_text_mask(_img):
    img = deepcopy(_img)

    row = img.shape[0]
    col = img.shape[1]
    mask = np.zeros((row, col), np.uint8)

    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(0.03 * col), int(0.5 * row))
    fontScale = 0.5
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(mask, 'Hello World',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                # lineType
                )

    plt.imsave(join("texture2_text_imgs", "mask.png"), mask, cmap="gray")

    numeric_mask = deepcopy(mask)

    numeric_mask[mask > 0] = 0
    numeric_mask[mask == 0] = 1
    return numeric_mask

if __name__ == "__main__":
    base_dir = "realworld_text_imgs"
    img = plt.imread(join(base_dir, "grayscale.jpg"))[:100, :100]
    if len(img.shape) > 2:
        img = img[:, :, 0]
    known = create_text_mask(img)
    np.save(join(base_dir, "mask.npy"), known)

    plt.imsave(join(base_dir, "grayscale.png"), img, cmap="gray")

    img_corr = deepcopy(img)
    img_corr = np.stack([img_corr, img_corr, img_corr], axis=2)
    img_corr[known == 0] = np.array([255, 0, 0])
    plt.imsave(join(base_dir, "grayscale_masked.png"), img_corr)


