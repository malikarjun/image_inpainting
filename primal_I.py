# total variation minimization
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from os.path import join

def reconstruct_image(img_corr, known):
    U = cp.Variable(shape=img_corr.shape)
    obj = cp.Minimize(cp.tv(U))
    constraints = [cp.multiply(known, U) == cp.multiply(known, img_corr)]
    prob = cp.Problem(obj, constraints)
    prob.solve( solver=cp.SCS)
    return U.value


if __name__ == "__main__":

    base_dirs = ["realworld_random_imgs", "realworld_text_imgs", "texture2_block_imgs", "texture2_random_imgs",
                 "texture2_text_imgs", "texture_block_imgs", "texture_random_imgs", "texture_text_imgs"]

    for base_dir in base_dirs:
        img = plt.imread(join(base_dir, "grayscale.png"))[:, :, 0]
        known = np.load(join(base_dir, "mask.npy"))
        img_corr = deepcopy(img)
        img_corr[known == 0] = 0

        img_recon = reconstruct_image(img_corr, known)

        plt.imsave(join(base_dir, "recon_grayscale_tv.png"), img_recon, cmap="gray")



