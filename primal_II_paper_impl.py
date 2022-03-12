from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from os.path import join

def corrupted(mask):
    return np.any(mask == 0)

def all_corrupted(mask):
    return np.all(mask == 0)

def dist(patch1, patch2, known_val):
    _patch1 = deepcopy(patch1)
    _patch2 = deepcopy(patch2)
    _patch1[known_val == 0] = 0
    _patch2[known_val == 0] = 0

    return np.linalg.norm(_patch1 - _patch2)

def construct_patch_matrix(target_patch, l_known, g_img, g_known):

    rows, cols = g_img.shape
    m = int(np.sqrt(rows))
    n = int(np.sqrt(cols))

    patches = []
    known = []
    patch_idx_dist = []

    for i in range(int(m/2), int(rows - m/2)):
        for j in range(int(n/2), int(cols - n/2)):
            si = int(i - m / 2)
            ei = int(i + m / 2)
            sj = int(j - n / 2)
            ej = int(j + n / 2)

            if corrupted(g_known[si:ei, sj:ej]):
                continue

            euclid_dist = dist(target_patch, g_img[si:ei, sj:ej], g_known[si:ei, sj:ej])

            patches.append(g_img[si:ei, sj:ej].flatten())
            known.append(g_known[si:ei, sj:ej].flatten())
            patch_idx_dist.append([euclid_dist, len(patches) - 1, [i, j]])

    patch_idx_dist = sorted(patch_idx_dist)

    correl_patches = [target_patch.flatten()]
    correl_known = [l_known.flatten()]
    for i in range(10):
        patch_idx = patch_idx_dist[i][1]
        correl_patches.append(patches[patch_idx])
        correl_known.append(known[patch_idx])

    return np.array(correl_patches).T, np.array(correl_known).T

def matrix_completion(mat, known):
    # return completed matrix, the first column will be used to update pixels
    U = cp.Variable(shape=mat.shape)
    constraints = [cp.multiply(known, U) == cp.multiply(known, mat)]

    prob = cp.Problem(cp.Minimize(cp.norm(U, "nuc")), constraints)
    prob.solve()
    print("optimal objective value: {}".format(prob.value))
    return U.value


def reconstruct_image(patch_matrix, img, sr, sc, m, n):
    img_recon = deepcopy(img)
    for i in range(m):
        for j in range(n):
            # TODO: do we really need the max thresholding here?
            img_recon[sr + i, sc + j] = max(0.0, patch_matrix[i * n + j, 0])
    return img_recon


def boundary_completion(img_corr, known, stride=1):

    rows, cols = img_corr.shape
    m = int(np.sqrt(rows))
    n = int(np.sqrt(cols))
    change_known = deepcopy(known)
    img_recon = deepcopy(img_corr)

    for i in range(int(m / 2), int(rows - m / 2), stride):
        for j in range(int(n / 2), int(cols - n / 2), stride):
            si = int(i - m / 2)
            ei = int(i + m / 2)
            sj = int(j - n / 2)
            ej = int(j + n / 2)

            if known[i, j] == 1:
                continue

            patch = img_corr[si:ei, sj:ej]
            _known = known[si:ei, sj:ej]

            if all_corrupted(_known):
                continue

            change_known[si:ei, sj:ej] = 1

            print("i,j = {}, {}".format(i, j))
            patch_matrix, patch_known = construct_patch_matrix(patch, _known, img_corr, known)
            recon_patch_matrix = matrix_completion(patch_matrix, patch_known)
            # we pass img_recon to reconstruct_image because we want to carry forward the work done in previous
            # for loop runs
            img_recon = reconstruct_image(recon_patch_matrix, img_recon, si, sj, m, n)

    return img_recon, change_known

def iterative_diffusion(img_corr, known):
    iter = 1
    img_recon = deepcopy(img_corr)
    while iter < 10:
        print("iter value {}".format(iter))

        img_recon, change_known = boundary_completion(img_corr=img_corr, known=known)
        img_corr = deepcopy(img_recon)

        if iter % 5 == 0:
            known = deepcopy(change_known)
        plt.imsave(join(base_dir, "recon_grayscale_{}.png").format(iter), img_recon, cmap="gray")
        iter += 1

    return img_recon


if __name__ == "__main__":

    base_dir = "texture_imgs"
    img = plt.imread(join(base_dir, "grayscale.png"))[:, :, 0]
    known = np.load(join(base_dir, "mask.npy"))
    img_corr = deepcopy(img)
    img_corr[known == 0] = 0

    img_recon = iterative_diffusion(img_corr=img_corr, known=known)
    plt.imsave(join(base_dir, "recon_grayscale.png"), img_recon, cmap="gray")







