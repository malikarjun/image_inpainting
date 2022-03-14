from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from os.path import join
from math import floor, ceil
from scipy.ndimage.filters import gaussian_filter


def none_corrupted(mask):
    return np.all(mask == 1)

def corrupted(mask):
    return np.any(mask == 0)

def mostly_corrupted(mask):
    return np.sum(mask)/(mask.shape[0] * mask.shape[1]) < 0.6

def all_corrupted(mask):
    return np.all(mask == 0)

def get_m_n(img, stage="structure", patch_size_mul=1):
    rows, cols = img.shape
    if stage == "structure":
        m = min(5, int(np.sqrt(rows)))
        n = min(5, int(np.sqrt(cols)))
    else:
        m = min(10, int(np.sqrt(rows)))
        n = min(10, int(np.sqrt(cols)))
    return m, n

def get_sij_eij(i, j, m, n):
    si = i - floor(m / 2)
    ei = i + ceil(m / 2)
    sj = j - floor(n / 2)
    ej = j + ceil(n/2)
    return si, ei, sj, ej

def dist(patch1, patch2, known_val):
    _patch1 = deepcopy(patch1)
    _patch2 = deepcopy(patch2)
    _patch1[known_val == 0] = 0
    _patch2[known_val == 0] = 0

    return np.linalg.norm(_patch1 - _patch2)

def construct_patch_matrix(target_patch, l_known, g_img, g_known, stride=1, stage="structure", patch_size_mul=1):

    rows, cols = g_img.shape
    m, n = get_m_n(g_img, stage=stage, patch_size_mul=patch_size_mul)

    patches = []
    known = []
    patch_idx_dist = []

    for i in range(int(m/2), int(rows - m/2), stride):
        for j in range(int(n/2), int(cols - n/2), stride):

            si, ei, sj, ej = get_sij_eij(i, j, m, n)

            if "random" in base_dir:
                if mostly_corrupted(g_known[si:ei, sj:ej]):
                    continue
            else:
                if corrupted(g_known[si:ei, sj:ej]):
                    continue

            euclid_dist = dist(target_patch, g_img[si:ei, sj:ej], g_known[si:ei, sj:ej])

            patches.append(g_img[si:ei, sj:ej].flatten())
            known.append(g_known[si:ei, sj:ej].flatten())
            patch_idx_dist.append([euclid_dist, len(patches) - 1, [i, j]])

    patch_idx_dist = sorted(patch_idx_dist)


    if "random" in base_dir:
        correl_patches = []
        correl_known = []
        if patch_idx_dist[0][0] != 0:
            correl_patches.append(target_patch.flatten())
            correl_known.append(l_known.flatten())
    else:
        correl_patches = [target_patch.flatten()]
        correl_known = [l_known.flatten()]

    for i in range(min(10, int(len(patch_idx_dist)/2))):
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
    # print("optimal objective value: {}".format(prob.value))
    return U.value


def reconstruct_image(patch_matrix, img, sr, sc, m, n):
    img_recon = deepcopy(img)
    for i in range(m):
        for j in range(n):
            # TODO: do we really need the max thresholding here?
            img_recon[sr + i, sc + j] = max(0.0, patch_matrix[i * n + j, 0])
    return img_recon


def boundary_completion(img_corr, known, stride=1, stage="structure", patch_size_mul=1):

    rows, cols = img_corr.shape
    m, n = get_m_n(img_corr, stage=stage, patch_size_mul=patch_size_mul)

    change_known = deepcopy(known)
    img_recon = deepcopy(img_corr)

    for i in range(int(m / 2), int(rows - m / 2), stride):
        for j in range(int(n / 2), int(cols - n / 2), stride):

            si, ei, sj, ej = get_sij_eij(i, j, m, n)

            # if known[i, j] == 1:
            #     continue

            patch = img_corr[si:ei, sj:ej]
            _known = known[si:ei, sj:ej]

            if all_corrupted(_known) or none_corrupted(_known):
            # if all_corrupted(_known):
                continue

            change_known[si:ei, sj:ej] = 1

            print("i,j = {}, {}".format(i, j))
            patch_matrix, patch_known = construct_patch_matrix(patch, _known, img_corr, known, stride=stride,
                                                               stage=stage, patch_size_mul=patch_size_mul)
            recon_patch_matrix = matrix_completion(patch_matrix, patch_known)
            # we pass img_recon to reconstruct_image because we want to carry forward the work done in previous
            # for loop runs
            img_recon = reconstruct_image(recon_patch_matrix, img_recon, si, sj, m, n)

    return img_recon, change_known

def update_known(known):
    lknown = deepcopy(known)
    lknown = gaussian_filter(lknown, sigma=0.25)
    lknown[lknown > 0] = 1
    return lknown

def iterative_diffusion(img_corr, known, stride=1, stage="structure"):
    iter = 1
    interval = 10
    img_recon = deepcopy(img_corr)
    while iter < 20:
        # print("iter value {}".format(iter))

        img_recon, change_known = boundary_completion(img_corr=img_corr, known=known, stride=stride, stage=stage)

        if iter % interval == 0:
            # plt.imsave("tmp1.png", known)
            # known = update_known(known)
            # plt.imsave("tmp2.png", known)
            known = deepcopy(change_known)

        img_corr = deepcopy(img_recon)
        plt.imsave(join(base_dir, "recon_grayscale_{}_{}.png".format(stage, iter)), img_recon, cmap="gray")

        iter += 1

    return img_recon


if __name__ == "__main__":

    base_dir = "texture_random_imgs"
    img = plt.imread(join(base_dir, "grayscale.png"))[:, :, 0]
    known = np.load(join(base_dir, "mask.npy"))
    img_corr = deepcopy(img)
    img_corr[known == 0] = 0

    img_recon = iterative_diffusion(img_corr=img_corr, known=known, stride=5)
    plt.imsave(join(base_dir, "recon_grayscale.png"), img_recon, cmap="gray")

    # img_recon = plt.imread(join(base_dir, "recon_grayscale.png"))[:, :, 0]
    # img_blurred = gaussian_filter(img_recon, sigma=1)
    #
    #
    # plt.imsave(join(base_dir, "blurred_grayscale.png"), img_blurred, cmap="gray")
    #
    # _img_blurred = deepcopy(img_blurred)
    # _img_blurred[known == 0] = 0
    # img_res_texture = img_corr - _img_blurred
    # img_res_texture[img_res_texture <= 0] = 0
    #
    # plt.imsave(join(base_dir, "res_texture_grayscale.png"), img_res_texture, cmap="gray")
    #
    # img_texture = iterative_diffusion(img_corr=img_res_texture, known=known, stride=1, stage="texture")
    # plt.imsave(join(base_dir, "texture_grayscale.png"), img_texture, cmap="gray")
