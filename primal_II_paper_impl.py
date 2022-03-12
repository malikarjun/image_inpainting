from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from os.path import join


def apply_line_masks(_img):
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

    img[mask == 0] = 1

    numeric_mask = np.ones((img.shape[0], img.shape[1]))
    numeric_mask[mask[:, :, 0] == 0] = 0

    return img, numeric_mask
# mask = np.load("data/inpaint/line_mask/train_mask/ILSVRC2012_val_00000537.npy")
# img_orig = plt.imread("data/imagenette2-160/train/n03394916/n03394916_29106.JPEG")
# img = plt.imread("data/imagenette2-160/train/n03394916/n03394916_1149.JPEG")



def dist(patch1, patch2, known_val):
    _patch1 = deepcopy(patch1)
    _patch2 = deepcopy(patch2)
    _patch1[known_val == 0] = 0
    _patch2[known_val == 0] = 0

    return np.linalg.norm(_patch1 - _patch2)


def construct_patch_matrix(target_patch, l_known, g_img, g_known, m, n, rows, cols):
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
    # prob.solve(verbose=True)
    prob.solve()
    print("optimal objective value: {}".format(prob.value))
    return U.value


def corrupted(patch):
    return np.any(patch == 0)


def all_corrupted(patch):
    return np.all(patch == 0)

def reconstruct_image(patch_matrix, img, sr, sc, m, n):
    for i in range(m):
        for j in range(n):
            img[sr + i, sc + j] = max(0.0, patch_matrix[i * n + j, 0])
            # img[sr + i, sc + j] = patch_matrix[i * n + j, 0]

if __name__ == "__main__":

    base_dir = "texture_imgs"
    img = plt.imread(join(base_dir, "grayscale.png"))[:, :, 0]
    # img_corr, known = apply_line_masks(img)

    known = np.load(join(base_dir, "mask.npy"))
    img_corr = deepcopy(img)
    img_corr[known == 0] = 0

    img_recon = deepcopy(img_corr)

    _rows, _cols = img.shape

    _m = int(np.sqrt(_rows))
    _n = int(np.sqrt(_cols))

    # _m = 8
    # _n = 8

    l = 1
    while l < 10:
        print("l value {}".format(l))
        # iterate over the matrix in patches and invoke methods if missing data is found

        change_known = deepcopy(known)

        for i in range(int(_m/2), int(_rows - _m/2)):
            for j in range( int(_n/2), int(_cols - _n/2)):
                # TODO: instead of continuing use min(i + _m, _rows) to handle boundary conditions
                si = int(i - _m/2)
                ei = int(i + _m/2)
                sj = int(j - _n/2)
                ej = int(j + _n/2)

                # if i == 27 and j == 27:
                #     pass
                #     print("here")

                if known[i][j] == 1:
                    continue

                patch = img_corr[si:ei, sj:ej]
                _known = known[si:ei, sj:ej]

                if all_corrupted(_known):
                    continue

                change_known[si:ei, sj:ej] = 1

                print("i,j = {}, {}".format(i, j))
                patch_matrix, patch_known = construct_patch_matrix(patch, _known, img_corr, known, _m, _n, _rows, _cols)
                recon_patch_matrix = matrix_completion(patch_matrix, patch_known)

                # recon_patch_matrix[:, 0] = patch_matrix[:, 1]

                reconstruct_image(recon_patch_matrix, img_recon, si, sj, _m, _n)
                # img_corr = deepcopy(img_recon)
                # known[si:ei, sj:ej] = 1

        img_corr = deepcopy(img_recon)
        # known = deepcopy(change_known)
        if l % 5 == 0:
            known = deepcopy(change_known)
        plt.imsave(join(base_dir, "recon_grayscale_{}.png").format(l), img_recon, cmap="gray")
        l += 1

    plt.imsave(join(base_dir, "recon_grayscale.png"), img_recon, cmap="gray")







