import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import makedirs
from os.path import exists, isdir, join, basename
import tarfile
import requests
import shutil
from glob import glob
from copy import deepcopy


def download_imagenette(base_path="data"):

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    zip_file_path = join(base_path, "imagenette2-160.tgz")

    if not exists(zip_file_path):
        print("downloading imagenette...")
        makedirs(base_path, exist_ok=True)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_file_path, 'wb') as f:
                f.write(response.raw.read())
    else:
        print("already downloaded...")

    if not isdir(zip_file_path[:-4]):
        print("extracting...")
        tar = tarfile.open(zip_file_path, "r:gz")
        tar.extractall(base_path)
        tar.close()
    else:
        print("already extracted...")

download_imagenette()


def process_imagenette(base_path="data"):
    dataset_path = join(base_path, "imagenette2-160")
    final_path = join(base_path, "inpaint", "original")

    if isdir(final_path):
        print("already processed...")
        return

    print("processing imagenette files...")

    for folder in glob(join(dataset_path, "*")):
        if not isdir(folder):
            continue
        makedirs(join(final_path, basename(folder)), exist_ok=True)
        dst = join(final_path, basename(folder))

        for subfolder in glob(join(folder, "*")):
            for file in glob(join(subfolder, "*")):
                img = plt.imread(file)
                if len(img.shape) == 2:
                    continue
                shutil.copyfile(file, join(dst, basename(file)))

process_imagenette()

def apply_line_masks(_img):
    img = deepcopy(_img)
    x, y, _ = img.shape
    # Prepare masking matrix
    # White background
    mask = np.full(img.shape, 255, np.uint8)
    for _ in range(np.random.randint(1, 10)):

        # Get random x locations to start line
        x1, x2 = np.random.randint(1, x), np.random.randint(1, x)

        # Get random y locations to start line
        y1, y2 = np.random.randint(1, y), np.random.randint(1, y)

        # Get random thickness of the line drawn
        thickness = np.random.randint(1, 10)

        # Draw black line on the white mask
        cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness)

    img[mask == 0] = 255
    return img


def mask_images(base_path="data", mask_dir="line_mask"):
    orig_path = join(base_path, "inpaint", "original")
    dst_path = join(base_path, "inpaint", mask_dir)

    if isdir(dst_path):
        print("{} already created...".format(mask_dir))
        return

    print("masking images using {}...".format(mask_dir))

    for folder in glob(join(orig_path, "*")):
        makedirs(join(dst_path, basename(folder)), exist_ok=True)
        for file in glob(join(folder, "*")):
            img = plt.imread(file)
            masked_img = apply_line_masks(img)
            plt.imsave(join(dst_path, basename(folder), basename(file)), masked_img)

mask_images()
