import os
import shutil
import cv2
import numpy as np


def get_4_neighbors_index(img, i, j):
    arr = np.array([[i - 1, j], [i, j - 1], [i, j + 1], [i + 1, j]])
    return arr[(arr[:, 0] >= 0) & (arr[:, 0] < img.shape[0]) & (arr[:, 1] >= 0) & (arr[:, 1] < img.shape[1])]


def get_8_neighbors_index(img, i, j):
    arr = np.array([[i - 1, j], [i, j - 1], [i, j + 1], [i + 1, j], [i - 1, j - 1], [i - 1, j + 1], [i + 1, j - 1], [i + 1, j + 1]])
    return arr[(arr[:, 0] >= 0) & (arr[:, 0] < img.shape[0]) & (arr[:, 1] >= 0) & (arr[:, 1] < img.shape[1])]


def get_8_neighbors(img, i, j):
    arr = get_8_neighbors_index(img, i, j)
    return img[arr[:, 0], arr[:, 1]]


def mean_filter(img, dim):
    kernel = np.ones((dim, dim)) / (dim * dim)
    pad = np.pad(img, (dim // 2, dim // 2), mode="constant")
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(pad[i : i + dim, j : j + dim] * kernel)
    return result


def median_filter(img, dim):
    pad = np.pad(img, (dim // 2, dim // 2), mode="constant")
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = pad[i : i + dim, j : j + dim]
            window = np.sort(window.flatten())
            result[i, j] = window[len(window) // 2]
    return result


def gaussian_filter(img, dim):
    kernel = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            kernel[i, j] = np.exp(-((i - dim // 2) ** 2 + (j - dim // 2) ** 2) / (2 * (dim // 2) ** 2))
    kernel /= np.sum(kernel)
    pad = np.pad(img, (dim // 2, dim // 2), mode="constant")
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(pad[i : i + dim, j : j + dim] * kernel)
    return result


if __name__ == "__main__":
    cwd = os.getcwd()
    images_dir = os.path.join(cwd, "images")
    results_dir = os.path.join(cwd, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    # read each image in the images directory
    for img_id, filename in enumerate(os.listdir(images_dir)):
        img = cv2.imread(os.path.join(images_dir, filename), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q1_3.jpg"), mean_filter(img, 3))
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q1_7.jpg"), mean_filter(img, 7))
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q2_3.jpg"), median_filter(img, 3))
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q2_7.jpg"), median_filter(img, 7))
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q3.jpg"), gaussian_filter(img, 5))
