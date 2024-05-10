import os
import shutil
import cv2
import numpy as np
import math


def get_4_neighbors(img, i, j):
    return np.array(
        [
            img[i - 1, j],
            img[i, j - 1],
            img[i, j + 1],
            img[i + 1, j],
        ]
    )


def get_8_neighbors(img, i, j):
    return np.array(
        [
            img[i - 1, j - 1],
            img[i - 1, j],
            img[i - 1, j + 1],
            img[i, j - 1],
            img[i, j + 1],
            img[i + 1, j - 1],
            img[i + 1, j],
            img[i + 1, j + 1],
        ]
    )


def to_gray(img):
    new_img = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    return new_img


def to_binary(img, inv=False):
    threshold = img.mean()
    d = 1
    while d > 0.001:
        new_threshold = (img[img > threshold].mean() + img[img <= threshold].mean()) / 2
        d = abs(new_threshold - threshold)
        threshold = new_threshold
    new_img = np.zeros_like(img)
    new_img[img > threshold] = 1 if inv else 0
    new_img[img <= threshold] = 0 if inv else 1
    return new_img


def to_distance(img):
    is_changed = True
    new_img = img.copy()
    while is_changed:
        is_changed = False
        temp_img = np.zeros_like(new_img)
        for i in range(1, new_img.shape[0] - 1):
            for j in range(1, new_img.shape[1] - 1):
                if new_img[i, j] != 0:
                    # temp_img[i, j] = 1 + np.min(get_8_neighbors(new_img, i, j))
                    temp_img[i, j] = 1 + np.min(get_4_neighbors(new_img, i, j))
                    if temp_img[i, j] != new_img[i, j]:
                        is_changed = True
        new_img = temp_img.copy()
    return new_img


def to_skeleton(img):
    result = img.copy()
    n = 1
    base_kernels = [
        np.array([[-1, 1, -1], [0, 1, 0], [-1, 1, -1]]),
        np.array([[1, 0, 1], [0, 1, -1], [-1, -1, -1]]),
        np.array([[1, 0, -1], [0, 1, 1], [-1, -1, -1]]),
        np.array([[1, 0, -1], [0, 1, -1], [-1, -1, 1]]),
        np.array([[-1, -1, -1], [0, 1, 1], [1, 0, -1]]),
    ]

    kernels = []
    for bk in base_kernels:
        kernels.append(bk)
        kernels.append(np.rot90(bk, 1))
        kernels.append(np.rot90(bk, 2))
        kernels.append(np.rot90(bk, 3))

    while n <= result.max():
        for i in range(1, result.shape[0] - 1):
            for j in range(1, result.shape[1] - 1):
                if result[i, j] == n:
                    is_remove = True
                    neighbors = get_8_neighbors(result, i, j)
                    if result[i, j] < np.max(neighbors):
                        # check connectivity
                        sample = result[i - 1 : i + 2, j - 1 : j + 2].copy()
                        a = sample.copy()
                        sample[sample > 1] = 1
                        for k in kernels:
                            if match(sample, k):
                                is_remove = False
                                break
                    if result[i, j] >= np.max(neighbors):
                        is_remove = False
                    if is_remove:
                        result[i, j] = 0
        n += 1
    result[result > 1] = 1
    return result


def thinning(img):
    result = img.copy()
    base_kernels = [
        np.array([[1, 1, 0], [1, 1, 0], [-1, -1, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
        np.array([[0, 0, -1], [0, 1, 1], [-1, 1, -1]]),
        np.array([[-1, 1, -1], [1, 1, 1], [-1, -1, -1]]),
    ]

    kernels = []
    for bk in base_kernels:
        kernels.append(bk)
        kernels.append(np.rot90(bk, 1))
        kernels.append(np.rot90(bk, 2))
        kernels.append(np.rot90(bk, 3))

    is_changed = True
    while is_changed:
        is_changed = False
        for k in kernels:
            for i in range(1, result.shape[0] - 1):
                for j in range(1, result.shape[1] - 1):
                    if match(result[i - 1 : i + 2, j - 1 : j + 2], k):
                        result[i, j] = 0
                        is_changed = True
    return result


def match(img, kernel):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if kernel[i, j] != -1 and kernel[i, j] != img[i, j]:
                return False
    return True


if __name__ == "__main__":
    # clean
    cwd = os.getcwd()
    images_dir = os.path.join(cwd, "images")
    results_dir = os.path.join(cwd, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    for img_id, filename in enumerate(os.listdir(images_dir)):
        img = cv2.imread(os.path.join(images_dir, filename))
        binary_img = to_binary(to_gray(img))
        # cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q1-0-bin.jpg"), binary_img * 255)
        distance_img = to_distance(binary_img)
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q1-1.jpg"), distance_img * 255 // np.max(distance_img))
        skeleton_img = to_skeleton(distance_img)
        # cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q1-2-skel.jpg"), skeleton_img * 255 // np.max(skeleton_img))
        skeleton_img = thinning(skeleton_img)
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_q1-2.jpg"), skeleton_img * 255 // np.max(skeleton_img))
