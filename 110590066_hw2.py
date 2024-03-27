import os
import shutil
import cv2
import numpy as np


def to_gray(img):
    new_img = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    return new_img


def to_binary(img, threshold):
    new_img = np.zeros_like(img)
    new_img[img > threshold] = 1
    new_img[img <= threshold] = 0
    return new_img


def to_label_seq(img, n):
    bound_img = np.ones((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint16)
    bound_img[1:-1, 1:-1] = img
    label = 2
    labels = set()
    for i in range(1, bound_img.shape[0] - 1):
        for j in range(1, bound_img.shape[1] - 1):
            neighbor = np.ones(4, dtype=np.uint16)
            if bound_img[i, j] == 0:
                neighbor[:2] = [bound_img[i, j - 1], bound_img[i - 1, j]]
                if n == 8:
                    neighbor[2:] = [bound_img[i - 1, j - 1], bound_img[i - 1, j + 1]]
                # only one of neighbor has label
                if len(neighbor[neighbor != 1]) == 1:
                    bound_img[i, j] = neighbor[neighbor != 1][0]
                # some of neighbors have label
                elif len(neighbor[neighbor != 1]) > 0:
                    bound_img[i, j] = neighbor[neighbor != 1].min()
                    for k in neighbor[
                        np.logical_and(neighbor != 1, neighbor != bound_img[i, j])
                    ]:
                        bound_img[bound_img == k] = bound_img[i, j]
                        labels.discard(k)
                # all neighbors are background
                else:
                    bound_img[i, j] = label
                    labels.add(label)
                    label += 1
    bound_img[bound_img == 1] = 0
    color_lst = np.random.randint(50, 200, (len(labels), 3))
    color_img = np.zeros((bound_img.shape[0], bound_img.shape[1], 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        color_img[bound_img == label] = color_lst[i]
    return color_img[1:-1, 1:-1]


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
        binary_img = to_binary(to_gray(img), [136, 179, 243][img_id])
        label_img = to_label_seq(binary_img, 4)
        cv2.imwrite(os.path.join(results_dir, filename[:-4] + "_q1-4.jpg"), label_img)
        label_img = to_label_seq(binary_img, 8)
        cv2.imwrite(os.path.join(results_dir, filename[:-4] + "_q1-8.jpg"), label_img)
