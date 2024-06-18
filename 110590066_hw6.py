import os
import shutil
import cv2
import numpy as np


def rgb2gray(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


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


def apply_sobel(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    pad = np.pad(img, 1, mode="constant")
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dx[i, j] = np.sum(pad[i : i + 3, j : j + 3] * kernel)
            dy[i, j] = np.sum(pad[i : i + 3, j : j + 3] * kernel.T)
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))
    return magnitude / np.max(magnitude), angle


def find_close_dir(grad_dir):
    closest_dir = np.zeros(grad_dir.shape)
    for i in range(1, grad_dir.shape[0] - 1):
        for j in range(1, grad_dir.shape[1] - 1):
            if (grad_dir[i, j] > -22.5 and grad_dir[i, j] <= 22.5) or (grad_dir[i, j] <= -157.5 and grad_dir[i, j] > 157.5):
                closest_dir[i, j] = 0
            elif (grad_dir[i, j] > 22.5 and grad_dir[i, j] <= 67.5) or (grad_dir[i, j] <= -112.5 and grad_dir[i, j] > -157.5):
                closest_dir[i, j] = 45
            elif (grad_dir[i, j] > 67.5 and grad_dir[i, j] <= 112.5) or (grad_dir[i, j] <= -67.5 and grad_dir[i, j] > -112.5):
                closest_dir[i, j] = 90
            else:
                closest_dir[i, j] = 135
    return closest_dir


def non_maximal_suppressor(grad_mag, closest_dir):
    thinned_output = np.zeros(grad_mag.shape)
    for i in range(1, grad_mag.shape[0] - 1):
        for j in range(1, grad_mag.shape[1] - 1):
            if closest_dir[i, j] == 0:
                if (grad_mag[i, j] > grad_mag[i, j + 1]) and (grad_mag[i, j] > grad_mag[i, j - 1]):
                    thinned_output[i, j] = grad_mag[i, j]
            elif closest_dir[i, j] == 45:
                if (grad_mag[i, j] > grad_mag[i + 1, j + 1]) and (grad_mag[i, j] > grad_mag[i - 1, j - 1]):
                    thinned_output[i, j] = grad_mag[i, j]
            elif closest_dir[i, j] == 90:
                if (grad_mag[i, j] > grad_mag[i + 1, j]) and (grad_mag[i, j] > grad_mag[i - 1, j]):
                    thinned_output[i, j] = grad_mag[i, j]
            else:
                if (grad_mag[i, j] > grad_mag[i + 1, j - 1]) and (grad_mag[i, j] > grad_mag[i - 1, j + 1]):
                    thinned_output[i, j] = grad_mag[i, j]
    return thinned_output / np.max(thinned_output)


def DFS(img):
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == 1:
                t_max = max(img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1], img[i, j - 1], img[i, j + 1], img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1])
                if t_max == 2:
                    img[i, j] = 2


def edge_tracking(img):
    result = np.copy(img)
    total_strong = np.sum(result == 2)
    while 1:
        DFS(result)
        if total_strong == np.sum(result == 2):
            break
        total_strong = np.sum(result == 2)
    for i in range(1, int(result.shape[0] - 1)):
        for j in range(1, int(result.shape[1] - 1)):
            if result[i, j] == 1:
                result[i, j] = 0
    result = result / np.max(result)
    return result


def double_threshold(img):
    result = np.zeros_like(img)
    low_ratio = 0.1
    high_ratio = 0.2
    low = np.min(img) + low_ratio * (np.max(img) - np.min(img))
    high = np.min(img) + high_ratio * (np.max(img) - np.min(img))

    result[img >= high] = 2
    img[img >= high] = 0
    result[img >= low] = 1
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
        img = cv2.imread(os.path.join(images_dir, filename), cv2.IMREAD_COLOR)
        gray = rgb2gray(img)
        blur = gaussian_filter(gray, 5)

        G, theta = apply_sobel(blur)
        # cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_g.jpg"), G * 255)
        # cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_th.jpg"), theta)

        closest_dir = find_close_dir(theta)
        sup = non_maximal_suppressor(G, closest_dir)
        # cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_sup.jpg"), sup * 255)

        thr = double_threshold(sup)
        edge = edge_tracking(thr)
        cv2.imwrite(os.path.join(results_dir, f"{filename[:-4]}_edge.jpg"), edge * 255)
