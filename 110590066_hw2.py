import os
import shutil
import cv2
import numpy as np


def convert_to_binary_trackbar(gray_img):
    cv2.namedWindow("Binary Image")
    cv2.createTrackbar("Threshold", "Binary Image", 0, 255, lambda x: None)

    while True:
        try:
            threshold = cv2.getTrackbarPos("Threshold", "Binary Image")
        except:
            break
        binary_img = to_binary(gray_img, threshold)
        cv2.imshow("Binary Image", binary_img)
        if cv2.waitKey(1) == ord("s"):
            break
    cv2.destroyAllWindows()

    print("Threshold: ", threshold)
    return to_binary(gray_img, threshold)


def to_gray(img):
    new_img = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    return new_img


def to_binary(img, threshold):
    new_img = np.zeros_like(img)
    new_img[img > threshold] = 255
    new_img[img <= threshold] = 0
    return new_img


def to_label_rec(img, n):
    bound_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    bound_img[1:-1, 1:-1] = img
    label = 1
    for i in range(1, bound_img.shape[0] - 1):
        for j in range(1, bound_img.shape[1] - 1):
            if bound_img[i, j] == 0:
                bound_img = fill(bound_img, i, j, label, n)
                label += 1
    hsv_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    hsv_img[:, :, 0] = (bound_img[1:-1, 1:-1] * 180 / label).astype(np.uint8)
    hsv_img[:, :, 1] = 255
    hsv_img[:, :, 2] = 255 * (bound_img[1:-1, 1:-1] > 0).astype(np.uint8)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


def fill(bound_img, i, j, label, n):
    if bound_img[i, j] == 0:
        bound_img[i, j] = label
        bound_img = fill(bound_img, i - 1, j, label, n)
        bound_img = fill(bound_img, i + 1, j, label, n)
        bound_img = fill(bound_img, i, j - 1, label, n)
        bound_img = fill(bound_img, i, j + 1, label, n)
        if n == 8:
            bound_img = fill(bound_img, i - 1, j - 1, label, n)
            bound_img = fill(bound_img, i - 1, j + 1, label, n)
            bound_img = fill(bound_img, i + 1, j - 1, label, n)
            bound_img = fill(bound_img, i + 1, j + 1, label, n)
    return bound_img


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
        binary_img = convert_to_binary_trackbar(to_gray(img))
        label_img = to_label_rec(binary_img, 4)
        cv2.imwrite(os.path.join(results_dir, filename[:-4] + "_q1.jpg"), label_img)
