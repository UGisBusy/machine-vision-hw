import os
import shutil
import cv2
import numpy as np


# 1.1 Convert the image to grayscale
def convert_to_grayscale(img):
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, g, b = img[i, j]
            gray = 0.3 * r + 0.59 * g + 0.11 * b
            new_img[i, j] = gray
    return new_img


# 1.2 Convert the image to binary using trackbar
def convert_to_binary_trackbar(gray_img):
    cv2.namedWindow("Binary Image")
    cv2.createTrackbar("Threshold", "Binary Image", 0, 255, lambda x: None)

    while True:
        try:
            threshold = cv2.getTrackbarPos("Threshold", "Binary Image")
        except:
            break
        binary_img = convert_to_binary(gray_img, threshold)
        cv2.imshow("Binary Image", binary_img)
        if cv2.waitKey(1) == ord("s"):
            break
    cv2.destroyAllWindows()

    print("Threshold: ", threshold)
    return convert_to_binary(gray_img, threshold)


# 1.2 Convert the image to binary
def convert_to_binary(gray_img, threshold):
    new_img = np.zeros_like(gray_img)
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            if gray_img[i, j] > threshold:
                new_img[i, j] = 255
            else:
                new_img[i, j] = 0
    return new_img


# 2.1 Resize the image without interpolation
def resize_without_interpolation(img, scale):
    new_img = np.zeros((int(img.shape[0] * scale), int(img.shape[1] * scale), 3), dtype=np.uint8)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j] = img[int(i / scale), int(j / scale)]
    return new_img


if __name__ == "__main__":
    # read directories and clean results directory
    cwd = os.getcwd()
    images_dir = os.path.join(cwd, "images")
    results_dir = os.path.join(cwd, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    # read each image in the images directory
    for img_id, filename in enumerate(os.listdir(images_dir)):
        img = cv2.imread(os.path.join(images_dir, filename))

        # 1.1 Convert the image to grayscale
        gray_img = convert_to_grayscale(img)
        cv2.imwrite(os.path.join(results_dir, filename[:-4] + "_q1-1.jpg"), gray_img)

        # 1.2 Convert the image to binary
        binary_img = convert_to_binary(gray_img, (81, 147, 70)[img_id])
        # binary_img = convert_to_binary_trackbar(gray_img)
        cv2.imwrite(os.path.join(results_dir, filename[:-4] + "_q1-2.jpg"), binary_img)

        # 2.1 Resize the image without interpolation
        double_img = resize_without_interpolation(img, 2)
        cv2.imwrite(os.path.join(results_dir, filename[:-4] + "_q2-1-double.jpg"), double_img)
        half_img = resize_without_interpolation(img, 0.5)
        cv2.imwrite(os.path.join(results_dir, filename[:-4] + "_q2-2-half.jpg"), half_img)
