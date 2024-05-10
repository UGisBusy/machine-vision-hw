import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_gray(img):
    new_img = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    return new_img


if __name__ == '__main__':
    cwd = os.getcwd()
    images_dir = os.path.join(cwd, "images")
    results_dir = os.path.join(cwd, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    
    # read each image in the images directory
    for img_id, filename in enumerate(os.listdir(images_dir)):
        img = cv2.imread(os.path.join(images_dir, filename))
        gray = to_gray(img)
        
        # bar chart of the histogram
        plt.hist(gray.ravel(), 256, [0, 256])
        plt.savefig(os.path.join(results_dir, f"{img_id}_hist.png"))
        plt.close()
    
        
        
        
        
        
        