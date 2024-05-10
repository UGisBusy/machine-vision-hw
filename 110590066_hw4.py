import os
import shutil
import cv2
import numpy as np
import heapq
import tqdm


def to_gray(img):
    new_img = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
    return new_img


def get_4_neighbors_index(img, i, j):
    arr = np.array([[i - 1, j], [i, j - 1], [i, j + 1], [i + 1, j]])
    return arr[(arr[:, 0] >= 0) & (arr[:, 0] < img.shape[0]) & (arr[:, 1] >= 0) & (arr[:, 1] < img.shape[1])]


def get_8_neighbors_index(img, i, j):
    arr = np.array([[i - 1, j], [i, j - 1], [i, j + 1], [i + 1, j], [i - 1, j - 1], [i - 1, j + 1], [i + 1, j - 1], [i + 1, j + 1]])
    return arr[(arr[:, 0] >= 0) & (arr[:, 0] < img.shape[0]) & (arr[:, 1] >= 0) & (arr[:, 1] < img.shape[1])]


def get_8_neighbors(img, i, j):
    arr = get_8_neighbors_index(img, i, j)
    return img[arr[:, 0], arr[:, 1]]


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, item, priority):
        heapq.heappush(self.queue, (priority, item))

    def pop(self):
        return heapq.heappop(self.queue)[1]

    def __len__(self):
        return len(self.queue)


def process_lables(img):
    label_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    label_colors = [(0, 0, 0)]
    label_poses = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = list(img[i, j])
            if color != [0, 0, 0]:
                label_poses.append((i, j))
                if color in label_colors:
                    label_img[i, j] = label_colors.index(color)
                else:
                    label_colors.append(color)
                    label_img[i, j] = len(label_colors) - 1
    print(len(label_colors) - 1, "labels found")
    return label_img, label_colors, label_poses


def water_shed_seg(img, label_img, label_poses):
    queue = PriorityQueue()
    result = label_img.copy()
    for ind in range(len(label_poses)):
        i, j = label_poses[ind]
        for ii, jj in get_8_neighbors_index(img, i, j):
            if result[ii, jj] == 0:
                arr = get_8_neighbors(img, ii, jj)
                arr -= np.full_like(arr, img[ii, jj])
                queue.push((ii, jj), np.average(np.abs(arr)))
                result[ii, jj] = -2

    a = result.copy()
    a[a[:, :] == -2] = 255
    cv2.imwrite("tt.png", a)
    progress = tqdm.tqdm(total=img.shape[0] * img.shape[1], desc="Tsunami Coming...")
    while len(queue) > 0:
        progress.update(1)
        i, j = queue.pop()

        neighbors_label = get_8_neighbors(result, i, j)
        if len(np.unique(neighbors_label[neighbors_label[:] > 0])) > 1:
            result[i, j] = -1  # edge
        else:
            result[i, j] = np.max(neighbors_label)

        for ii, jj in get_8_neighbors_index(img, i, j):
            if result[ii, jj] == 0:
                arr = get_8_neighbors(img, ii, jj)
                arr -= np.full_like(arr, img[ii, jj])
                queue.push((ii, jj), np.average(np.abs(arr)))
                result[ii, jj] = -2
    progress.close()
    return result


def coloring(label_img, label_colors, img):
    label_img_flat = label_img.flatten()
    new_img_flat = np.zeros((img.shape[0] * img.shape[1], 3), dtype=np.uint8)
    for ind, color in enumerate(label_colors):
        if ind == 0:
            continue
        new_img_flat[label_img_flat == ind] = color
    new_img = new_img_flat.reshape((img.shape[0], img.shape[1], 3))
    return cv2.addWeighted(img, 0.5, new_img, 0.5, 0)


if __name__ == "__main__":
    cwd = os.getcwd()
    images_dir = os.path.join(cwd, "images")
    results_dir = os.path.join(cwd, "results")
    labels_dir = os.path.join(cwd, "labels")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    # read each image in the images directory
    for img_id, filename in enumerate(os.listdir(images_dir)):
        img = cv2.imread(os.path.join(images_dir, filename))
        gray = to_gray(img)

        label_img = cv2.imread(os.path.join(labels_dir, f"{filename[:-4]}_label.png"))
        label_img, label_colors, label_poses = process_lables(label_img)

        water_img = water_shed_seg(gray, label_img, label_poses)

        color_img = coloring(water_img, label_colors, img)
        cv2.imwrite(os.path.join(results_dir, f"{img_id}_result.jpg"), color_img)
