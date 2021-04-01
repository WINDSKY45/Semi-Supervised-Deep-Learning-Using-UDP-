from sklearn.neighbors import NearestNeighbors
import scipy.io as sio
import numpy as np
import cv2 as cv


def store(neighbor_file_path, vecs, core_index, around_index):
    core_num = len(core_index)
    core_vecs = vecs[core_index]
    around_num = len(around_index)
    around_vecs = vecs[around_index]
    nn = NearestNeighbors(n_neighbors=around_num).fit(around_vecs)
    with open(neighbor_file_path, "wb") as file:
        # 文件前两个字节记录around_vecs的数量，也即每行数量
        file.write(np.array([around_num], dtype="uint16").tostring())
        for i in range(core_num):
            _, nIndex = nn.kneighbors(core_vecs[i].reshape(1, -1))
            neighbor_index = around_index[nIndex]
            file.write(neighbor_index.astype("uint16").tostring())
    return nn


def load(neighbor_file_path, core_vecs_index):
    with open(neighbor_file_path, "rb") as file:
        around_num_byte = file.read(2)
        around_num = np.frombuffer(around_num_byte, "uint16")[0]
        neighbor_index = np.zeros((len(core_vecs_index), around_num), "uint16")
        for i, index in enumerate(core_vecs_index):
            offset = index * around_num * 2 + 2
            file.seek(offset)
            data = file.read(around_num * 2)
            neighbor_index[i] = np.frombuffer(data, "uint16")
        return neighbor_index


if __name__ == "__main__":
    neighbor_file_path = "neighbor_index.txt"
    data_dict = sio.loadmat('MNIST')
    samples = data_dict['fea'] / 256
    labels = data_dict['gnd']
    # if not os.path.exists(neighbor_file_path):
    #     store(neighbor_file_path, samples, np.arange(10), np.arange(10, 20))
    neighbor = load(neighbor_file_path, [0])
    cv.imshow("core", samples[0].reshape(28, 28))
    cv.imshow("around", samples[neighbor[0, 0]].reshape(28, 28))
    cv.imshow("away", samples[neighbor[0, -1]].reshape(28, 28))
    cv.waitKey(0)
