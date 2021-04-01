from sklearn.neighbors import NearestNeighbors
import numpy as np
import tensorflow as tf
import store_knn


def find_neighbors(data,
                   labeled_data_index,
                   unlabeled_data_index,
                   num_n,
                   num_r,
                   neigh_file_path = None,
                   sigma = 4.):
    '''
    将Unlabeled依照于Labeled的距离升序排序，并返回近邻点和远离点
    Args:
        data: 所有样本数据
        labeled_data_index: 有标签样本索引
        unlabeled_data_index: 无标签样本索引
        num_n: 距离位于前num_n的点视为近邻点
        num_r: 距离位于后num_r的点视为远离点
        neigh_file_path: 已完成NearestNeighbors计算的邻接文件，为None则重新计算
    Returns:
        neighbor_index: 近邻点在unlabeled中的索引
        remote_index: 远离点在unlabeled中的索引
    '''
    if not neigh_file_path:
        # labeled = data[labeled_data_index]
        # unlabeled = data[unlabeled_data_index]
        A = np.vstack((data[labeled_data_index], data[unlabeled_data_index]))
        nIndex, RBF_matrix = compute_distances_no_loops(A, sigma)
        # nn = NearestNeighbors(n_neighbors=unlabeled.shape[0]).fit(unlabeled)
        # _, nIndex = nn.kneighbors(labeled)
    else:
        nIndex = store_knn.load(neigh_file_path, labeled_data_index)

    neighbor_index = nIndex[:, :num_n]
    remote_index = nIndex[:, -num_r:]
    return neighbor_index, remote_index, RBF_matrix

def compute_distances_no_loops(X, sigma):
    m,n = X.shape
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (m,1))
    dist = H + H.T -2*G #距离矩阵
    RBF_matrix = np.exp(-dist / 2 / sigma ** 2)
    return np.argsort(dist)[:, 1:], RBF_matrix

def dense_to_one_hot(labels_dense, num_classes=2):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[(
        index_offset + labels_dense.ravel()).astype("int")] = 1
    return labels_one_hot


def call_precision(predictions, labels, isCNN = False):
    if not isCNN:
        pre = tf.reshape(predictions, (-1, predictions.shape[2]))
    else:
        pre = predictions
    correct_prediction = tf.equal(
        tf.argmax(pre, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def average_divide(total, num_parts):
    """
    将数字total尽量均匀地分为num_parts份，返回各份的大小
    e.g. total=11, size=3 -> [3, 4, 4]
    """
    if total % num_parts == 0:
        return [total//num_parts] * num_parts
    num = total // num_parts
    res = average_divide(total-num, num_parts-1)
    res.append(num)
    return res

def find_wij(data_1, data_2, sigma = 4.):
    w_knn = np.ndarray((data_1.shape[0], 1))
    for i in range(data_1.shape[0]):
        w_knn[i][0] = np.exp(-np.sum(((data_1[i] - data_2[i]) / sigma) **2) / 2)
        
    return w_knn


def RBF(data_1, data_2, k_of_knn, sigma = 4.):
    print('data_1:', data_1.shape)
    print('data_2:', data_2.shape)
    w_knn = np.ndarray((data_1.shape[0], k_of_knn))
    for i in range(data_1.shape[0]):
        for j in range(k_of_knn):
            w_knn[i][j] = np.exp(-np.sum(((data_1[i] - data_2[i * k_of_knn + j]) / sigma) ** 2, axis=1) / 2)#此行有bug
    return w_knn.reshape(-1, k_of_knn, 1)


class DataSet:
    """
    数据集类，data和label都是只读的，不会在类中修改排序
    """
    def __init__(self, data, one_hot_labels):
        assert data.shape[0] == one_hot_labels.shape[0], (
            "data.shape: %s labels.shape: %s" % (data.shape,
                                                 one_hot_labels.shape))

        self.single_sets = []
        self.num_examples_type = []
        for label in one_hot_labels.T:
            index = np.where(label == 1)[0]
            self.num_examples_type.append(len(index))
            self.single_sets.append(BatchArray(index))

        self._data = data
        self._num_examples = data.shape[0]
        self._data_shape = data.shape[1:]
        self._labels = one_hot_labels
        self._label_types = one_hot_labels.shape[1]
        self._rotate = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def label_types(self):
        return self._label_types

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def data_shape(self):
        return self._data_shape

    def next_batch(self, batch_size, weight=None):
        """
        Return the next `batch_size` examples from this data set.
        将会尽量保证每类数据平均输出
        batch_size: int
        weight: List[int]
        """
        index = []
        if weight is None:
            out_num = average_divide(batch_size, self._label_types)
            for i, single_set in enumerate(self.single_sets):
                # rotation 的目的在于平等对待每一类，防止由于average_divide
                # 结果不均匀，导致某类的数量较少
                rotation = (self._rotate+i) % self._label_types
                index.append(single_set.next_batch(out_num[rotation]))
        else:
            out_num = CircleList(average_divide(batch_size, np.sum(weight)))
            # 将batch_size均分为权重总和份，再按照各类的权重选取对应的份数
            for i, single_set in enumerate(self.single_sets):
                rotation = self._rotate+int(np.sum(weight[:i]))
                divide_batch = out_num[rotation:rotation+weight[i]]
                index.append(single_set.next_batch(divide_batch))
        self._rotate += 1
        if self._rotate > self._label_types:
            self._rotate = 0
        index = np.hstack(index)
        # np.random.shuffle(index)

        return self._data[index], self._labels[index], index

    def random_batch(self, batch_size):
        index = np.random.randint(0, self.num_examples, batch_size)
        return self._data[index], self._labels[index], index


class BatchArray:
    """
    支持batch输出的array
    need_shuffle - 是否在完成一轮输出后打乱array排序
    """
    def __init__(self, array, need_shuffle=True):
        self.array = array
        self._num_examples = array.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.need_shuffle = need_shuffle

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples, (
            """
            batch size过大
            batch size:{}
            数据集大小:{}
            """.format(batch_size, self._num_examples))
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the index
            if self.need_shuffle:
                np.random.shuffle(self.array)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.array[start:end]


class CircleList:
    """
    循环列表，按照索引循环地输出结果。
    e.g. 索引k，输出self._data[k-mn]，其中n为self._num，m为使0 <= k-mn < n的一个整数
    """
    def __init__(self, data):
        self._num = len(data)
        self._data = data

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, end = key.start, key.stop
            assert start <= end
            end = end - (start//self._num)*self._num
            start = start - (start//self._num)*self._num
            if end > self._num:
                return self._data[start:self._num] + self[self._num:end]
            return self._data[start:end]
        else:
            key = key - (key//self._num)*self._num
            return self._data[key]

