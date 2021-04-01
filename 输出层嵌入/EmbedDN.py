import tensorflow as tf
import numpy as np
# from tqdm import tqdm
import scipy.io as sio
from network_and_loss import forward, graph_loss, supervised_loss
from utils import (find_neighbors, call_precision, dense_to_one_hot,
                   DataSet, RBF)
from findIndex import equal_proportion_sampling

labeled_data_size = 100
num_features = 784
num_classes = 10
batch_size = 10
lambda_f = 0.001
k_of_knn = 10
m_of_knn = 50
SEED = 666
np.random.seed(SEED)

data_dict = sio.loadmat('MNIST')
# normalization
samples = data_dict['fea'] / 256
labels = dense_to_one_hot(data_dict['gnd'], num_classes)

# 有标记数据
# l_index = np.random.randint(5000, size=labeled_data_size)
l_index = equal_proportion_sampling(data_dict['gnd'][:5000], num_classes, labeled_data_size)
l_dataset = DataSet(samples[l_index], labels[l_index])

# 无标记数据
# u_index = np.arange(5000, 60000)
u_index = np.random.randint(5000, 60000, size = 2000)
uindex_dataset = DataSet(u_index, labels[u_index])

sample_data = np.vstack((samples[l_index], samples[u_index]))
sample_label = np.vstack((labels[l_index], labels[u_index]))

# 测试数据
test_dataset = DataSet(samples[60000:], labels[60000:])

neighborIndex, remoteIndex, RBF_matrix = find_neighbors(
    samples,
    l_index,
    u_index,
    num_n = k_of_knn,
    num_r = m_of_knn,
    neigh_file_path = None,
    # sigma = 3.5
    # neigh_file_path='./neigh_core5k_around_5k-60k.txt'
)

input_data_shape = (None, 1, num_features)
input_neighbor_shape = (None, k_of_knn, num_features)
input_remote_shape = (None, m_of_knn, num_features)
label_shape = (None, num_classes)
wij_nshape = (None, k_of_knn, 1)
wij_rshape = (None, m_of_knn, 1)
xl = tf.placeholder(tf.float32, input_data_shape, name="labeled_data_input")
xn = tf.placeholder(tf.float32, input_neighbor_shape, name="neighbor_data_input")
xr = tf.placeholder(tf.float32, input_remote_shape, name="remote_data_input")

wij_n = tf.placeholder(tf.float32, wij_nshape, name="RBF_weights_neighbor")
wij_r = tf.placeholder(tf.float32, wij_rshape, name='RBF_weights_remote')
# yl = forward('CNN', xl)
# yn = forward('CNN', xn)
# yl_embed = forward('DNN_2_embed', xl)
yl = forward('DNN_1', xl)
yn = forward('DNN_1', xn)
yr = forward('DNN_1', xr)
yt = tf.placeholder(tf.float32, label_shape, name="labeled_data_label")

with tf.name_scope("loss"):
    loss1 = supervised_loss('Cross_Entropy')(yl, yt)
    loss1 += graph_loss('UDP with Kernel Weighting')(yl, yn, yr, wij_n, wij_r)* lambda_f
    loss_sum_1 = tf.summary.scalar('loss1', loss1[0][0])
    # loss += graph_loss('LE')(yl, yn, 1.0)
    # loss += graph_loss('LE')(yl, yr, 0.0)
    loss2 = graph_loss('UDP with Kernel Weighting')(yl, yn, yr, wij_n, wij_r) * lambda_f
    loss_sum_2 = tf.summary.scalar('loss2', loss2[0][0])

lr = 0.0001

# opt = tf.train.AdagradOptimizer(lr)
opt = tf.train.AdamOptimizer(lr)
# opt = tf.train.MomentumOptimizer(lr, momentum = 0.5)

gvs1 = opt.compute_gradients(loss1)
clipped_gvs1 = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs1]
train1 = opt.apply_gradients(clipped_gvs1)

gvs2 = opt.compute_gradients(loss2)
# print("gvs2", gvs2)
clipped_gvs2 = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs2 if grad is not None]
train2 = opt.apply_gradients(clipped_gvs2)

with tf.name_scope("accuracy"):
    test_acc = call_precision(yl, yt)
    acc_sum_1 = tf.summary.scalar('test_acc', test_acc)

    train_acc = call_precision(yl, yt)
    acc_sum_2 = tf.summary.scalar('training_acc', train_acc)

sess = tf.InteractiveSession()
tf.random.set_random_seed(SEED)
tf.global_variables_initializer().run()
writer = tf.summary.FileWriter('logs', sess.graph)
no_epoch = 100000
for i in range(no_epoch):
    # breakpoint()
    xl_batch, yt_batch, xl_batch_index = l_dataset.next_batch(batch_size)
    xuindex_batch, _, xuindex_batch_index = uindex_dataset.random_batch(batch_size)
    xu_batch = samples[xuindex_batch]
    xn_batch_index = neighborIndex[xl_batch_index, :k_of_knn]# .flatten()
    xr_batch_index = remoteIndex[xl_batch_index, :m_of_knn]
    xun_batch_index = neighborIndex[xuindex_batch_index + labeled_data_size, :k_of_knn]# .flatten()
    xur_batch_index = remoteIndex[xuindex_batch_index + labeled_data_size, :m_of_knn]
    wl_n = RBF_matrix[xl_batch_index.reshape(batch_size, 1), xn_batch_index].reshape(-1, k_of_knn, 1)# 该reshape是必要的，不能去掉
    wl_r = RBF_matrix[xl_batch_index.reshape(batch_size, 1), xr_batch_index].reshape(-1, m_of_knn, 1)
    wu_n = RBF_matrix[(xuindex_batch_index + labeled_data_size).reshape(batch_size, 1), xun_batch_index].reshape(-1, k_of_knn, 1)
    wu_r = RBF_matrix[(xuindex_batch_index + labeled_data_size).reshape(batch_size, 1), xur_batch_index].reshape(-1, m_of_knn, 1)
    # breakpoint()
    _, loss_1 = sess.run(
        [train1,loss_sum_1],
        feed_dict={
            xl: xl_batch.reshape(-1, 1, num_features),
            xn: sample_data[xn_batch_index].reshape(-1, k_of_knn, num_features),
            xr: sample_data[xr_batch_index].reshape(-1, m_of_knn, num_features),
            yt: yt_batch,
            wij_n: wl_n,
            wij_r: wl_r
        })
    writer.add_summary(loss_1, i)
    _, loss_2 = sess.run(
        [train2,loss_sum_2],
        feed_dict={
            xl: xu_batch.reshape(-1, 1, num_features),
            xn: sample_data[xun_batch_index].reshape(-1, k_of_knn, num_features),
            xr: sample_data[xur_batch_index].reshape(-1, m_of_knn, num_features),
            wij_n: wu_n,
            wij_r: wu_r
        })
    writer.add_summary(loss_2, i)
    # test_x_batch, test_y_batch, _ = test_dataset.next_batch(300)
    test_accuracy_sum = sess.run(
        acc_sum_1, feed_dict={
            xl: test_dataset.data.reshape(-1, 1, num_features),
            yt: test_dataset.labels
        })
    writer.add_summary(test_accuracy_sum, i)
    train_accuracy_sum = sess.run(
        acc_sum_2,
        feed_dict={
            xl: xu_batch.reshape(-1, 1, num_features),
            yt: uindex_dataset.labels[xuindex_batch_index]
        })
    writer.add_summary(train_accuracy_sum, i)

writer.close()
