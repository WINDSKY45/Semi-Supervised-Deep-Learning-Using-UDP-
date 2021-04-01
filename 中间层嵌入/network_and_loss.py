import tensorflow as tf
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D

Alpha = 1.

def supervised_loss(name):
    supervised = {'hinge_loss': hinge_loss,
                  'abs_hinge': abs_hinge_loss,
                  'abs_quadratic': abs_quadratic_loss,
                  'Cross_Entropy': Cross_Entropy}

    return supervised[name]


def graph_loss(name):
    manifold = {'LE': Laplacian_Eigenmaps,
                'LE_2':Laplacian_Eigenmaps_2,
                'LE_CNN':Laplacian_Eigenmaps_forCNN,
                'UDP with Kernel Weighting':UDP}
    return manifold[name]


def hinge_loss(y, yt):
    loss = tf.reduce_mean(tf.losses.hinge_loss(logits=y, labels=yt))
    return loss


def abs_hinge_loss(y, yt):
    m = tf.cast(1.0, tf.float32)
    loss = tf.reduce_mean(tf.nn.relu(m - tf.abs(y * yt)))
    return loss


def abs_quadratic_loss(y, yt):
    with tf.name_scope("abs_quadratic_loss"):
        loss = abs_hinge_loss(y, yt)
        return tf.square(loss)

def Cross_Entropy(y, yt):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = yt))
    return loss

def Laplacian_Eigenmaps(yi, yj, wij):
    # attention that this distance is not Euclidean Distance !!!
    # this is square of Euclidean
    with tf.name_scope("Laplacian_Eigenmaps"):
        dist = tf.reduce_sum(tf.square(yi - yj), 1)
        dist = tf.reshape(dist, (1, -1))
        w_ij = tf.to_float(wij)


        loss = tf.matmul(dist, w_ij)
        loss = tf.multiply(Alpha, loss)
        return tf.reduce_mean(loss)

def Laplacian_Eigenmaps_2(yi, yj, wij):
    with tf.name_scope('Laplacian_Eigenmaps_2'):
        dist = tf.reduce_sum(tf.square(yi - yj), 2)
        dist = tf.reshape(dist, (1, -1))
        w_ij = tf.to_float(wij)
        w_ij = tf.reshape(w_ij, (-1, 1))

        loss = tf.matmul(dist, w_ij)
        loss = tf.multiply(Alpha, loss)
        return loss

def Laplacian_Eigenmaps_forCNN(yi, yj, wij, batch_size = 10, num_classes = 10):
    with tf.name_scope('Laplacian_Eigenmaps_CNN'):
        yi = tf.reshape(yi, (-1, 1, num_classes))
        yj = tf.reshape(yj, (batch_size, -1, num_classes))
        dist = tf.reduce_sum(tf.square(yi - yj), 2)
        dist = tf.reshape(dist, (1, -1))
        w_ij = tf.to_float(wij)
        w_ij = tf.reshape(w_ij, (-1, 1))

        loss = tf.matmul(dist, w_ij)
        loss = tf.multiply(Alpha, loss)
        return loss

def UDP(yl, yn, yr, wij_n, wij_r):
    with tf.name_scope('UDP'):
        loss = Laplacian_Eigenmaps_2(yl, yn, wij_n)/Laplacian_Eigenmaps_2(yl, yr, wij_r)
        return loss

# ---------network implement part------------
def forward(name, x):
    layerlst = {
        'DNN_1': DNN_model_1,
        'DNN_2_label': DNN_embedding_space_2 + DNN_model_2_label,
        'DNN_2_embed': DNN_embedding_space_2,
        'DNN_3_label': DNN_model_3_label,
        'DNN_3_unlabel': DNN_model_3_unlabel,
        'CNN': CNN_model_1
    }
    y = x
    for layer in layerlst[name]:
        print(y)
        y = layer(y)
    return y

DNN_model_1 = [
    # Reshape((784, )),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    # Dropout(0.25),
    Dense(10),
    Activation('softmax')
]

DNN_embedding_space_2 = [
    # Reshape((784, )),
    Dense(128, activation = 'relu'),
    Dense(50, activation = 'relu')
]

DNN_model_2_label = [
# DNN_model_2_label = DNN_embedding_space_2 + [
    # Dense(50, activation = 'relu'),
    Dense(10),
    Activation('softmax')
]

DNN_embedding_space_3 = [
    # Reshape((784, )),
    Dense(128, activation = 'relu')
]

DNN_model_3_label = DNN_embedding_space_3 + [
    Dense(64, activation = 'relu'),
    Dense(10),
    Activation('softmax')
]
DNN_model_3_unlabel = DNN_embedding_space_3 + [
    # Dense(64, activation = 'relu'),
    Dense(10),
    Activation('softmax')
]

CNN_model_1 = [
    Conv2D(filters=32, kernel_size = (3, 3), kernel_initializer='glorot_normal', padding="same"),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_normal', padding="same"),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Reshape((7*7*64, )),
    Dense(256, activation='relu', kernel_initializer='glorot_normal'),
    Dense(10, activation='softmax', kernel_initializer='glorot_normal')
]

'''
CNN_embedding_space_2 = []

CNN_model_2_label = CNN_embedding_space_2 + [
    Dense(128, activation='sigmoid', kernel_initializer='lecun_uniform'),
    Reshape((128, 1, 1)),

    Conv2D(kernel_size=(2, 1), filters=8),
    Activation('sigmoid'),
    MaxPooling2D(pool_size=(2, 1)),
    Reshape((63 * 1 * 8,)),

    Dense(128, activation='tanh', kernel_initializer='lecun_uniform'),
    Dense(15, kernel_initializer='lecun_uniform'),
    Activation('softmax')
]

CNN_model_2_unlabel = CNN_embedding_space_2

# Auxiliary

CNN_embedding_space_3 = [
    Reshape((144, 1, 1)),
    Conv2D(kernel_size=(2, 1), filters=32, kernel_initializer='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),
]

CNN_model_3_label = houston_embedding_space_3 + [
    Conv2D(kernel_size=(2, 1), filters=16, kernel_initializer='glorot_normal'),
    Activation('tanh'),
    MaxPooling2D(pool_size=(2, 1)),

    Reshape((35 * 1 * 16,)),
    Dense(128, activation='sigmoid', kernel_initializer='glorot_normal'),
    Dense(15, kernel_initializer='glorot_normal'),
    Activation('softmax')
]

CNN_model_3_unlabel = houston_embedding_space_3 + [
    Reshape((71 * 1 * 32,)),
    Dense(15, kernel_initializer='glorot_normal'),
    Activation('sigmoid')
]
'''
