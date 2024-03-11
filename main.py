import multiprocessing
import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
BATCH_SIZE = 64
vector_length = 1024

# Loading the data
def Load_Data(dstfolder = r'ChineseNumber/'):
    print('loading data')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.vstack((X_train, X_test))
    y_train = np.append(y_train, y_test)
    
    # check the width and height of image
    assert IMAGE_HEIGHT == 28 and IMAGE_WIDTH == 28
    
    # load the output data if exist
    if os.path.exists('data/X_test.npy'):
        X_test = np.load('data/X_test.npy')
    
    # the output image function as label
    else:
        Imgs = []
        for i in range(10):
            Imgs.append(cv2.imread(dstfolder + str(i) + '.bmp',
                                   cv2.IMREAD_GRAYSCALE))
        X_test = np.zeros(shape=(0, IMAGE_WIDTH, IMAGE_HEIGHT), dtype='uint8')
        for i in tqdm(y_train):
            X_test = np.insert(X_test, X_test.shape[0], Imgs[i], axis=0)
        np.save('data/X_test.npy', X_test)
        
    # X_train.shape = X_test.shape = (70000,28,28)
    # y_train.shape = (70000,) , function as labels
    return X_train, X_test, y_train

def create_batch(index):
    x = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    y = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)

    if index < X_source.shape[0] - BATCH_SIZE:
        _batch_size = index + BATCH_SIZE
    else:
        _batch_size = X_source.shape[0]
    
    # Normalize the data to 0-1 scale.
    for k, image in enumerate(X_source[index:_batch_size]):
        x[k, :, :, :] = image / 255.0

    for k, image in enumerate(b_i[index:_batch_size]):
        y[k, :, :, :] = image / 255.0

    return x, y

def input_layer(encoder_input):
    conv = tf.layers.conv2d(
        inputs=encoder_input,
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer=tf2.initializers.GlorotUniform(),
        activation=tf.nn.tanh
    )

    conv_output = tf.layers.flatten(conv)

    dense = tf.layers.dense(
        inputs=conv_output,
        units=1024,
        activation=tf.nn.tanh
    )

    inputs = tf.layers.dense(
        inputs=dense,
        units=vector_length,
        activation=tf.nn.tanh
    )

    return inputs

def output_layer(code_sequence, batch_size):
    dense = tf.layers.dense(
        inputs=code_sequence,
        units=1024,
        activation=tf.nn.tanh
    )

    output = tf.layers.dense(
        inputs=dense,
        units=(IMAGE_HEIGHT - 2) * (IMAGE_WIDTH - 2) * 3,
        activation=tf.nn.tanh
    )

    deconv_input = tf.reshape(
        output,
        (batch_size, IMAGE_HEIGHT - 2, IMAGE_WIDTH - 2, 3)
    )

    deconv1 = tf.layers.conv2d_transpose(
        inputs=deconv_input,
        filters=3,
        kernel_size=(3, 3),
        kernel_initializer=tf2.initializers.GlorotUniform(),
        activation=tf.sigmoid
    )

    output = tf.cast(tf.reshape(deconv1, (batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) * 255.0, tf.uint8)

    return deconv1, output


def predict(X, batch_size=1):
    feed = {
        input_images: X.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) / 255.0,
        output_images: np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32),
        t_batch_size: batch_size
    }
    return session.run([output_batch], feed_dict=feed)[0]


tf.disable_eager_execution()
tf.disable_v2_behavior()

np.random.seed(42)
tf.set_random_seed(42)

EPOCHS_COUNT = 200
use_gpu = True

X_source, b_i, y_train = Load_Data()
X_source = X_source[:, :, :, np.newaxis]
b_i = b_i[:, :, :, np.newaxis]


graph = tf.Graph()
with graph.as_default():
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)

    with tf.device('/gpu:0' if use_gpu else '/cpu:0'):
        input_images = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        output_images = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        t_batch_size = tf.placeholder(tf.int32, shape=())

        code_layer = input_layer(encoder_input=input_images)
        deconv_output, output_batch = output_layer(
            code_sequence=code_layer,
            batch_size=t_batch_size
        )

        loss = tf.nn.l2_loss(output_images - deconv_output)
        
        # Set decay learning rate 
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.00025,
            global_step=global_step,
            decay_steps=int(X_source.shape[0] / (2 * BATCH_SIZE)),
            decay_rate=0.9,
            staircase=True
        )
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_step = trainer.minimize(loss)

if __name__ == '__main__':
    config = tf.ConfigProto(
        intra_op_parallelism_threads=multiprocessing.cpu_count(),
        inter_op_parallelism_threads=multiprocessing.cpu_count(),
        log_device_placement=True,
        allow_soft_placement=True,
        device_count={
            'CPU': 1,
            'GPU': 1 if use_gpu else 0
        }
    )
    session = tf.InteractiveSession(graph=graph, config=config)
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    
    # Load saved model if exists
    if os.path.exists('saved_model/model.ckpt.data-00000-of-00001'):
        print('Start loading')
        saver.restore(session, "./saved_model/model.ckpt")
        
    else: 
        print('Start training')
        for i in range(EPOCHS_COUNT):
            total_loss = 0.0
    
            for j in range(0, X_source.shape[0], BATCH_SIZE):
                X, Y = create_batch(j)
    
                feed_dict = {
                    input_images: X,
                    output_images: Y,
                    t_batch_size: BATCH_SIZE
                }
    
                _, t_loss = session.run([training_step, loss], feed_dict=feed_dict)
                total_loss += t_loss
    
            print('Epoch: {} -> Loss: {}'.format(
                i + 1, total_loss / float(b_i.shape[0]))
            )
        # Inside the training loop, after training is complete
        # Save the model
        save_path = saver.save(session, "saved_model/model.ckpt")
        print("Model saved in path: %s" % save_path)
        
    # Saving predicting result
    cnt = 0
    for i in tqdm(range(50), desc='Writing images to disc'):
        restored_images = np.zeros(shape=(2, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        restored_images[0, :, :, :] = X_source[i]

        predicted = predict(restored_images[0])[0]
        cv2.imwrite('./input/{}'.format(cnt)+'_'+str(y_train[i])+'.jpg', restored_images[0])
        cv2.imwrite('./output/{}'.format(cnt)+'_'+str(y_train[i])+'.jpg', predicted)
        cnt += 1
        
    