### Author: Shashank Tyagi ###
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

def _read_and_preprocess_image(image_name):
    image = np.array(Image.open(image_name))
    # do preprocessing here

    return image

def data_generator(data_dir, num_epochs, batch_size):
# current implementation uses only PNG images. Can be modified to uses other file types
    all_files = glob.glob(os.path.join(DATA_DIR, '*', '*.png'))
    filename_queue = tf.train.string_input_producer(all_files,
                                                    num_epochs=NUM_EPOCHS)

# process image and labels here
    image_name = filename_queue.dequeue()
    image = _read_and_preprocess_image(image_name)
    label = tf.cast(image_name.split('/')[-2], tf.int32)
###############################

    images_batch, labels_batch = tf.train.shuffle_batch([image, label],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)
    return images_batch, labels_batch



if __name__ == '__main__':


    # dir containing Keras style png images
    DATA_DIR = './'
    NUM_EPOCHS = 10
    BATCH_SIZE = 2
    images_batch, labels_batch = data_generator(DATA_DIR, NUM_EPOCHS, BATCH_SIZE)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    init = tf.local_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            step += 1
            batch_x, batch_y = sess.run([images_batch, labels_batch])
            print step, batch_x.shape, batch_y.shape
    except tf.errors.OutOfRangeError:
        print 'Done!'
    finally:
        coord.request_stop()

    coord.join()
    sess.close()

    # import ipdb;ipdb.set_trace()
