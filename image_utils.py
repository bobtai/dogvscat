import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array

CAT_IMAGES_PATH = "/Users/Bob/PetImages/Cat/"
DOG_IMAGES_PATH = "/Users/Bob/PetImages/Dog/"
DATA_PATH = "/Users/Bob/dogvscat/data/"
MODEL_PATH = "/Users/Bob/dogvscat/model/cnn_model.h5"

TRAIN_DATA = "train"
TEST_DATA = "test"
CAT = "cat"
DOG = "dog"
IMAGE_LABELS = {0: CAT, 1: DOG}

TARGET_WIDTH = 70
TARGET_HEIGHT = 70


def get_all_images_path(image_type):
    """
    to get all paths of cat or dog images
    :param image_type: "cat" or "dog"
    :return: the paths of cat or dog images
    """
    images_dir = CAT_IMAGES_PATH if image_type == CAT else DOG_IMAGES_PATH
    all_images = os.listdir(images_dir)
    for i in range(len(all_images)):
        all_images[i] = images_dir + all_images[i]
    return all_images


def convert_images_to_array_in_tensorflow(image_paths):
    """
    convert the input images to a 4D(len, 70, 70, 3) vector in tensorflow
    :param image_paths:
    :return: an images vector which contains the features of input images
    """
    # define a graph, includes tensor and flow
    images_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    images_reader = tf.WholeFileReader()
    image_names, image_contents = images_reader.read(images_queue)
    # decode images to rgb array
    decoded_images = tf.image.decode_jpeg(image_contents, channels=3)
    # resize inconsistent size images to same size
    resizing_images = tf.image.resize_image_with_crop_or_pad(decoded_images, 500, 500)
    # resize images to small size for reducing computing
    compressed_images = tf.image.resize_images(resizing_images, [TARGET_WIDTH, TARGET_HEIGHT])

    images_array = []
    with tf.Session() as sess:
        # initial all variables
        initial = tf.global_variables_initializer()
        sess.run(initial)

        # create multi threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # get all images matrix, and they were appended to images_array
        for i in range(len(image_paths)):
            one_image = sess.run(compressed_images)
            images_array.append(one_image)

        # stop multi threads
        coord.request_stop()
        coord.join(threads)

    return images_array


def convert_images_to_array_in_keras(image_paths):
    """
    convert the input images to a 4D(len, 70, 70, 3) vector in keras
    :param image_paths:
    :return: an images vector which contains the features of input images
    """
    images_array = []
    for i in range(len(image_paths)):
        img = load_img(image_paths[i], target_size=(TARGET_WIDTH, TARGET_HEIGHT))  # this is a PIL image
        img_arr = img_to_array(img)  # this is a numpy array with shape (70, 70, 3)
        images_array.append(img_arr)

    return images_array  # this is a numpy array with shape (len, 70, 70, 3)


def plot_images(features, labels, offset, length, predictions=[]):
    """
    :param features:
    :param labels:
    :param offset:
    :param length:
    :param predictions:
    """
    fig = plt.gcf()
    fig.set_size_inches(15, 15)

    if length > 20:
        length = 20

    for i in range(length):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(features[offset])
        title = IMAGE_LABELS[labels[offset][0]]
        if len(predictions) > 0:
            title += "=>" + IMAGE_LABELS[predictions[offset]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        offset += 1

    plt.subplots_adjust(hspace=0.30)
    plt.show()


def get_one_type_full_array(type, offset, length):
    """
    to get cat or dog 4D vector
    :param type:
    :param offset:
    :param length:
    :return: a cat or dog 4D vector
    """
    image_paths = get_all_images_path(type)
    batch = 100
    top = offset + length
    image_features = []
    for i in range(offset, top, batch):
        batch_image_paths = image_paths[i:(i+batch)]
        if i == offset:
            image_features = convert_images_to_array_in_keras(batch_image_paths)
        else:
            image_features = image_features + convert_images_to_array_in_keras(batch_image_paths)
    print(type + ". " + str(offset) + "~" + str(top) + " is finished.")
    return image_features


def prepare_data(file_name, offset, length):
    """
    prepare training or testing data
    :param file_name: TRAIN_DATA or TEST_DATA
    :param offset:
    :param length:
    :return: a cat and dog full 4D vector
    """
    cat_features = get_one_type_full_array(CAT, offset, length)
    dog_features = get_one_type_full_array(DOG, offset, length)
    all_features = cat_features + dog_features
    all_features = np.array(all_features, dtype=np.uint8)

    cat_labels = [0] * length
    dog_labels = [1] * length
    all_labels = cat_labels + dog_labels
    all_labels = np.array(all_labels, dtype=np.uint8)

    indexes = np.random.permutation(all_labels.shape[0])
    rand_all_features = all_features[indexes]
    rand_all_labels = all_labels[indexes]
    print("image_features's shape=", rand_all_features.shape)
    print("image_labels's shape=", rand_all_labels.shape)

    # save data
    rand_all_features.tofile(DATA_PATH + file_name + "_features")
    rand_all_labels.tofile(DATA_PATH + file_name + "_labels")


def load_data(file_name):
    """
    load training and testing data
    :param file_name:
    :return: image features and labels vectors
    """
    features = np.fromfile(DATA_PATH + file_name + "_features", dtype=np.uint8)
    features = features.reshape(-1, TARGET_WIDTH, TARGET_HEIGHT, 3)
    labels = np.fromfile(DATA_PATH + file_name + "_labels", dtype=np.uint8)
    labels = labels.reshape(-1, 1)
    return features, labels


if __name__ == "__main__":
    # prepare_data(TRAIN_DATA, 1, 5000)
    # prepare_data(TEST_DATA, 5001, 1000)

    features, labels = load_data(TRAIN_DATA)
    plot_images(features, labels, 0, 10)
