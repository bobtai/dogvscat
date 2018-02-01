import numpy as np
import pandas as pd

import images_utils as img_utils

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

np.random.seed(10)


def train_model():
    # load training data
    train_features, train_labels = img_utils.load_data(img_utils.TRAIN_DATA)
    print("train_features.shape=", train_features.shape)
    print("train_labels.shape=", train_labels.shape)

    # normalize data
    train_features = train_features / 255  # normalize value between 0 to 1
    train_labels = np_utils.to_categorical(train_labels)  # change label to onehot format
    # print("train_features[0] = ", train_features[0])
    # print("train_labels[0] = ", train_labels[0])

    # create model
    model = Sequential()

    # add convolution layer 1 has 32 filters, which will generate 32 pictures
    # original picture size = 70*70, 3 is RGB value
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     input_shape=(train_features.shape[1],  # shape[1] = 70
                                  train_features.shape[2],  # shape[2] = 70
                                  train_features.shape[3]),  # shape[3] = 3
                     activation='relu',
                     padding='same'))
    # add pooling layer 1 to reduce picture dimension
    model.add(MaxPooling2D(pool_size=(2, 2)))  # the picture dimension become 35*35 after pooling

    # add convolution layer 2 has 64 filters, which will generate 64 pictures
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # the picture dimension become 17.5*17.5 after pooling

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # add flatten layer to convert 128*width*height pictures to 1D vector
    model.add(Flatten())

    # add dense layer has 512 neurons
    model.add(Dense(512, activation='relu'))  # hidden layer
    # if training result is overfitting, add dropout to avoid overfitting
    model.add(Dropout(rate=0.30))

    # add dense layer has 2 neurons, because the number of categories is 2
    model.add(Dense(2, activation='softmax'))  # output layer

    # define training method
    model.compile(loss='categorical_crossentropy',  # loss function
                  optimizer='adam',  # optimizer
                  metrics=['accuracy'])  # show accuracy in the training process

    # start training
    model.fit(x=train_features,  # assign features
              y=train_labels,  # assign labels
              epochs=10,  # training 10 epochs
              batch_size=100,  # training 10000/100 times in one epoch
              verbose=2)  # show the training process

    # save model
    model.save(img_utils.MODEL_PATH)
    print("Training model is successful.")


def test_model():
    # load testing data
    test_features, test_labels = img_utils.load_data(img_utils.TEST_DATA)
    print("test_features.shape=", test_features.shape)
    print("test_labels.shape=", test_labels.shape)

    # normalize data
    normalized_test_features = test_features / 255
    onehot_test_labels = np_utils.to_categorical(test_labels)

    # load model
    model = load_model(img_utils.MODEL_PATH)

    # evaluate model
    accuracy = model.evaluate(normalized_test_features, onehot_test_labels)
    print("testing accuracy is", accuracy[1])

    # get result of prediction
    predictions = model.predict_classes(normalized_test_features)

    # plot result of prediction
    img_utils.plot_images(test_features, test_labels, 0, 20, predictions)

    # create confusion matrix to cross comparison
    reshape_labels = test_labels.reshape(len(test_labels))
    cross_table = pd.crosstab(reshape_labels, predictions,
                              rownames=['label'], colnames=['predict'])
    print(cross_table)

    # create labels and predictions comparison table
    comparison_table = pd.DataFrame({'label': reshape_labels,
                                     'predict': predictions})
    print(comparison_table[(comparison_table.label == 0)
                           & (comparison_table.predict == 1)])


if __name__ == "__main__":
    # train_model()
    test_model()
