import image_utils as img_utils

import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


# load model
model = load_model(img_utils.MODEL_PATH)


def classify_new_image(image_path):
    # normalize data
    image = load_img(image_path, target_size=(img_utils.TARGET_WIDTH, img_utils.TARGET_HEIGHT))
    image_feature = img_to_array(image)
    image_feature = image_feature.reshape(-1, img_utils.TARGET_WIDTH, img_utils.TARGET_HEIGHT, 3)
    normalized_feature = image_feature / 255

    # get predict class
    predictions = model.predict_classes(normalized_feature)
    print("This picture is", img_utils.IMAGE_LABELS[predictions[0]])

    # get predict probabilities
    probabilities = model.predict(normalized_feature)
    for i in range(len(img_utils.IMAGE_LABELS)):
        print(img_utils.IMAGE_LABELS[i] + ' :%1.5f' % (probabilities[0][i]))

    # plot predict image
    plt.figure(figsize=(2, 2))
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    input_image_path = "/Users/Bob/PetImages/Cat/2.jpg"
    classify_new_image(input_image_path)
