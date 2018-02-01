# kaggle dogs vs. cats

## Data Source
來源一：原網站（需登入）
<https://www.kaggle.com/c/dogs-vs-cats/data>
來源二：微軟（可直接下載）
<https://www.microsoft.com/en-us/download/details.aspx?id=54765>

## Raw Data
貓圖：共 12500 張，0.jpg ~ 12499.jpg
範例：1.jpg
![](https://raw.githubusercontent.com/bobtai/dogvscat/master/images/raw_data_cat.png)
狗圖：共 12500 張，0.jpg ~ 12499.jpg
範例：4.jpg
![](https://raw.githubusercontent.com/bobtai/dogvscat/master/images/raw_data_dog.png)

## Preprocessed data
images_utils.py
```python
# 設定全域變數
CAT_IMAGES_PATH = "/Users/Bob/PetImages/Cat/"  # 貓圖路徑
DOG_IMAGES_PATH = "/Users/Bob/PetImages/Dog/"  # 狗圖路徑
DATA_PATH = "/Users/Bob/dogvscat/data/"  # 訓練和測試資料集存放路徑

if __name__ == "__main__":
    # 使用 10000 張貓和狗圖產生訓練資料集，共 20000 張。
    prepare_data(TRAIN_DATA, 1, 10000)
    # 使用剩下的 2500 張貓和狗圖產生測試資料集，共 5000 張。
    prepare_data(TEST_DATA, 10001, 2500)
```
產生好的資料集如下：
![](https://raw.githubusercontent.com/bobtai/dogvscat/master/images/dataset.png)

## train and save model
images_utils.py
```python
# 設定模型存放路徑
MODEL_PATH = "/Users/Bob/dogvscat/model/cnn_model.h5"
```
train_n_test_model.py
```python
if __name__ == "__main__":
    train_model()
```
訓練完，模型如下：
![](https://raw.githubusercontent.com/bobtai/dogvscat/master/images/model.png)

## test model
train_n_test_model.py
```python
if __name__ == "__main__":
    test_model()
```

## classify new data
classify_new_image.py
```python
if __name__ == "__main__":
    # 設定某張新圖的路徑
    input_image_path = "/Users/Bob/PetImages/Cat/2.jpg"
    classify_new_image(input_image_path)
```