from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
import keras
from ResNetModel import ResNet

# 入力画像の次元とチャンネル
img_rows, img_cols, img_channels = 32, 32, 3

batch_size = 256
num_classes = 10
epochs = 50

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# one_hotに変換
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model=ResNet(img_rows,img_cols,img_channels, x_train)
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = model.fit(x_train, y_train,
                            batch_size=256,
                            epochs=50,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            shuffle=True)