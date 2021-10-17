#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import numpy as np
#import matplotlib.pyplot as plt


from tensorflow.keras.datasets import mnist          # библиотека базы выборок Mnist
from tensorflow import keras                         # библиотека Keras
from tensorflow.keras.layers import Dense, Flatten   # модули библиотеки Keras для создания слоев нейронной сети

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # загрузка выборки из базы данных

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10) # загрузка классов для обучения нейронной сети
y_test_cat = keras.utils.to_categorical(y_test, 10)   # загрузка классов для тестирования нейронной сети

# создание модели нейронной сети
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль

# компиляция модели
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# обучение модели на обучающей выборке
model.fit(x_train, y_train_cat, batch_size=32, epochs=100)

# проверка точности на тренировочной выборке
model.evaluate(x_test, y_test_cat)

# сохранение обученной модели
model.save('test8.h5') 