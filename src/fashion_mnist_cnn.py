# Импорт необходимых библиотек
from keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import numpy as np

def load_train(path):
    """
    Загружает обучающие данные из .npy файлов.
    """
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    # Преобразуем плоский вектор (784) в изображение 28x28 с одним каналом и нормализуем
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
    return features_train, target_train


def create_model(input_shape):
    """
    Создаёт модель свёрточной нейросети.
    Возвращает скомпилированную модель Keras.
    """
    model = Sequential()
    optimizer = Adam()

    # Первый свёрточный слой: 6 фильтров 5x5, активация tanh, входное изображение 28x28x1
    model.add(Conv2D(filters=6, kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='tanh'))
    
    # Усредняющий пулинг с окном 2x2 и шагом 2
    model.add(AvgPool2D(pool_size=(2,2), strides=2))

    # Второй свёрточный слой: 16 фильтров 5x5, без padding ('valid'), активация tanh
    model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
    
    # Усредняющий пулинг с окном 2x2 и шагом 2
    model.add(AvgPool2D(pool_size=(2,2), strides=2))

    # Преобразование многомерного выхода в плоский вектор
    model.add(Flatten())

    # Полносвязные слои
    model.add(Dense(units=120, activation='tanh'))
    model.add(Dense(units=84, activation='tanh'))
    # Выходной слой с 10 нейронами (по числу классов) и softmax
    model.add(Dense(units=10, activation='softmax'))

    # Компиляция модели
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=5,
                steps_per_epoch=None, validation_steps=None):
    """
    Обучает модель на тренировочных данных и оценивает на тестовых.
    - model: скомпилированная модель Keras
    - train_data: кортеж (features_train, target_train)
    - test_data: кортеж (features_test, target_test)
    - batch_size: размер мини-выборки
    - epochs: количество эпох
    - steps_per_epoch: количество шагов за эпоху (если не задано, определяется автоматически)
    - validation_steps: количество шагов валидации
    Возвращает обученную модель.
    """
    features_train, target_train = train_data
    features_test, target_test = test_data

    model.fit(features_train, target_train,
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model
