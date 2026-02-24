# Импорт необходимых библиотек
from tensorflow.keras.datasets import fashion_mnist
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

def load_train(path):
    """
    Загружает обучающие признаки и метки из файлов .npy.
    Возвращает кортеж (признаки, целевых меток) после нормализации и преобразования в плоский вид.
    """
    # Загрузка признаков и целевых меток из бинарных файлов NumPy
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    # Преобразование каждого изображения из матрицы 28x28 в вектор длиной 784
    # и нормализация значений пикселей к диапазону [0, 1] делением на 255
    features_train = features_train.reshape(features_train.shape[0], 28 * 28) / 255.0
    
    return features_train, target_train


def create_model(input_shape):
    """
    Создаёт полносвязную нейронную сеть для классификации 10 классов.
    Возвращает скомпилированную модель Keras.
    """
    model = Sequential()  # инициализация последовательной модели
    
    # Первый скрытый слой с 256 нейронами, активация ReLU
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    
    # Второй скрытый слой с 128 нейронами, активация ReLU
    model.add(Dense(128, activation='relu'))
    
    # Третий скрытый слой с 64 нейронами, активация ReLU
    model.add(Dense(64, activation='relu'))
    
    # Выходной слой: 10 нейронов (по числу классов), активация softmax для вероятностного вывода
    model.add(Dense(10, activation='softmax'))
    
    # Компиляция модели:
    # - оптимизатор: стохастический градиентный спуск (SGD)
    # - функция потерь: sparse_categorical_crossentropy
    # - метрика: точность (accuracy)
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=5,
                steps_per_epoch=None, validation_steps=None):
    """
    Обучает модель на тренировочных данных и оценивает на тестовых.
    train_data, test_data — кортежи (признаки, метки).
    Возвращает обученную модель.
    """
    # Распаковка тренировочных и тестовых данных
    features_train, target_train = train_data
    features_test, target_test = test_data
    
    # Запуск обучения
    # validation_data — данные для валидации после каждой эпохи
    # batch_size — количество образцов на одно обновление весов
    # epochs — количество эпох (полных проходов по данным)
    # steps_per_epoch, validation_steps — используются при работе с генераторами (здесь не нужны)
    # verbose=2 — вывод одной строки на эпоху
    # shuffle=True — перемешивание данных перед каждой эпохой
    model.fit(features_train, target_train,
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
    
    return model
