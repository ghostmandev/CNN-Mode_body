from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, AveragePooling2D

class Model():
    def Our_Model():
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 3)))
        model.add(MaxPool2D(strides=2))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(strides=2))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(strides=2))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(strides=2))
        #     model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
        #     model.add(MaxPool2D(strides=2))
        model.add(Flatten())
        #     model.add(Dense(units = 512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=84, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=9, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


