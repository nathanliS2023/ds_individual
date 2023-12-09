from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasClassifier
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def create_model(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_nn_model(input_dim, learning_rate=0.001):
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    return KerasClassifier(model=create_model, input_dim=input_dim, epochs=100, batch_size=10, verbose=0,
                           learning_rate=learning_rate, callbacks=[early_stopping])
