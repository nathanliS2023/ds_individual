from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def create_model(input_dim):
    # Define a simple neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_nn_model(input_dim):
    # Wrap the model with KerasClassifier
    return KerasClassifier(build_fn=create_model, input_dim=input_dim, epochs=100, batch_size=10, verbose=0)
