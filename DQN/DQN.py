#from keras.models import Sequential
#from keras.layers import Conv2D, Dense, Flatten
#from keras.optimizers import Adam
from torch.nn import Conv2d, ReLU, Flatten, Linear, Sequential

def create_dqn(learn_rate, input_dims, n_actions, conv_units, dense_units):
    model = Sequential([
        # provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis
                Conv2d(conv_units, (3,3), activation='relu', padding='same', input_shape=input_dims),
                Conv2d(conv_units, (3,3), activation='relu', padding='same'),
                Conv2d(conv_units, (3,3), activation='relu', padding='same'),
                Conv2d(conv_units, (3,3), activation='relu', padding='same'),
                Flatten(),
                Linear(dense_units, activation='relu'),
                ReLU,
                Linear(dense_units, activation='relu'),
                ReLU,
                Linear(n_actions, activation='linear')])

    model.compile(optimizer=Adam(lr=learn_rate, epsilon=1e-4), loss='mse')

    return model
