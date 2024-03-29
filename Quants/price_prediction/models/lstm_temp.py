from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model


class BaseLSTM():
    def __init__(self, input_shape, lr, epoch, batch, model_name):
        # self.args = args
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch
        self.model_name = model_name
        self.input_shape = input_shape
        self.early_stopping()
    
    def early_stopping(self):
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=50, verbose=0, mode='min', restore_best_weights=True)