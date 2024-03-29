from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.linear_model import LinearRegression
import xgboost as xgb

def split_train_test(features, targets, split=0.9):
    N = len(features)
    training_idx = int(N * split)

    X_train, y_train = features[:training_idx].values, targets[:training_idx].values
    X_test, y_test = features[training_idx:].values, targets[training_idx:].values

    return X_train, y_train, X_test, y_test

def build_lstm_model(input_shape,  input_units=128, lr=0.001, p=0.2):
    model = Sequential([
        LSTM(input_units, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dropout(p),
        Dense(1)
    ])
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

def train_lstm(X_train, y_train, X_test, y_test, window):
    model = build_lstm_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10, verbose=1, mode='min', restore_best_weights=True)
    # Fit the model
    model.fit(X_train, y_train, batch_size=window, epochs=128, validation_data=(X_test, y_test), verbose=1, callbacks=early_stopping)
    return model

def build_lstm_base_model(input_shape, input_units=128, hidden_units=16, lr=0.001, p=0.2):
    model = Sequential([
        LSTM(input_units, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dropout(p),
        Dense(hidden_units)
    ])
    opt = Adam(learning_rate=lr)
    model.compile(opt, loss='mean_squared_error')
    return model

def build_lstm_linear(X_train, y_train, X_test, y_test, batch_size, epochs=128):

    input_shape = (X_train.shape[1], 1) # (Num_features, 1)
    lstm = build_lstm_base_model(input_shape)
    callback = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10, verbose=0, mode='min', restore_best_weights=True)
    lstm.fit(X_train, y_train, batch_size=batch_size, epochs=epochs
             , validation_data=(X_test, y_test), verbose=1, callbacks=callback)

    # Using LSTM to generate hidden features for the LR model
    hidden_X_train = lstm.predict(X_train)
    # hidden_X_test = lstm.predict(X_test)

    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(hidden_X_train, y_train)
    
    return lstm, lr_model

def build_lstm_xgboost(X_train, y_train, X_test, y_test, batch_size, epochs=128):

    input_shape = (X_train.shape[1], 1) # (Num_features, 1)
    lstm = build_lstm_base_model(input_shape)
    callback = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10, verbose=0, mode='min', restore_best_weights=True)
    lstm.fit(X_train, y_train, batch_size=batch_size, epochs=epochs
             , validation_data=(X_test, y_test), verbose=1, callbacks=callback)

    # Using LSTM to generate hidden features for the LR model
    hidden_X_train = lstm.predict(X_train)

    xgboost = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
    xgboost.fit(hidden_X_train, y_train)
    
    return lstm, xgboost


