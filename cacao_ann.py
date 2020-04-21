import os
import matplotlib.pyplot as plt
import numpy as np

import cacao_config as CF

def set_keras_backend(backend):
    os.environ['KERAS_BACKEND'] = backend
    # os.environ['THEANO_FLAGS'] = 'device=cuda,force_device=True,floatX=float32'
set_keras_backend('theano')

from keras.utils.generic_utils import get_custom_objects

def swish(x, beta=1):
    return (x * sigmoid(beta * x))

def set_up_swish(custom_objects=None):
    get_custom_objects()['swish'] = swish

# Set up Custom Activation
set_up_swish()


from keras.backend import sigmoid
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_json



# custom create model
def create_model(optimizer, network, activation, epoch):
    model = Sequential()
    if network == 'FF':
        model.add(Dense(4, input_dim=4, kernel_initializer='truncated_normal' , activation=activation))
        model.add(Dense(1, kernel_initializer='truncated_normal', activation='sigmoid'))
    elif network == 'DFF':
        model.add(Dense(5, input_dim=4, kernel_initializer='truncated_normal' , activation=activation))
        model.add(Dense(5, kernel_initializer='truncated_normal', activation=activation))
        model.add(Dense(1, kernel_initializer='truncated_normal', activation='sigmoid'))
    else:
        raise
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])
    return model


def test_ann(optimizer=None, network=None, activation=None, epoch=None, save_model=False):
    if activation not in ['relu', 'swish']:
        raise

    src_csv = os.path.join(CF.CROPPED_DIR, 'baseline', 'baseline_lindx.csv')
    dataset = np.loadtxt(src_csv, delimiter=",", skiprows=1)

    # split into input (X) and output (Y) variables
    X = dataset[:,2:6]
    Y = dataset[:,6] / 100

    model = create_model(optimizer, network, activation, epoch)

    # Fit the model
    history = model.fit(X, Y,
        epochs=epoch, batch_size=64, validation_split=0.2, verbose=2, use_multiprocessing=True)


    _act = {'relu': 'ReLU', 'swish': 'Swish'}
    _opt = {
        'sgd': 'SGD',
        'adagrad': 'Adagrad',
        'adam': 'Adam',
        'rmsprop': 'RMSprop',
        'adamax': 'Adamax'
    }
    title = '{} {} ({})'.format(_opt[optimizer], _act[activation], network)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    # evaluate the model
    test_path = os.path.join(CF.CROPPED_DIR, '2050s', '2050s_lindx.csv')
    test_dataset = np.loadtxt(test_path, delimiter=",", skiprows=1)

    # split into input (X) and output (Y) variables
    X_test = test_dataset[:,2:6]
    Y_test = test_dataset[:,6] / 100

    # test prediction
    scores = model.evaluate(X_test, Y_test)
    print("\n", scores)
    print("%s: %.2f%%. MSE: %.8f" % (model.metrics_names[1], scores[1]*100, scores[0]))

    if save_model:
        # serialize model to JSON
        m_content = model.to_json(indent=2)

        m_dir = os.path.dirname(__file__)
        m_name = f'model_{optimizer}_{activation}_{network}'

        model_json = os.path.join(m_dir, f'{m_name}.json')
        with open(model_json, "w") as json_file:
            json_file.write(m_content)

        # serialize weights to HDF5
        model_h5 = os.path.join(m_dir, f'{m_name}.hd5')
        model.save_weights(model_h5)


def apply_model(csv_file, json_file, weights_file, model_name=None):
    import json
    import pandas as pd

    if not model_name:
        raise
 
    # load json and create model
    model_json = None
    with open(json_file , 'r' ) as f:
        model_json = f.read()
    loaded_model = model_from_json(model_json)

    # # load weights into new model
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")


    df = pd.read_csv(csv_file, header=0)

    # set new df
    new_df = df.loc[:, df.columns != 'lindx']
    clim_data = new_df.iloc[:,2:6]

    # test prediction
    result = loaded_model.predict(clim_data,
                batch_size=64,
                workers=2,
                use_multiprocessing=True)
    
    result *= 100
    lindx_data = np.round(result, 2)

    new_df = new_df.assign(lindx=lindx_data)

    # TO CSV
    out_dir = os.path.dirname(csv_file)
    out_csv = os.path.join(out_dir, f'{model_name}_lindx.csv')
    new_df.to_csv(out_csv, index=False)


def main():
    # --------------------
    # 1. Test ANN Model
    # --------------------

    optimizer = 'adam'
    network = 'DFF'
    activation = 'swish'
    epoch = 1000
    
    # test_ann(
    #     optimizer=optimizer,
    #     network=network,
    #     activation=activation,
    #     epoch=epoch,
    #     save_model=True
    # )

    # --------------------
    # 2. Generate TIF from prediction
    # --------------------
    # import cacao_config as CF

    # period = 'baseline'
    # model_name = f'model_adam_swish_DFF'

    # csv_file = os.path.join(CF.CROPPED_DIR, period, f'{period}_lindx.csv')
    # json_file = os.path.join(CF.MODELS_DIR, f'{model_name}.json')
    # weights_file = os.path.join(CF.MODELS_DIR, f'{model_name}.hd5')

    # apply_model(csv_file, json_file, weights_file, model_name=model_name)

if __name__ == '__main__':
    main()

    
