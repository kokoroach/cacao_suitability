import os

# -----------------------
# SET THEANO AS BACKEND
# -----------------------
def set_keras_backend(backend):
    os.environ['KERAS_BACKEND'] = backend
    # os.environ['THEANO_FLAGS'] = 'device=cuda,force_device=True,floatX=float32'
set_keras_backend('theano')


# -----------------------
# SET UP SWISH: An Activation Function
# -----------------------
from keras.utils.generic_utils import get_custom_objects

def swish(x, beta=1):
    return (x * sigmoid(beta * x))

def set_up_swish(custom_objects=None):
    get_custom_objects()['swish'] = swish

# Set up Custom Activation
set_up_swish()


# -----------------------
# ANN CORE
# -----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cacao_config as CF
import wclim_config as WC

from keras.backend import sigmoid
from keras.layers import Dense, Activation
from keras.models import Sequential, model_from_json



def generate_dataset():
    # TRAIN Data
    period = 'baseline'

    src_csv = os.path.join(CF.LINDX_DIR, period, f'{period}_lindx.csv')
    baseline = np.loadtxt(src_csv, delimiter=",", skiprows=1)

    # split into input (X) and output (Y) variables
    X = baseline[:,2:6]
    Y = baseline[:,6] / 100

    # TEST Data
    period = '2050s'

    test_csv = os.path.join(CF.LINDX_DIR, period, f'{period}_lindx.csv')
    future = np.loadtxt(test_csv, delimiter=",", skiprows=1)

    # split into input (X) and output (Y) variables
    X_test = future[:,2:6]
    Y_test = future[:,6] / 100

    return X, Y, X_test, Y_test


def create_model(optimizer, network, activation, epoch):
    model = Sequential()
    if len(network) == 1:
        model.add(Dense(network[0], input_dim=4, kernel_initializer='truncated_normal' , activation=activation))
        model.add(Dense(1, kernel_initializer='truncated_normal', activation='sigmoid'))
    elif len(network) == 2:
        model.add(Dense(network[0], input_dim=4, kernel_initializer='truncated_normal' , activation=activation))
        model.add(Dense(network[1], kernel_initializer='truncated_normal', activation=activation))
        model.add(Dense(1, kernel_initializer='truncated_normal', activation='sigmoid'))
    else:
        raise Exception

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])
    return model


def test_activation(activation, layer, optimizer='sgd', epoch=30):
    if layer == 1:
        network = (3)
    elif layer == 2:
        network = (2,1)
    else:
        raise Exception

    X, Y, _, _ = generate_dataset()

    model = create_model(optimizer=optimizer, network=network, activation=activation, epoch=epoch)

    # Fit the model
    history = model.fit(X, Y,
        epochs=epoch, batch_size=64, validation_split=0.2, verbose=2, use_multiprocessing=True)

    title = f'TEST Activation: {activation} ({layer}L)'

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title(title)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.show()

    scores = model.evaluate(X, Y)
    print("\n", scores)
    print("%s: %.2f%%. MSE: %.8f" % (model.metrics_names[1], scores[1]*100, scores[0]))


def _get_title(optimizer, activation, network):
    act = {
        'relu': 'ReLU',
        'elu': 'ELU',
        'swish': 'Swish',
        'softplus': 'Softplus'
    }
    opt = {
        'sgd': 'SGD',
        'adagrad': 'Adagrad',
        'adam': 'Adam',
        'rmsprop': 'RMSProp',
        'adamax': 'Adamax'
    }
    if len(network) == 1:
        network = '({})'.format(network[0])

    return '{} {} {}'.format(opt[optimizer], act[activation], network)


def test_models(epoch=100):
    optimizers = ['sgd', 'adagrad', 'rmsprop', 'adam', 'adamax']
    activations = ['relu', 'elu', 'swish', 'softplus']
    topologies = [
        (3,), (8,), (9,), (12,),
        (2,1), (4,4), (5,4), (6,6)
    ]

    dataset = generate_dataset()

    for opt in optimizers:
        for act in activations:
            for top in topologies:
                run_model(optimizer=opt, activation=act, network=top, epoch=epoch, dataset=dataset)


def train_model(optimizer=None, activation=None, network=None, epoch=1000):
        dataset = generate_dataset()

        run_model(optimizer=optimizer, activation=activation, network=network, epoch=epoch, dataset=dataset, is_final=True)


def run_model(optimizer=None, network=None, activation=None, epoch=None, save_model=False, dataset=None, is_final=False):
    if not is_final:
        save_dir = os.path.join(CF.ANN_DIR, 'test_model')
    else:
        save_dir = os.path.join(CF.ANN_DIR, 'model_selection')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = create_model(optimizer, network, activation, epoch)
    (X, Y, X_test, Y_test) = dataset

    # Fit the model
    history = model.fit(X, Y,
        epochs=epoch, batch_size=64, validation_split=0.2, verbose=2, use_multiprocessing=True)

    title = _get_title(optimizer, activation, network)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title(title)

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')


    if len(network) == 1:
        filename = '{}_{}_{}'.format(optimizer, activation, network[0])
    else:
        filename = '{}_{}_{}-{}'.format(optimizer, activation, network[0], network[1])

    image_path = os.path.join(save_dir, f'{filename}.png')
    plt.savefig(image_path, orientation='landscape')

    # test prediction
    scores = model.evaluate(X_test, Y_test)
    result = "\n%s \t%s: %.2f%%, MSE: %.8f" % (title, model.metrics_names[1], scores[1]*100, scores[0])

    result_file =  os.path.join(save_dir, '_model_result.txt')
    with open(result_file, 'a+') as f:
        f.write(result)

    plt.clf()

    if not is_final:
        return

    save_dir = os.path.join(save_dir, 'artifacts')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # serialize model to JSON
    m_content = model.to_json(indent=2)

    model_json = os.path.join(save_dir, f'{filename}.json')
    with open(model_json, "w") as json_file:
        json_file.write(m_content)

    # serialize weights to HDF5
    model_h5 = os.path.join(save_dir, f'{filename}.hd5')
    model.save_weights(model_h5)


def apply_model(optimizer=None, network=None, activation=None):
    artifacts_dir = os.path.join(CF.ANN_DIR, 'model_selection', 'artifacts')

    if len(network) == 1:
        filename = '{}_{}_{}'.format(optimizer, activation, network[0])
    else:
        filename = '{}_{}_{}-{}'.format(optimizer, activation, network[0], network[1])


    # load json and create model
    json_file = os.path.join(artifacts_dir, f'{filename}.json')
    with open(json_file , 'r' ) as f:
        model_json = f.read()
    loaded_model = model_from_json(model_json)

    # load weights into new model
    weights_file = os.path.join(artifacts_dir, f'{filename}.hd5')
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")

    # Set periods to all
    periods = list(WC.PERIODS.keys())
    periods.append('baseline')

    for period in periods:
        csv_file = os.path.join(CF.LINDX_DIR, period, f'{period}_lindx.csv')

        df = pd.read_csv(csv_file, header=0)

        clim_data = df.iloc[:,2:6]

        # get prediction
        result = loaded_model.predict(clim_data,
                    batch_size=64,
                    workers=2,
                    use_multiprocessing=True)

        result *= 100

        lindx_data = pd.Series(result.flatten())

        df['lindx'] = lindx_data.astype(float).round(2)

        # TO CSV
        out_csv = os.path.join(CF.LINDX_DIR, period, f'{period}_lindx_predicted.csv')
        df.to_csv(out_csv, index=False)



if __name__ == '__main__':

    # ----------------------
    # 1. TEST ACTIVATION FUNCTIONS
    # ----------------------
    # layer = 2
    # activation = 'swish'
    # test_activation(activation, layer)

    # ----------------------
    # 2. TEST MODELS
    # ----------------------
    # epoch = 100
    # test_models(epoch)

    # ----------------------
    # 3. TRAIN SELECTED MODELS
    # ----------------------
    # optimizer = 'adamax'
    # activation = 'swish'
    # network = (6,6)

    # train_model(
    #     optimizer=optimizer,
    #     activation=activation,
    #     network=network)


    # ----------------------
    # 4. GENERATE PREDICTION TO LINDX CSV
    # ----------------------
    # optimizer = 'adamax'
    # activation = 'softplus'
    # network = (5,4)

    # apply_model(optimizer=optimizer, network=network, activation=activation)

    pass
