import os, re
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv1D, TimeDistributed, LSTM, \
                         multiply, Cropping1D, GRU, CuDNNGRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.applications import vgg16
from keras import backend as K
import numpy as np
import tensorflow as tf

CAL_PATH = os.path.dirname(__file__)
PARAM_PATH = CAL_PATH + '/params/'
WEIGHTS_PATH = CAL_PATH + '/weights/'

# set up for the model rebuild
last_layer_keys = ['0_red_light', '1_hazard_stop', '2_speed_sign',\
                   '3_relative_angle', '4_center_distance', '5_veh_distance']
[RED_LIGHT, HAZARD_STOP, SPEED_SIGN, RELATIVE_ANGLE, CENTER_DISTANCE, VEH_DISTANCE] = last_layer_keys

# properties of the last layers
outputs = [2,2,4,1,1,1]
activations = ['softmax']*3 + [None]*3
cond = [False, False, False, True, True, False]

OUTPUT_SHAPES = {k:outputs[i] for i,k in enumerate(last_layer_keys)}
ACTIVATIONS = {k:activations[i] for i,k in enumerate(last_layer_keys)}
CONDITIONAL = {k:cond[i] for i,k in enumerate(last_layer_keys)}

def split_model(base_model, split_idx):
    """
    split the given model into two model instances
    """
    # rebuild the front model
    front_model = Model(inputs=base_model.input,
                        outputs=base_model.get_layer(index=split_idx).output)
    front_out = base_model.get_layer(index=split_idx).output_shape

    # build the new "tail" model
    last_layers = base_model.layers[split_idx+1:]
    inp = Input(shape=front_out[1:])
    x = inp
    for layer in last_layers: x = layer(x)
    out = x
    tail_model = Model(inp, out)

    return front_model, tail_model

def get_conv_model():
    """
    get the front model and tail model of a conv model
    and the used preprocessing function
    """
    base_model = vgg16.VGG16(include_top=False,
                             weights='imagenet',
                             input_shape=(100,222,3))
    front_model, tail_model = split_model(base_model, -5)
    preprocessing = vgg16.preprocess_input

    return front_model, tail_model, preprocessing

def load_CAL_network():
    """
    Load a model with a defined architecture from a specific training epoch.
    If block_model is true -> load all the task block models
    (workaround for keras model loading bug)
    """
    # Build the model and load the weights
    model = get_full_model()
    w_name = WEIGHTS_PATH + 'full_model_weights.h5'
    model.load_weights(w_name)

    return model

def conv_bn_dropout(x, p=0.1):
    x = BatchNormalization(axis=1)(x)
    x = Dropout(p)(x)
    return x

def vgg_to_timedistributed(seq_len, conv_dp):
    # get the tail_model
    _, tail_model, _ = get_conv_model()
    # get the new input shape
    conv_inp = tail_model.layers[0].input_shape[1:]
    inp_shape = (seq_len,) + conv_inp

    # turn into a time distributed layer
    # start at idx 1 (skip Input layer)
    inp = Input(shape=inp_shape)
    x = TimeDistributed(tail_model.layers[1], name=tail_model.layers[1].name)(inp)
    for i,l in enumerate(tail_model.layers[2:]):
        if conv_dp:
            x = TimeDistributed(BatchNormalization(axis=1), name='Batchnorm_{}'.format(i))(x)
            x = TimeDistributed(Dropout(.2), name='Conv_Dropout_{}'.format(i))(x)
        x  = TimeDistributed(l, name=l.name)(x)
    prediction = TimeDistributed(Flatten())(x)
    m = Model(inputs=[inp], outputs=prediction)
    return m

def dense_block(x_inp, p=0.5, n=64):
    """
    standard dense block (Dense - BatchNorm - Dropout)
    x = input
    n = number of nodes in the dense layer
    p = dropout
    """
    x= Dense(n, activation='relu', )(x_inp)
    x= BatchNormalization()(x)
    x= Dropout(p)(x)
    return x

def GRU_block(x_inp, p=0.5, n=64):
    # x = CuDNNGRU(n)(x_inp)
    x = GRU(n, reset_after=True, recurrent_activation='sigmoid')(x_inp)
    x = Dropout(p)(x)
    return x

def conv1D_block(x_inp, p=0.5, n=64):
    """
    standard dense block (Dense - BatchNorm - Dropout)
    x = input
    n = number of nodes in the dense layer
    p = dropout
    """
    seq_len = int(x_inp.shape[1])
    x = Conv1D(n, seq_len, activation='relu')(x_inp)
    x = Lambda(lambda y: K.squeeze(y, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(p)(x)
    return x

def get_task_block(name, params, x_seq):
    # set up
    p, no_nodes, seq_len = params['p'], params['no_nodes'], params['seq_len']

    if params['block_type'] == 'dense': block = dense_block
    elif params['block_type'] == 'GRU': block = GRU_block
    elif params['block_type'] == 'conv1D': block = conv1D_block

    # get the input sequence
    inp_shape = x_seq.get_shape().as_list()[1:]
    x_inp = Input(shape=inp_shape)

    # build the task block
    if CONDITIONAL[name]:
        # bool tensor for directional switch
        bool_tensor = tf.constant([-1]*no_nodes + [0]*no_nodes + [1]*no_nodes)

        # directional input
        dir_input = Input(shape=(1,), name='dir_input')
        dir_bool = Lambda(lambda d: tf.equal(K.cast(d, 'int32'), bool_tensor))(dir_input)
        dir_bool = Lambda(lambda d: K.cast(d, 'float32'),)(dir_bool)

        x = block(x_inp, p, no_nodes*3)
        x = multiply([x, dir_bool])

        pred = Dense(OUTPUT_SHAPES[name], activation=ACTIVATIONS[name], name=name)(x)
        task_block = Model(inputs=[x_inp, dir_input], outputs=[pred])
    else:

        x = block(x_inp, p, no_nodes)

        pred = Dense(OUTPUT_SHAPES[name], activation=ACTIVATIONS[name], name=name)(x)
        task_block = Model(inputs=[x_inp], outputs=[pred])

    print("Built Task Block {}".format(name))
    return task_block

def get_sequence_idcs(seq_len, dilation):
    seq_idcs = np.arange(seq_len)
    rest = seq_len%dilation
    if not rest: start_idx = dilation-1
    else: start_idx = rest - 1
    seq_idcs = seq_idcs[start_idx::dilation]
    return list(seq_idcs.astype('int32'))

def get_dilated_sequence(x, dilation):
    seq_len = int(x.shape[1])
    idcs = get_sequence_idcs(seq_len,dilation)
    x = Lambda(lambda y: tf.gather(y, idcs, axis=1))(x)
    return x

def get_time_slice(x, slice_len):
    seq_len = int(x.shape[1])
    x = Cropping1D((seq_len - slice_len,0))(x)
    return x

def get_x_sequence(x, seq_len, dilation):
    x = get_time_slice(x, seq_len)
    x = get_dilated_sequence(x, dilation)
    return x

def get_task_block_params(name):
    return np.load(PARAM_PATH + name + '_params.npy').item()

def get_full_model():
    # set up
    predictions = []
    dir_input = Input(shape=(1,), name='dir_input')

    # build lrcn with maximal sequence length of 14
    params = get_task_block_params(RELATIVE_ANGLE)
    model = vgg_to_timedistributed(14, True)
    x = model.output

    # build the task blocks
    for k in last_layer_keys:
        params = get_task_block_params(k)
        x_seq = get_x_sequence(x, params['seq_len'], params['dilation'])

        block = get_task_block(k, params, x_seq)

        # get the prediction of every task block
        if CONDITIONAL[k]:
            predictions.append(block([x_seq, dir_input]))
        else:
            predictions.append(block([x_seq]))

        del block

    # build the full model
    full_model = Model(inputs=[model.input, dir_input], outputs=predictions)

    return full_model
