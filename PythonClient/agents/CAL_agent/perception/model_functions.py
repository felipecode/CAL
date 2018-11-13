import os, re
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv1D, TimeDistributed, LSTM, \
                         multiply, Cropping1D, GRU, CuDNNGRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.applications import vgg16
from keras import backend as K
import numpy as np
import tensorflow as tf

<<<<<<< HEAD
CAL_PATH = os.path.dirname(__file__)
PARAM_PATH = CAL_PATH + '/params/'
WEIGHTS_PATH = CAL_PATH + '/weights/'

# set up for the model rebuild
=======
DATA_HOME_DIR = os.path.dirname(__file__) + '/data/'
MODEL_PATH = DATA_HOME_DIR + '/models/'
WEIGHTS_PATH = DATA_HOME_DIR + '/results/'
>>>>>>> upstream/master
last_layer_keys = ['0_red_light', '1_hazard_stop', '2_speed_sign',\
                   '3_relative_angle', '4_center_distance', '5_veh_distance']
[RED_LIGHT, HAZARD_STOP, SPEED_SIGN, RELATIVE_ANGLE, CENTER_DISTANCE, VEH_DISTANCE] = last_layer_keys

<<<<<<< HEAD
# properties of the last layers
outputs = [2,2,4,1,1,1]
activations = ['softmax']*3 + [None]*3
cond = [False, False, False, True, True, False]

OUTPUT_SHAPES = {k:outputs[i] for i,k in enumerate(last_layer_keys)}
ACTIVATIONS = {k:activations[i] for i,k in enumerate(last_layer_keys)}
CONDITIONAL = {k:cond[i] for i,k in enumerate(last_layer_keys)}

=======
>>>>>>> upstream/master
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

<<<<<<< HEAD
=======

>>>>>>> upstream/master
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

<<<<<<< HEAD
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
=======
### parameter study
def get_task_block_params(name):
    return np.load(MODEL_PATH + name + '_params.npy').item()
    
def reload_model_from_episode(name):
    """Load a model with a defined architecture from a specific training epoch"""
    if name == 'full_model_ep_3936':
        model = get_final_model()
    else:
        params = get_task_block_params(name)
        model = get_model_master(params)

    w_name = WEIGHTS_PATH + '{}.h5'.format(name)
    model.load_weights(w_name)

    return model, name
>>>>>>> upstream/master

def conv_bn_dropout(x, p=0.1):
    x = BatchNormalization(axis=1)(x)
    x = Dropout(p)(x)
    return x

<<<<<<< HEAD
def vgg_to_timedistributed(seq_len, conv_dp):
=======
def dense_block(x, p=0.5, n=64):
    """
    standard dense block (Dense - BatchNorm - Dropout)
    x = input
    n = number of nodes in the dense layer
    p = dropout
    """
    x_out = Dense(n, activation='relu', )(x)
    x_out = BatchNormalization()(x_out)
    x_out = Dropout(p)(x_out)
    return x_out

def vgg_to_timedistributed(model_type, seq_len, conv_dp):
>>>>>>> upstream/master
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

<<<<<<< HEAD
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
=======
def LSTM_block(x, p=0.5, n=64):
    x = CuDNNLSTM(n)(x)
    x = Dropout(p)(x)
    return x

def GRU_block(x, p=0.5, n=64):
    x = CuDNNGRU(n)(x)
    x = Dropout(p)(x)
    return x

def conv1D_block(x, p=0.5, n=64):
>>>>>>> upstream/master
    """
    standard dense block (Dense - BatchNorm - Dropout)
    x = input
    n = number of nodes in the dense layer
    p = dropout
    """
<<<<<<< HEAD
    seq_len = int(x_inp.shape[1])
    x = Conv1D(n, seq_len, activation='relu')(x_inp)
=======
    seq_len = int(x.shape[1])
    x = Conv1D(n, seq_len, activation='relu')(x)
>>>>>>> upstream/master
    x = Lambda(lambda y: K.squeeze(y, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(p)(x)
    return x

<<<<<<< HEAD
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
=======
def get_model_master(params):
    # set up
    predictions = []
    p = params['p']
    no_nodes = params['no_nodes']
    seq_len = params['seq_len']
    try: dilation = params['dilation']
    except KeyError: dilation = 1
    if params['block_type'] == 'dense': block = dense_block
    elif params['block_type'] == 'GRU': block = GRU_block
    elif params['block_type'] == 'LSTM': block = LSTM_block
    elif params['block_type'] == 'conv1D': block = conv1D_block
    try: conv_dp = params['conv_dp']
    except KeyError: conv_dp = False

    # bool tensor for directional switch
    bool_tensor = tf.constant([-1]*no_nodes + [0]*no_nodes + [1]*no_nodes)

    # determine the seq_len after dilation
    dilated_seq_len = seq_len/dilation
    if seq_len%dilation!=0: dilated_seq_len +=1

    # build lrcn
    model = vgg_to_timedistributed('VGG16', dilated_seq_len, conv_dp)
    x = model.output
    if params['block_type'] == 'dense':
        x = get_last_time_slice(x, dilated_seq_len)

    ### CLASSIFICATION
    # red_light
    x0 = block(x, p, no_nodes)
    predictions.append(Dense(2, activation='softmax', name=RED_LIGHT)(x0))
    # hazard_stop
    x1 = block(x, p, no_nodes)
    predictions.append(Dense(2, activation='softmax', name=HAZARD_STOP)(x1))
    # speed_sign
    x2 = block(x, p, no_nodes)
    predictions.append(Dense(4, activation='softmax', name=SPEED_SIGN)(x2))


    ### REGRESSION
    dir_input = Input(shape=(1,), name='dir_input')
    dir_bool = Lambda(lambda d: tf.equal(K.cast(d, 'int32'), bool_tensor))(dir_input)
    dir_bool = Lambda(lambda d: K.cast(d, 'float32'),)(dir_bool)
    # relative_angle
    x3 = block(x, p, no_nodes*3)
    x3 = multiply([x3, dir_bool])
    predictions.append(Dense(1, name=RELATIVE_ANGLE)(x3))
    # center_distance
    x4 = block(x, p, no_nodes*3)
    x4 = multiply([x4, dir_bool])
    predictions.append(Dense(1, name=CENTER_DISTANCE)(x4))
    # veh_distance
    x5 = block(x, p, no_nodes)
    predictions.append(Dense(1, name=VEH_DISTANCE)(x5))

    model = Model(inputs=[model.input, dir_input], outputs=predictions)
    return model

### load from parameter study , build final models
def get_prev_layer(model, layer, list_idx=0):
    inp = layer.input
    if isinstance(inp, list): inp = inp[list_idx]
    prev_layer_name = re.findall('(.*?)/', inp.name)[0]
    return model.get_layer(prev_layer_name)

def get_task_block(model_name):
    # load the model at the specified episode
    model, name = reload_model_from_episode(model_name)
    params = get_task_block_params(model_name)

    # get all layers in the task block
    l = model.get_layer(model_name)
    layers = []

    # get all layers
    while True:
        layers.append(l)
        l = get_prev_layer(model, l)
        if isinstance(l, TimeDistributed): break

    # reverse the order
    layers = layers[::-1]

    if model_name==RELATIVE_ANGLE \
    or model_name==CENTER_DISTANCE:
        # get the directional layers
        l = model.get_layer(model_name)
        dir_layers = []
        while True:
            dir_layers.append(l)
            try:  l = get_prev_layer(model, l, list_idx=1)
            except: break  # reached dir_input layer

        # reverse the order
        dir_layers = dir_layers[::-1]

        # standard task block
        inp = Input(shape=layers[0].input_shape[1:])
        x0 = layers[0](inp)
        i = 1
        while True:
            x0 = layers[i](x0)
            if isinstance(layers[i], Dropout): break
            i += 1

        # dir input
        dir_input = Input(shape=(1,), name='dir_input')
        x1 = dir_layers[0](dir_input)
        x1 = dir_layers[1](x1)
        # multiply and output
        x = multiply([x0, x1])
        x = layers[-1](x)
        pred = x
        task_model = Model(inputs=[inp, dir_input], outputs=[pred])

    else:
        # build the model
        inp = Input(shape=layers[0].input_shape[1:])
        x = layers[0](inp)
        for l in layers[1:]:  x = l(x)
        pred = x
        task_model = Model(inputs=[inp], outputs=[pred])

    print("Built Task Block {}".format(model_name))

    return task_model, params
>>>>>>> upstream/master

def get_sequence_idcs(seq_len, dilation):
    seq_idcs = np.arange(seq_len)
    rest = seq_len%dilation
    if not rest: start_idx = dilation-1
    else: start_idx = rest - 1
    seq_idcs = seq_idcs[start_idx::dilation]
    return list(seq_idcs.astype('int32'))

def get_dilated_sequence(x, dilation):
<<<<<<< HEAD
=======
    import tensorflow as tf
>>>>>>> upstream/master
    seq_len = int(x.shape[1])
    idcs = get_sequence_idcs(seq_len,dilation)
    x = Lambda(lambda y: tf.gather(y, idcs, axis=1))(x)
    return x

<<<<<<< HEAD
=======
def get_last_time_slice(x, seq_len):
    x = Cropping1D((seq_len-1,0))(x)
    x = Lambda(lambda y: K.squeeze(y, 1))(x)
    return x

>>>>>>> upstream/master
def get_time_slice(x, slice_len):
    seq_len = int(x.shape[1])
    x = Cropping1D((seq_len - slice_len,0))(x)
    return x

def get_x_sequence(x, seq_len, dilation):
    x = get_time_slice(x, seq_len)
    x = get_dilated_sequence(x, dilation)
    return x

<<<<<<< HEAD
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
=======
def get_time_dist_model(model_name, seq_len):
    model, name = reload_model_from_episode(model_name)

    # get the layers inside the time distribution wrapper
    layers = [model.layers[0]]
    for l in model.layers[1:]:
        if not type(l)==TimeDistributed: break
        layers.append(l.layer)

    # get the new input shape
    conv_inp = layers[0].input_shape[2:]
    inp_shape = (seq_len,) + conv_inp

    # gee
    inp = Input(shape=inp_shape)
    x = TimeDistributed(layers[1], name=layers[1].name)(inp)
    for i,l in enumerate(layers[2:]):
        x  = TimeDistributed(l, name=l.name)(x)
    prediction = x
    m = Model(inputs=[inp], outputs=prediction)
    return m

def get_final_model():
    # set up
    predictions = []
    dir_input = Input(shape=(1,), name='dir_input')
    
    # load all the blocks
    b0, p0 = get_task_block(RED_LIGHT)
    b1, p1 = get_task_block(HAZARD_STOP)
    b2, p2 = get_task_block(SPEED_SIGN)
    b3, p3 = get_task_block(RELATIVE_ANGLE)
    b4, p4 = get_task_block(CENTER_DISTANCE)
    b5, p5 = get_task_block(VEH_DISTANCE)

    # build lrcn
    model = get_time_dist_model(RELATIVE_ANGLE, 14)
    x = model.output

    # get the predictions
    x0 = get_x_sequence(x, p0['seq_len'], p0['dilation'])
    predictions.append(b0([x0]))
    x1 = get_x_sequence(x, p1['seq_len'], p1['dilation'])
    predictions.append(b1([x1]))
    x2 = get_x_sequence(x, p2['seq_len'], p2['dilation'])
    predictions.append(b2([x2]))
    x3 = get_x_sequence(x, p3['seq_len'], p3['dilation'])
    predictions.append(b3([x3, dir_input]))
    x4 = get_x_sequence(x, p4['seq_len'], p4['dilation'])
    predictions.append(b4([x4, dir_input]))
    x5 = get_x_sequence(x, p5['seq_len'], p5['dilation'])
    predictions.append(b5([x5]))

    model = Model(inputs=[model.input, dir_input],
                  outputs=predictions)
    return model


>>>>>>> upstream/master
