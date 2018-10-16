from model_functions import *
import numpy as np
import os
from PIL import Image

# the outputs of the network are normalized to a range of -1 to 1
# therefore we need to multiply it by the normalization constants
# (which where calculated on the training set)
NORM_DICT = {'center_distance': 1.6511945645500001,
             'veh_distance': 50.0,
             'relative_angle': 0.759452569632}

class CAL_network(object):
    def __init__(self):
        # get the conv_model
        front_model, _, preprocessing = get_conv_model()
        self.conv_model = front_model
        self.preprocessing = preprocessing
        self.model = load_CAL_network()

    def predict(self, im, meta):
        """
        input: transformed image, meta input(i.e. direction) as a list
        Returns the predictions dictionary
        """
        if not isinstance(meta, list):
            raise TypeError('Meta needs to be a list')

        prediction = {}
        meta = [np.array([meta[i]]) for i in range(len(meta))]

        # predict the classes and the probabilites of the validation set
        preds = self.model.predict([im] + meta, batch_size=1, verbose=0)

        # CLASSIFICATION
        classification_labels = ['red_light', 'hazard_stop', 'speed_sign']
        classes = [[False, True],[False, True],[-1, 30, 60, 90]]
        for i, k in enumerate(classification_labels):
            prediction[k] = self.labels2classes(preds[i], classes[i])

        # REGRESSION
        regression_labels = ['relative_angle', 'center_distance', 'veh_distance']
        for i, k in enumerate(regression_labels):
            prediction[k] = preds[i+len(classification_labels)][0][0]
            prediction[k] = np.clip(prediction[k],-1,1)
            # renormalization
            prediction[k] = prediction[k]*NORM_DICT[k]

        return prediction

    def preprocess_image(self, im, sequence=False):
        im = Image.fromarray(im)
        im = im.crop((0,120,800,480)).resize((222,100))

        # reshape and resize the image
        x = np.asarray(im, dtype=K.floatx())
        x = np.expand_dims(x,0)

        # preprocess image
        x = self.preprocessing(x)
        x = self.conv_model.predict(x, batch_size=1, verbose=0)

        if sequence: x = np.expand_dims(x,1)
        return x

    def labels2classes(self, prediction, c):
        # turns predicted probs to onehot idcs predictions
        # returns a tuple oft the predicted class and its probability
        # == predict classes

        # if speed sign -> take only current prediction
        if prediction.shape[1]==3:
            prediction = prediction[:,0,:]

        max_idx = np.argmax(prediction)
        predicted_class = c[max_idx]
        predicted_proba = np.max(prediction)
        return (predicted_class, predicted_proba)
