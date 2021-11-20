import tensorflow as tf

from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

from model.Model import CDCN
from model.Loss import Accuracy, DepthLoss


def trainModel(train, y_train=None, valid=None, ds_name="", theta=0.7, epochs=3, input_shape=(256, 256, 3), steps_per_epoch=None, use_nn=True):
    model = CDCN(theta, input_shape=input_shape)

    if use_nn:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[BinaryAccuracy(), Precision(), Recall()])
    else:
        model.compile(loss=DepthLoss(),
                      optimizer='adam',
                      metrics=[Accuracy])

    # Train model
    if y_train:
        history = model.fit(train, y_train, epochs=epochs,
                            validation_data=valid, steps_per_epoch=steps_per_epoch)
    else:
        history = model.fit(
            train, epochs=epochs, validation_data=valid, steps_per_epoch=steps_per_epoch)

    model.save('trained_models/model_'+ds_name)
    return model, history


def benchmarkModel(model, valid, y_valid=None):
    # Evalution Using Testing Set
    if y_valid:
        scores = model.evaluate(valid, y_valid, verbose=1)
    else:
        scores = model.evaluate(valid, verbose=1)
    return scores
