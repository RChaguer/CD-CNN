import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np

from random import randint


def getDepthMap(label, label_weight=1):
    return tf.map_fn(lambda l: tf.ones((32, 32), dtype=tf.float32) * label_weight if l == 1 else tf.ones((32, 32), dtype=tf.float32) * (1 - label_weight), label)


def Normalization(image, label, use_nn=True):
    image = tf.cast(image/255., tf.float32)
    if not use_nn:
        label = getDepthMap(label)
    return image, label


def Normalization_MSU(x):
    return x/255.


def getX(image, label):
    return image


def getY(image, label):
    return label


IMG_SIZE = 256
MAX_FRAMES = 20


def cropCenterSquare(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def loadVideo(path, max_frames=MAX_FRAMES, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cropCenterSquare(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def prepareAllVideos(df, max_frames=MAX_FRAMES):
    X = list()
    Y = list()
    for index, row in df.iterrows():
        frames = loadVideo(row['video'], max_frames=max_frames)
        X.append(frames)
        Y.append(row['label'])
    return X, Y


def prepareAllVideosV2(df, max_frames=MAX_FRAMES):
    X = list()
    Y = list()
    for index, row in df.iterrows():
        frames = loadVideo(row['video'], max_frames=max_frames)
        for x in frames:
            X.append(x)
            Y.append(row['label'])
    return np.array(X), np.array(Y)


def pickInputImages(X, Y, nb_images_per_video=3):
    new_df = pd.DataFrame(columns=['image', 'label'])

    for i, frames in enumerate(X):
        print(frames.shape)
        n = frames.shape[0]
        for _ in range(nb_images_per_video):
            new_row = {'image': frames[randint(
                0, n-1), :, :, :], 'label': Y[i]}
            new_df = new_df.append(new_row, ignore_index=True)
    return new_df
