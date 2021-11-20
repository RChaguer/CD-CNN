
import tensorflow as tf
import pandas as pd
import os
import cv2
from tensorflow.keras.utils import image_dataset_from_directory
from data.Utils import Normalization, Normalization_MSU, getX, getY, prepareAllVideosV2


class NUAADataLoader():
    def __init__(self, source_path):
        self.data_dirs = {'raw': source_path+'/raw', 'normalized': source_path +
                          '/normalized', 'detected': source_path + '/detected'}

        self.train = {}
        self.valid = {}
        self.x = {}
        self.y = {}

    def loadData(self, img_height=256, img_width=256, batch_size=8, seed=123):
        for ds, ds_path in self.data_dirs.items():
            self.train[ds] = image_dataset_from_directory(
                ds_path,
                validation_split=0.2,
                subset="training",
                labels="inferred",
                label_mode="binary",
                image_size=(img_height, img_width),
                seed=seed,
                batch_size=batch_size)

            self.valid[ds] = image_dataset_from_directory(
                ds_path,
                validation_split=0.2,
                subset="validation",
                labels="inferred",
                label_mode="binary",
                image_size=(img_height, img_width),
                seed=123,
                batch_size=batch_size)

    def normalizeData(self):
        for ds, _ in self.data_dirs.items():
            self.train[ds] = self.train[ds].map(
                Normalization, num_parallel_calls=tf.data.AUTOTUNE)
            self.valid[ds] = self.valid[ds].map(
                Normalization, num_parallel_calls=tf.data.AUTOTUNE)

    def splitValidationXY(self):
        for ds, ds_path in self.data_dirs.items():
            self.x[ds] = self.valid[ds].map(
                getX, num_parallel_calls=tf.data.AUTOTUNE)
            self.y[ds] = self.valid[ds].map(
                getY, num_parallel_calls=tf.data.AUTOTUNE)

    def getTrainSet(self, label):
        return self.train[label]

    def getValidSet(self, label):
        return self.valid[label]

    def getSplittedValidXY(self, label):
        return self.x[label], self.y[label]


class MSUDataLoader():
    def __init__(self, source_path):
        self.real_vid_loc = source_path+"/scene01/real/"
        self.attack_vid_loc = source_path+"/scene01/attack/"
        self.train_txt_file = source_path+"/train_sub_list.txt"
        self.test_txt_file = source_path+"/test_sub_list.txt"

    def preLoadData(self):
        ext = '.mp4'
        with open("train_combined.txt", "w") as out:
            with open(self.train_txt_file, "r") as file:
                for line in file:
                    line = line.rstrip()
                    for attack in ['android_SD_', 'laptop_SD_']:
                        for typ in ['ipad_video_', 'iphone_video_', 'printed_photo_']:
                            for id, rf in enumerate(['attack', 'real']):
                                if attack == 'laptop_SD_':
                                    ext = '.mov'
                                if rf == 'attack':
                                    string = self.attack_vid_loc+rf+"_client0"+line+"_"+attack+typ+"scene01"+ext
                                else:
                                    string = self.real_vid_loc+rf+"_client0"+line+"_"+attack+"scene01"+ext
                                if os.path.exists(string):
                                    # -----> format: folder/image_name
                                    out.write(string+","+str(id)+"\n")
                                ext = '.mp4'

        self.train_df = pd.read_csv(r'train_combined.txt', header=None)
        self.train_df.columns = ["video", "label"]
        self.train_df.to_csv(r"train_combined.csv", index=None)

        with open("test_combined.txt", "w") as out:
            with open(self.test_txt_file, "r") as file:
                for line in file:
                    line = line.rstrip()
                    for attack in ['android_SD_', 'laptop_SD_']:
                        for typ in ['ipad_video_', 'iphone_video_', 'printed_photo_']:
                            for id, rf in enumerate(['attack', 'real']):
                                if attack == 'laptop_SD_':
                                    ext = '.mov'
                                if rf == 'attack':
                                    string = self.attack_vid_loc+rf+"_client0"+line+"_"+attack+typ+"scene01"+ext
                                else:
                                    string = self.real_vid_loc+rf+"_client0"+line+"_"+attack+"scene01"+ext
                                if os.path.exists(string):
                                    # -----> format: folder/image_name
                                    out.write(string+","+str(id)+"\n")
                                ext = '.mp4'

        self.test_df = pd.read_csv(r'test_combined.txt', header=None)
        self.test_df.columns = ["video", "label"]
        self.test_df.to_csv(r"test_combined.csv", index=None)

    def dataLoad(self, train_frames=30, test_frames=1):
        self.x_train, self.y_train = prepareAllVideosV2(
            self.train_df, train_frames)
        self.x_test, self.y_test = prepareAllVideosV2(
            self.test_df, test_frames)

    def normalizeData(self):
        self.x_train = list(map(Normalization_MSU, self.x_train))
        self.x_test = list(map(Normalization_MSU, self.x_test))

    def getTrainSet(self):
        return self.x_train, self.y_train

    def getValidSet(self):
        return self.x_test, self.y_test
