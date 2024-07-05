import tensorflow as tf
import numpy as np
import xmltodict
import random
import os

import utils
import params


class PascalVOC(tf.keras.utils.Sequence):
    """ A class to yield PascalVoc data with format,
            Image: shape=(h, w, c)
            Bbox : un-normalized bbox coordinates, (xmin, ymin, xmax, ymax)
            Class: a sparse list
    """

    def __init__(self, annot_files, limit=None, shuffle=True, preprocessing=True):
        self.annot_files = annot_files
        self.annot_path = params.annot_path
        self.image_path = params.image_path

        self.is_shuffle = shuffle
        self.is_preprocessing = preprocessing
        self.limit = limit
        if self.is_shuffle:
            random.shuffle(self.annot_files)

    def __len__(self):
        return self.limit if self.limit else len(self.annot_files)

    def on_epoch_end(self):
        if self.is_shuffle:
            random.shuffle(self.annot_files)

    def __getitem__(self, i):
        image, bboxes, classes = None, [], []

        # Open Annotation File
        annot_file = self.annot_files[i]
        annotation = open(os.path.join(self.annot_path, annot_file), 'r', encoding='utf-8').read()
        annotation = xmltodict.parse(annotation)['annotation']

        # Get Image
        image_file = annotation['filename']
        image = tf.io.decode_jpeg(tf.io.read_file(os.path.join(self.image_path, image_file)))

        # Get bboxes and classes from annotation
        for stuff in annotation['object'] if isinstance(annotation['object'], list) else [annotation['object']]:
            sorted_bbox = [stuff['bndbox'][key] for key in ['xmin', 'ymin', 'xmax', 'ymax']]
            bboxes.append(utils.utils.convert_to_xywh(np.float32(sorted_bbox)))
            classes.append(params.CLASS_NAMES.index(stuff['name']))

        # Preprocessing
        image, bboxes, classes = tf.cast(image, tf.float32), tf.cast(bboxes, tf.float32), tf.cast(classes, tf.int32)
        if self.is_preprocessing:
            image, bboxes = utils.preprocessing.random_flip_horizontal(image, bboxes)

        return image, bboxes, classes
