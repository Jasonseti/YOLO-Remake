import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from dataset import PascalVOC
from yolo import YOLO
import utils


""" PASCAL VOC Dataset """
annotation_files = os.listdir('../../Jupyter Files/Datasets/PascalVOC/Annotations')
train_ds = PascalVOC(annotation_files[:17000], shuffle=True, preprocessing=True)
valid_ds = PascalVOC(annotation_files[17000:], shuffle=False, preprocessing=False)


""" You Only Look Once """
YOLO = YOLO(
    backbone=None,
    head=tf.keras.models.load_model('Models/head')
)
# print(YOLO.head.summary())

# YOLO.compile(optimizer=tf.keras.optimizers.Adam(2e-4))
YOLO.load_optimizer('Models/optimizer', 1e-6)

# print(YOLO.train_step(valid_ds.__getitem__(0)))
for epoch in range(1):
    print('Epoch ' + str(epoch + 1) + ':')
    YOLO.fit(train_ds, validation_data=valid_ds)
    YOLO.save_models()


""" Visualize Results """
for _ in range(10):
    utils.visualization.visualize_yolo(train_ds, YOLO)
