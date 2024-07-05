import tensorflow as tf
from keras import layers
import numpy as np
import pickle

from encoder import LabelEncoder
import params
import utils


class YOLO(tf.keras.Model):
    """ A model to detect objects and its location within an image.
            Input : Image, constant shape, shape=(h, w, c)
            Output : [a grid matrix of class scores, shape=(grids * grids, n_classes),
                      a grid matrix of bboxes shifts, shape=(grids * grids * n_boxes, 4),
                      a grid matrix of box confidence scores=(grids * grids * n_boxes, 1)
    """

    def __init__(self, backbone=None, head=None):
        super(YOLO, self).__init__()
        self.encoder = LabelEncoder()
        self._image_size = params.image_size
        self._grids = params.grids
        self._n_boxes = params.n_boxes
        self._stride = params.stride
        self._feature_shape = tuple(np.int32(np.array(self._image_size) / self._stride))

        self._lambda_conf = params.lambda_conf
        self._iou_threshold = params.iou_threshold
        self._score_threshold = params.score_threshold

        self.backbone = backbone if backbone else utils.models.get_backbone()
        self.backbone.trainable = params.backbone_trainable
        self.head = head if head else self.get_head()

        self.obj_loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        self.box_loss_fn = tf.keras.losses.Huber(reduction='none')
        self.conf_loss_fn = tf.keras.losses.Huber(reduction='none')
        self.obj_loss_metric = tf.keras.metrics.Mean(name='obj_loss_metric')
        self.box_loss_metric = tf.keras.metrics.Mean(name='box_loss_metric')
        self.conf_loss_metric = tf.keras.metrics.Mean(name='conf_loss_metric')

    def get_head(self):
        inputs = tf.keras.Input(self._feature_shape + (self.backbone.output_shape[-1],))
        x = layers.Conv2D(512, kernel_size=2, strides=2)(inputs)

        for size in []:
            x = layers.LeakyReLU(0.1)(x)
            x = layers.Conv2D(size, kernel_size=1, strides=1)(x)

        x = layers.Flatten()(x)
        for size in [512, 1024]:
            x = layers.LeakyReLU(0.1)(x)
            x = layers.Dense(size)(x)

        cls_output = layers.Dense(len(params.CLASS_NAMES) * self._grids ** 2, activation='softmax')(layers.Dropout(0.2)(x))
        box_output = layers.Dense(4 * self._n_boxes * self._grids ** 2)(x)
        conf_output = layers.Dense(1 * self._n_boxes * self._grids ** 2)(x)

        return tf.keras.Model(inputs, [cls_output, box_output, conf_output], name='YOLO_Head')

    def get_predictions(self, x, training=False):
        x = tf.image.resize(x, self._image_size)
        x = tf.expand_dims(x, 0)
        features = self.backbone(x, training=training)
        predictions = self.head(features, training=training)

        pred_classes = tf.reshape(predictions[0][0], [-1, len(params.CLASS_NAMES)])
        pred_boxes = tf.reshape(predictions[1][0], [-1, self._n_boxes, 4])
        pred_conf = tf.reshape(predictions[2][0], [-1, self._n_boxes, 1])
        return pred_classes, pred_boxes, pred_conf

    def loss_fn(self, pred_classes, pred_boxes, pred_conf, matched_indices, class_targets, box_targets, gt_boxes):
        # Decide which box to use
        pred_boxes = tf.gather(pred_boxes, matched_indices)
        iou_scores = utils.utils.compute_iou(tf.expand_dims(gt_boxes, 1), pred_boxes)
        matched_boxes_id = tf.argmax(iou_scores, axis=-1, output_type=tf.int32)
        matched_boxes_id = tf.stack([tf.range(tf.size(matched_indices)), matched_boxes_id], axis=-1)

        # Calculate Loss
        obj_loss = self.obj_loss_fn(class_targets, tf.gather(pred_classes, matched_indices))
        box_loss = self.box_loss_fn(box_targets, tf.gather_nd(pred_boxes, matched_boxes_id))
        conf_scores = tf.expand_dims(tf.reduce_max(iou_scores, axis=-1), 0)
        conf_loss = self.conf_loss_fn(conf_scores, tf.gather_nd(pred_conf, matched_boxes_id))

        return tf.reduce_mean(obj_loss), tf.reduce_mean(box_loss), tf.reduce_mean(conf_loss) * self._lambda_conf


    def train_step(self, training_data):
        image, bboxes, classes = training_data
        image, bboxes = tf.image.resize(image, self._image_size), utils.utils.normalize_bboxes(tf.shape(image), bboxes)
        matched_indices, class_targets, box_targets = self.encoder.encode_labels(tf.shape(image), bboxes, classes)

        with tf.GradientTape(persistent=True) as tape:
            pred_classes, pred_boxes, pred_conf = self.get_predictions(image, training=True)

            obj_loss, box_loss, conf_loss = self.loss_fn(
                pred_classes, pred_boxes, pred_conf,
                matched_indices, class_targets, box_targets,
                bboxes
            )
            total_loss = obj_loss + box_loss + conf_loss

        if params.backbone_trainable:
            self.optimizer.minimize(total_loss, self.backbone.trainable_variables, tape=tape)
        self.optimizer.minimize(total_loss, self.head.trainable_variables, tape=tape)

        self.obj_loss_metric.update_state(obj_loss)
        self.box_loss_metric.update_state(box_loss)
        self.conf_loss_metric.update_state(conf_loss)
        return {'class': self.obj_loss_metric.result(),
                'box': self.box_loss_metric.result(),
                'confidence': self.conf_loss_metric.result()}

    def test_step(self, testing_data):
        image, bboxes, classes = testing_data
        image, bboxes = tf.image.resize(image, self._image_size), utils.utils.normalize_bboxes(tf.shape(image), bboxes)
        matched_indices, class_targets, box_targets = self.encoder.encode_labels(tf.shape(image), bboxes, classes)

        pred_classes, pred_boxes, pred_conf = self.get_predictions(image, training=False)
        obj_loss, box_loss, conf_loss = self.loss_fn(
            pred_classes, pred_boxes, pred_conf,
            matched_indices, class_targets, box_targets,
            bboxes
        )

        self.obj_loss_metric.update_state(obj_loss)
        self.box_loss_metric.update_state(box_loss)
        self.conf_loss_metric.update_state(conf_loss)
        return {'class': self.obj_loss_metric.result(),
                'box': self.box_loss_metric.result(),
                'confidence': self.conf_loss_metric.result()}


    def save_models(self):
        if params.backbone_trainable:
            self.backbone.save('Models/backbone')
        self.head.save('Models/head')
        pickle.dump(self.optimizer.get_weights(), open('Models/optimizer', 'wb'))

    def load_optimizer(self, optimizer_path, l_rate):
        optimizer = utils.models.load_optimizer(
            optimizer_path, [self.backbone.trainable_variables, self.head.trainable_variables], l_rate
        )
        self.compile(optimizer=optimizer)

    def predict_image(self, image):
        pred_classes, pred_bboxes, pred_conf = self.get_predictions(image)
        pred_bboxes = self.encoder.decode_boxes(pred_bboxes)
        classes = tf.argmax(pred_classes, axis=-1)
        bboxes = utils.utils.unnormalize_bboxes(tf.shape(image), pred_bboxes)
        cls_scores = tf.reduce_max(pred_classes, axis=-1)

        # Tile and Flatten
        classes = tf.reshape(tf.tile(tf.expand_dims(classes, 1), [1, self._n_boxes]), [-1])
        bboxes = tf.reshape(bboxes, [-1, 4])
        cls_scores = tf.tile(tf.expand_dims(cls_scores, 1), [1, self._n_boxes])
        scores = tf.reshape(cls_scores * tf.squeeze(pred_conf), [-1])

        # Score Masking
        positive_mask = tf.greater_equal(scores, self._score_threshold)
        classes = tf.boolean_mask(classes, positive_mask)
        bboxes = tf.boolean_mask(bboxes, positive_mask)
        scores = tf.boolean_mask(scores, positive_mask)

        return classes, bboxes, scores
