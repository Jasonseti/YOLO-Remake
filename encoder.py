import tensorflow as tf

import params


class LabelEncoder:
    """ A class to encode YOLO labels """

    def __init__(self):
        self._grids = params.grids
        self.anchor_boxes = self.get_anchor_boxes()

    def get_anchor_boxes(self):
        """ Returns normalized anchor boxes based on grid size, shape=(grids * grids, 4) """

        rx = (tf.range(self._grids, dtype=tf.float32) + 0.5) / self._grids
        ry = (tf.range(self._grids, dtype=tf.float32) + 0.5) / self._grids
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1)

        grid_size = tf.cast(1 / self._grids, tf.float32)
        anchor_boxes = tf.concat([centers, tf.tile([[[grid_size, grid_size]]], [self._grids, self._grids, 1])], axis=-1)

        return tf.reshape(anchor_boxes, [-1, 4])

    def encode_labels(self, image_shape, gt_boxes, classes):
        """ Encode bboxes and classes as class targets and box targets
                Input : [resized image, shape=(h, w, c),
                         normalized bboxes, shape=(num_obj, 4), (center_x, center_y, width, height),
                         classes, shape=(1,), sparse]
                Output: [matched_indices, which grids a box belongs to, shape=(num_gt_boxes,)
                         class_target, shape=(num_classes)
                         box_target, box shift target, ranging between 0 and 1, shape=(num_gt_boxes, 4)]
        """

        matched_indices = tf.cast(
            tf.floor(gt_boxes[..., 0] * self._grids) + tf.floor(gt_boxes[..., 1] * self._grids) * self._grids
            , tf.int32
        )

        # Class Target
        class_targets = tf.one_hot(classes, len(params.CLASS_NAMES))
        # Box Target
        matched_boxes = tf.gather(self.anchor_boxes, matched_indices)
        box_targets = tf.concat([
                (gt_boxes[..., :2] - matched_boxes[..., :2]) * self._grids,
                gt_boxes[..., 2:]
            ], axis=-1
        )

        return matched_indices, class_targets, box_targets

    def decode_boxes(self, bboxes):
        anchor_boxes = tf.tile(tf.expand_dims(self.anchor_boxes, 1), [1, params.n_boxes, 1])
        return tf.concat([
                bboxes[..., :2] / self._grids + anchor_boxes[..., :2],
                bboxes[..., 2:]
            ], axis=-1
        )
