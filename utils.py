import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pickle

from params import CLASS_NAMES


class utils:
    @staticmethod
    def convert_to_xywh(boxes):
        return tf.concat([
            (boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]
        ], axis=-1
        )

    @staticmethod
    def convert_to_corners(boxes):
        return tf.concat([
            boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0
        ], axis=-1
        )

    @staticmethod
    def swap_xy(boxes):
        return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)

    @staticmethod
    def normalize_bboxes(image_shape, boxes):
        image_shape = tf.cast(image_shape, tf.float32)
        return tf.stack([
            boxes[..., 0] / image_shape[1],
            boxes[..., 1] / image_shape[0],
            boxes[..., 2] / image_shape[1],
            boxes[..., 3] / image_shape[0]
        ], axis=-1
        )

    @staticmethod
    def unnormalize_bboxes(image_shape, boxes):
        image_shape = tf.cast(image_shape, tf.float32)
        return tf.stack([
            boxes[..., 0] * image_shape[1],
            boxes[..., 1] * image_shape[0],
            boxes[..., 2] * image_shape[1],
            boxes[..., 3] * image_shape[0]
        ], axis=-1
        )

    @staticmethod
    def compute_iou(boxes1, boxes2):
        boxes1_corners = utils.convert_to_corners(boxes1)
        boxes2_corners = utils.convert_to_corners(boxes2)

        lu = tf.maximum(boxes1_corners[..., :2], boxes2_corners[..., :2])
        rd = tf.minimum(boxes1_corners[..., 2:], boxes2_corners[..., 2:])
        intersection = tf.maximum(0.0, rd - lu)
        intersection_area = intersection[..., 0] * intersection[..., 1]

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        union_area = boxes1_area + boxes2_area - intersection_area

        return intersection_area / union_area


class visualization:
    @staticmethod
    def plot_visualization(image, bboxes, classes=None, scores=None, gt_boxes=None):
        plt.style.use('dark_background')
        plt.axis('off')
        plt.imshow(image.numpy().astype('uint8'))
        ax = plt.gca()
        for i, (x, y, w, h) in enumerate(bboxes):
            ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=None, edgecolor='red'))
            text = ''
            if classes is not None:
                text += CLASS_NAMES[int(classes[i])]
            if scores is not None:
                text += ' | ' + str(round(scores[i].numpy(), 2))
            ax.text(x - w / 2, y - h / 2, text, color='white', fontsize=12,
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})
        if gt_boxes is not None:
            for x, y, w, h in gt_boxes:
                ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=None, edgecolor='green'))
        plt.show()

    @staticmethod
    def visualize_datasets(datasets, index=None):
        index = index if index is not None else random.randint(0, datasets.__len__())
        image, bboxes, classes = datasets.__getitem__(index)

        visualization.plot_visualization(image, bboxes=bboxes, classes=classes)

    @staticmethod
    def visualize_yolo(dataset, YOLO, index=None):
        index = index if index is not None else random.randint(0, dataset.__len__())
        image, gt_boxes, classes = dataset.__getitem__(index)
        classes, bboxes, scores = YOLO.predict_image(image)

        visualization.plot_visualization(image, bboxes, classes, scores, gt_boxes=gt_boxes)

class preprocessing:
    @staticmethod
    def random_flip_horizontal(image, bboxes):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            image_shape = tf.cast(tf.shape(image)[:-1], tf.float32)
            bboxes = tf.stack([
                image_shape[1] - bboxes[:, 0],
                bboxes[:, 1],
                bboxes[:, 2],
                bboxes[:, 3]
            ], axis=-1
            )

        return image, bboxes


class models:
    @staticmethod
    def load_optimizer(optimizer_path, list_of_grad_vars, l_rate):
        optimizer_weights = pickle.load(open(optimizer_path, 'rb'))

        optimizer = tf.keras.optimizers.Adam(l_rate)
        for grad_vars in list_of_grad_vars:
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            optimizer.apply_gradients(zip(zero_grads, grad_vars))

        optimizer.set_weights(optimizer_weights)
        return optimizer

    @staticmethod
    def get_backbone():
        inputs = tf.keras.Input((None, None, 3))
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        outputs = tf.keras.applications.ResNet50(include_top=False)(x)

        return tf.keras.Model(inputs, outputs, name='ResNet50_Backbone')
