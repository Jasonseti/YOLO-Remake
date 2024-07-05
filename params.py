
""" Basic params """
annot_path = '../../Jupyter Files/Datasets/PascalVOC/Annotations'
image_path = '../../Jupyter Files/Datasets/PascalVOC/JPEGImages'
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

""" Encoder params """
aspect_ratios = [0.5, 1.0, 2.0]

""" Model params """
backbone_trainable = False
image_size = (448, 448)
grids = 7
n_boxes = 3
stride = 32
lambda_conf = 1.0

iou_threshold = 0.5
score_threshold = 0.5
