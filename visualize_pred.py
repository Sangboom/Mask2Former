import torch
import sys
# sys.path.insert(0, "Mask2Former")
import tempfile
from pathlib import Path
import numpy as np
import cv2
import random
import glob
import argparse
# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools.coco import COCO
import os
from tqdm import tqdm
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.data.datasets import register_coco_instances
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        # visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        # return predictions, vis_output, initial_vis_output, error_vis_output
        return predictions, vis_output


# dataset_name = 'coco_2017_val'
# dataset_name = 'cityscapes_fine_instance_seg_val'
register_coco_instances('armbench_test_class1', {}, 'datasets/armbench/mix-object-tote/test.json', 'datasets/armbench/mix-object-tote/images')
dataset_name = 'armbench_test_class1'
config_file_path = "configs/armbench-object-only/instance-segmentation/maskformer2_R50_bs2_1x_800.yaml"


out_dir = config_file_path.replace('configs', 'output').replace('.yaml', '/vis_results_box_only')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(config_file_path)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = config_file_path.replace('configs', 'output').replace('.yaml', '/model_final.pth')
demo = VisualizationDemo(cfg)
metadata = MetadataCatalog.get(dataset_name)
dataset_dicts = DatasetCatalog.get(dataset_name)

for idx, d in enumerate(tqdm(dataset_dicts)):
    
    img = cv2.imread(d["file_name"])
    # out_dir_per_file = out_dir + '/' + d['file_name'].split('/')[-1].replace('.jpg', '')
    # if not os.path.exists(out_dir_per_file):
    #     os.makedirs(out_dir_per_file)

    min_size = cfg.INPUT.MIN_SIZE_TEST # size of the smallest size of the image
    max_size = cfg.INPUT.MAX_SIZE_TEST # max size of the side of the image
    tfm = ResizeShortestEdge(min_size, max_size).get_transform(img)
    resized = tfm.apply_image(img)

    predictions, vis_output = demo.run_on_image(resized)
    vis_output = vis_output.get_image()[:, :, ::-1]
    # cv2.imwrite(out_dir_per_file + '/{}.png'.format(idx), vis_output)
    cv2.imwrite(out_dir + '/{}.png'.format(d['file_name'].split('/')[-1].replace('.jpg', '')), vis_output)
    