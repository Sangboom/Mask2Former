# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.data import MetadataCatalog, DatasetCatalog

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/armbench-object-only/instance-segmentation/maskformer2_R50_bs2_1x_800.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # parser.add_argument("--video-input", help="Path to video file.")
    # parser.add_argument(
    #     "--input",
    #     nargs="+",
    #     help="A list of space separated input images; "
    #     "or a single glob pattern such as 'directory/*.jpg'",
    # )
    # parser.add_argument(
    #     "--output",
    #     help="A file or directory to save output visualizations. "
    #     "If not given, will show output in an OpenCV window.",
    # )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    register_coco_instances('armbench_test_class1', {}, 'datasets/armbench/mix-object-tote/test.json', 'datasets/armbench/mix-object-tote/images')
    dataset_name = 'armbench_test_class1'

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    config_file_path = "configs/armbench-object-only/instance-segmentation/maskformer2_R50_bs2_1x_800.yaml"
    out_dir = config_file_path.replace('configs', 'output').replace('.yaml', '/vis_results_box_only')


    for idx, d in enumerate(tqdm(dataset_dicts)):
    
        img = cv2.imread(d["file_name"])
        # out_dir_per_file = out_dir + '/' + d['file_name'].split('/')[-1].replace('.jpg', '')
        # if not os.path.exists(out_dir_per_file):
        #     os.makedirs(out_dir_per_file)

        min_size = cfg.INPUT.MIN_SIZE_TEST # size of the smallest size of the image
        max_size = cfg.INPUT.MAX_SIZE_TEST # max size of the side of the image
        tfm = ResizeShortestEdge(min_size, max_size).get_transform(img)
        resized = tfm.apply_image(img)
        predictions, vis_output = demo.run_on_image(img)
        vis_output = vis_output.get_image()[:, :, ::-1]
        # cv2.imwrite(out_dir_per_file + '/{}.png'.format(idx), vis_output)
        cv2.imwrite(out_dir + '/{}.png'.format(d['file_name'].split('/')[-1].replace('.jpg', '')), vis_output)