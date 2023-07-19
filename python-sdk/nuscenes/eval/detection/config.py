# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

import json
import os

import fsspec
from nuscenes.eval.detection.data_classes import DetectionConfig


def config_factory(configuration_name: str) -> DetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    Note that this only works if the config file is located in the nuscenes/eval/detection/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """

    # Check if config exists.
    fs = fsspec.filesystem("gcs")
    cfg_path = f"gs://reco-ds/nuscenes/nuscenes-devkit/eval/detection/configs/{configuration_name}.json"
    assert fs.exists(cfg_path), \
        'Requested unknown configuration {}'.format(configuration_name)

    # Load config file and deserialize it.
    with fs.open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = DetectionConfig.deserialize(data)

    return cfg
