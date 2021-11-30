# nuScenes dev-kit eval wrapper copied from python-sdk/nuscenes/eval/detection/evaluate.py.
# Original code written by Holger Caesar & Oscar Beijbom, 2018. Edited for internal usage.

from typing import List
from pyquaternion import Quaternion
from typing import Tuple, Dict, Any
from glob import glob
import numpy as np
from copy import deepcopy
import os

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import calc_ap, calc_tp, match_boxes, stats_from_matches
from nuscenes.eval.detection.constants import TP_METRICS, DETECTION_NAMES
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList


class DetectionEvalWrapper:
    """
    This is the un-official nuScenes detection evaluation code.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, scale, orientation.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    AP_ERRORS = 'mean_dist_aps'
    TP_ERRORS = 'label_tp_errors'
    TRANSLATION_ERROR = "trans_err"
    SCALE_ERROR = "scale_err"
    ORIENTATION_ERROR = "orient_err"

    def __init__(self,
                 gt_boxes: EvalBoxes,
                 pred_boxes: EvalBoxes,
                 verbose: bool = False):
        """
        Init method.
        :param gt_boxes: Ground Truth boxes.
        :param pred_boxes: Predicted boxes.
        :param verbose: Specify as true to print intermediate logs to stdout.
        """

        self.verbose = verbose

        # todo|note class ranges are not used. The range can be specified in the __call__ args.
        self.cfg = DetectionConfig(class_range={
            "car": 100,
            "truck": 100,
            "bus": 100,
            "trailer": 100,
            "construction_vehicle": 100,
            "pedestrian": 100,
            "motorcycle": 100,
            "bicycle": 100,
            "traffic_cone": 100,
            "barrier": 100
        },
        dist_fcn="center_distance",
        min_recall=0.1,
        min_precision=0.1,
        max_boxes_per_sample=500,
        dist_ths=[0.0],  # todo|note not used
        dist_th_tp=0.0,  # todo|note not used
        mean_ap_weight=0,  # todo|note not used
    )

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')

        self.pred_boxes = pred_boxes
        self.gt_boxes = gt_boxes

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

    def _evaluate(self, min_z: float, max_z: float, rel_dist_ths: List[float], rel_tp_dist_th: float) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :param min_z: Min allowed Z. Filters boxes whose Z value is less than this.
        :param max_z: Max allowed Z. Filter boxes whose Z value is more than this.
        :param rel_dist_ths: Relative distance thresholds needed for matching GT to predictions, and then APs are averaged.
        :param rel_tp_dist_th: Relative distance Threshold for the true positive metric.
        :return: A tuple of high-level and the raw metric data.
        """
        rel_dist_ths_ = deepcopy(rel_dist_ths)
        if rel_tp_dist_th not in rel_dist_ths:
            rel_dist_ths_.append(rel_tp_dist_th)

        # -----------------------------------
        # Step 0: Filter boxes for the specified range.
        # -----------------------------------
        gt_boxes = self._filter_boxes(self.gt_boxes, min_z=min_z, max_z=max_z, verbose=self.verbose)
        pred_boxes = self._filter_boxes(self.pred_boxes, min_z=min_z, max_z=max_z, verbose=self.verbose)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for rel_dist_th in rel_dist_ths_:
            matches = match_boxes(gt_boxes, pred_boxes, rel_dist_th=rel_dist_th)
            for class_name in self.cfg.class_names:
                md = stats_from_matches(matches, class_name)
                metric_data_list.set(class_name, rel_dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for rel_dist_th in rel_dist_ths:
                metric_data = metric_data_list[(class_name, rel_dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, rel_dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, rel_tp_dist_th)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        return metrics, metric_data_list

    def _filter_boxes(self,
                      boxes: EvalBoxes,
                      min_z: float,
                      max_z: float,
                      verbose: bool = False) -> EvalBoxes:
        """
        Applies filtering to boxes based on the Z value.
        :param boxes: An instance of the EvalBoxes class to be filtered.
        :param min_z: Min allowed Z.
        :param max_z: Max allowed Z.
        :param verbose: Whether to print to stdout.
        """
        boxes = deepcopy(boxes)
        # Accumulators for number of filtered boxes.
        total, dist_filter = 0, 0
        for ind, sample_token in enumerate(boxes.sample_tokens):
            # Filter on distance.
            total += len(boxes[sample_token])
            boxes.boxes[sample_token] = [box for box in boxes[sample_token] if
                                         max_z >= box.translation[1] >= min_z]
            dist_filter += len(boxes[sample_token])

        if verbose:
            print("=> Original number of boxes: %d" % total)
            print("=> After distance based filtering: %d" % dist_filter)

        return boxes

    def __call__(self, min_z: float, max_z: float, rel_dist_thresholds: List[float], rel_tp_dist_threshold: float) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation.
        :param min_z: Min allowed Z. Filters boxes whose Z value is less than this.
        :param max_z: Max allowed Z. Filter boxes whose Z value is more than this.
        :param rel_dist_thresholds: Relative distance thresholds needed for matching GT to predictions, and then APs are averaged.
        :param rel_tp_dist_threshold: Relative distance threshold for the true positive metric.
        :return: A dict that stores the high-level metrics and meta data.
        """

        # Run evaluation.
        metrics, metric_data_list = self._evaluate(min_z=min_z, max_z=max_z, rel_dist_ths=rel_dist_thresholds, rel_tp_dist_th=rel_tp_dist_threshold)

        metrics_summary = metrics.serialize()

        if self.verbose:
            # Print per-class metrics.
            print('Object Class\tAP\tATE\tASE\tAOE')
            class_aps = metrics_summary[self.AP_ERRORS]
            class_tps = metrics_summary[self.TP_ERRORS]
            for class_name in class_aps.keys():
                if class_name.lower() in ["car", "pedestrian"]:
                    print('%s     \t%.3f\t%.3f\t%.3f\t%.3f'
                          % (class_name, class_aps[class_name],
                             class_tps[class_name][self.TRANSLATION_ERROR],
                             class_tps[class_name][self.SCALE_ERROR],
                             class_tps[class_name][self.ORIENTATION_ERROR]))

        return metrics_summary

def read_detections(path):
    # debug code to read predictions and gt from npy files.
    filenames = glob(path + "/*.npy")
    if len(filenames) <= 0:
        raise ValueError(f"'{path}' does not exists, or does not contain .npy files.")

    gt_boxes, pred_boxes = EvalBoxes(), EvalBoxes()
    for filename in filenames:
        data: Dict[str, Any] = np.load(filename, allow_pickle=True).item()
        gt_boxes_list, pred_boxes_list = [], []
        for i in range(data["gt_boxes"].shape[0]):
            if np.all(data["gt_boxes"][i][3:6] > 0.0) and data["gt_boxes"][i][2] > 0.0 and data["gt_class"][i].lower() in DETECTION_NAMES:
                gt_boxes_list.append(DetectionBox(sample_token=str(data["index"]),
                                                  translation=(data["gt_boxes"][i][0], data["gt_boxes"][i][2], data["gt_boxes"][i][1]),
                                                  size=data["gt_boxes"][i][3:6],
                                                  rotation=Quaternion(axis=[0, 0, 1], radians=data["gt_boxes"][i][6]).q,
                                                  detection_name=data["gt_class"][i].lower()))

        for i in range(data["pred_boxes"].shape[0]):
            if np.all(data["pred_boxes"][i][3:6] > 0.0) and data["pred_boxes"][i][2] > 0.0 and data["pred_class"][i].lower() in DETECTION_NAMES:
                pred_boxes_list.append(DetectionBox(sample_token=str(data["index"]),
                                                    translation=(data["pred_boxes"][i][0], data["pred_boxes"][i][2], data["pred_boxes"][i][1]),
                                                    size=data["pred_boxes"][i][3:6],
                                                    rotation=Quaternion(axis=[0, 0, 1], radians=data["pred_boxes"][i][6]).q,
                                                    detection_name=data["pred_class"][i].lower(),
                                                    detection_score=float(data["pred_conf"][i])))

        gt_boxes.add_boxes(str(data["index"]), gt_boxes_list)
        pred_boxes.add_boxes(str(data["index"]), pred_boxes_list)

    return gt_boxes, pred_boxes


if __name__ == "__main__":
    # Try eval code.

    # todo|note specify the path which has numpy files with predictions and gt data.
    #  for details about what the .npy files should contain, see the :func:`read_detections`.
    _gt_boxes, _pred_boxes = read_detections("specify path here")

    nusc_eval = DetectionEvalWrapper(gt_boxes=_gt_boxes, pred_boxes=_pred_boxes, verbose=True)

    for _min_z, _max_z in zip([0, 20, 40, 60, 80, 0], [20, 40, 60, 80, 100, 100]):
        rel_ap_thresholds = [0.05]
        print(f"Range of prediction and detections: min_z: {_min_z}, max_z: {_max_z}")
        print(f"relative AP_thresholds: {rel_ap_thresholds}")
        metrics_summary = nusc_eval(min_z=_min_z, max_z=_max_z, rel_dist_thresholds=rel_ap_thresholds, rel_tp_dist_threshold=0.05)
