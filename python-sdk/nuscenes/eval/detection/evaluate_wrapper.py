# nuScenes dev-kit.
# Original code written by Holger Caesar & Oscar Beijbom, 2018.
# Edited for internal usage.

from pyquaternion import Quaternion
from typing import Tuple, Dict, Any
from glob import glob
import numpy as np
from copy import deepcopy

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList

class DetectionEval:
    """
    This is the un-official nuScenes detection evaluation code.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale.
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
                 gt_boxes,
                 pred_boxes,
                 verbose: bool = True):

        self.verbose = verbose

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
        dist_ths=[0.0],  # not used
        dist_th_tp=0.0,  # not used
        mean_ap_weight=0,  # not used
    )

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')

        self.pred_boxes = pred_boxes
        self.gt_boxes = gt_boxes

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

    def _evaluate(self, min_z, max_z, dist_ths, tp_dist_th) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        dist_ths_ = deepcopy(dist_ths)
        if tp_dist_th not in dist_ths:
            dist_ths_.append(tp_dist_th)

        # -----------------------------------
        # Step 0: Filter boxes for the specified range.
        # -----------------------------------
        pred_boxes = self._filter_boxes(self.pred_boxes, min_z=min_z, max_z=max_z, verbose=False)
        gt_boxes = self._filter_boxes(self.gt_boxes, min_z=min_z, max_z=max_z, verbose=False)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in dist_ths_:
                md = accumulate(gt_boxes, pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, tp_dist_th)]
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
        Applies filtering to boxes. Distance, bike-racks and points per box.
        :param boxes: An instance of the EvalBoxes class.
        :param max_z: Maps the detection name to the eval distance threshold for that class.
        :param verbose: Whether to print to stdout.
        """
        boxes = deepcopy(boxes)
        # Accumulators for number of filtered boxes.
        total, dist_filter = 0, 0
        for ind, sample_token in enumerate(boxes.sample_tokens):
            # Filter on distance.
            total += len(boxes[sample_token])
            boxes.boxes[sample_token] = [box for box in boxes[sample_token] if
                                         max_z >= box.translation[2] >= min_z]
            dist_filter += len(boxes[sample_token])

        if verbose:
            print("=> Original number of boxes: %d" % total)
            print("=> After distance based filtering: %d" % dist_filter)

        return boxes

    def __call__(self, min_z, max_z, dist_thresholds, tp_dist_threshold, verbose=False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation.
        :return: A dict that stores the high-level metrics and meta data.
        """

        # Run evaluation.
        metrics, metric_data_list = self._evaluate(min_z=min_z, max_z=max_z, dist_ths=dist_thresholds, tp_dist_th=tp_dist_threshold)

        metrics_summary = metrics.serialize()

        if verbose:
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
    gt_boxes = EvalBoxes()
    pred_boxes = EvalBoxes()
    for filename in filenames:
        data = np.load(filename, allow_pickle=True).item()
        gt_boxes_list, pred_boxes_list  = [], []
        for i in range(data["gt_boxes"].shape[0]):
            if data["gt_boxes"][i][3] > 0.0 and data["gt_class"][i].lower() in ["car", "pedestrian"] and data["gt_boxes"][i][2] > 0.0:
                gt_boxes_list.append(DetectionBox(sample_token=str(data["index"]),
                                                  translation=data["gt_boxes"][i][:3],
                                                  size=data["gt_boxes"][i][3:6],
                                                  rotation=Quaternion(axis=[0, 0, 1], angle=data["gt_boxes"][i][6]).elements,
                                                  detection_name=data["gt_class"][i].lower()))

        for i in range(data["pred_boxes"].shape[0]):
            if data["pred_class"][i].lower() in ["car", "pedestrian"] and data["pred_boxes"][i][2] > 0.0:
                pred_boxes_list.append(DetectionBox(sample_token=str(data["index"]),
                                                    translation=data["pred_boxes"][i][:3],
                                                    size=data["pred_boxes"][i][3:6],
                                                    rotation=Quaternion(axis=[0, 0, 1], angle=data["pred_boxes"][i][6]).elements,
                                                    detection_name=data["pred_class"][i].lower(),
                                                    detection_score=float(data["pred_conf"][i])))

        gt_boxes.add_boxes(str(data["index"]), gt_boxes_list)
        pred_boxes.add_boxes(str(data["index"]), pred_boxes_list)

    return gt_boxes, pred_boxes

if __name__ == "__main__":
    # Test eval code.
    gt_boxes, pred_boxes = read_detections("/home/alok/yolo-export-pred")
    nusc_eval = DetectionEval(gt_boxes=gt_boxes, pred_boxes=pred_boxes)
    for min_z, max_z in zip([0, 20, 40, 60, 80, 0], [20, 40, 60, 80, 100, 100]):
        ap_thresholds = list(np.linspace(0.50, max_z*0.05, num=4))
        print("AP_thresholds: ", ap_thresholds)
        print(f"min_z: {min_z}, max_z: {max_z}")
        metrics_summary = nusc_eval(min_z=min_z, max_z=max_z, dist_thresholds=ap_thresholds, tp_dist_threshold=4.0, verbose=True)
