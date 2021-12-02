# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.
import copy
from typing import Callable, List, Optional

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData, BoxMatch


def match_boxes(
        gt_boxes: EvalBoxes,
        pred_boxes: EvalBoxes,
        dist_fcn: Callable,
        rel_dist_th: Optional[float] = None,
        dist_th: Optional[float] = None,
) -> List[BoxMatch]:
    """
    Matches prediction and GT boxes based on their distances.

    :param gt_boxes: The GT boxes from the dataset
    :param pred_boxes: The predicted boxes from the ego vehicle
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param rel_dist_th: Relative distance threshold based on GT box depth for a match. A box is a match if it fulfills
        either the relative or absolute distance threshold.
    :param dist_th: Distance threshold based on GT box depth for a match. A box is a match if it fulfills
        either the relative or absolute distance threshold.
    :return: Matched boxes. These are matched based on both class name and distance.
    """
    assert (rel_dist_th is not None) or (dist_th is not None), "Specify exactly at least one of rel_dist_th or dist_th"
    # Organize the predictions in a single list.
    pred_confs = [box.detection_score for box in pred_boxes.all]

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    matched_boxes = []
    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes.all[ind]
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == pred_box.detection_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        def is_match(proposal_dist, dist_to_ego):
            match = False
            if rel_dist_th is not None:
                # Just using y distance, similar to the distance metrics
                match = match or proposal_dist < (dist_to_ego * rel_dist_th)
            if dist_th is not None:
                match = match or proposal_dist < dist_th
            return match

        if match_gt_idx is not None and is_match(min_dist, gt_boxes[pred_box.sample_token][match_gt_idx].translation[1]):
            taken.add((pred_box.sample_token, match_gt_idx))
            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            matched_boxes.append(BoxMatch(gt_box_match, pred_box))
        else:
            matched_boxes.append(BoxMatch(None, pred_box))

    # Populate with false negatives (missed detections)
    for seq_key, sample_box_list in gt_boxes.boxes.items():
        for gt_idx, gt_box in enumerate(sample_box_list):
            if (gt_box.sample_token, gt_idx) not in taken:
                matched_boxes.append(BoxMatch(gt_box, None))
    return matched_boxes


def stats_from_matches(
        matched_boxes: List[BoxMatch],
        class_name: str,
        verbose: Optional[bool] = False
) -> DetectionMetricData:
    """
    Computes the same stats as accumulate, but from matched boxes.

    :param matched_boxes:  The pre-matched boxes, generally from match_boxes
    :param class_name: Class to compute AP on.
    :return: metrics. The raw data for a number of metrics.
    """
    # Count the positives.
    class_boxes = [box for box in matched_boxes if box.detection_name == class_name]
    npos = sum([box.has_gt for box in class_boxes])
    if verbose:
        print("Found {} GT of class {} out of {}.".
              format(npos, class_name, len(matched_boxes)))
    class_pred_boxes = [box for box in matched_boxes if box.has_pred]
    matched_tps = [box for box in class_pred_boxes if box.tp]

    if len(matched_tps) == 0:
        # If there are no TP detections at all, we return the corresponding array.
        return DetectionMetricData.no_predictions()

    period = np.pi if class_name == 'barrier' else 2 * np.pi

    match_data = {'trans_err': [center_distance(match.gt, match.pred) for match in matched_tps],
                  'vel_err': [velocity_l2(match.gt, match.pred) for match in matched_tps],
                  'scale_err': [1 - scale_iou(match.gt, match.pred) for match in matched_tps],
                  'orient_err': [yaw_diff(match.gt, match.pred, period=period) for match in matched_tps],
                  'attr_err': [1 - attr_acc(match.gt, match.pred) for match in matched_tps],
                  'conf': [box.detection_score for box in matched_tps]}

    tp = np.cumsum([box.tp for box in class_pred_boxes]).astype(float)
    fp = np.cumsum([box.fp for box in class_pred_boxes]).astype(float)
    conf = np.array([box.detection_score for box in class_pred_boxes])

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               rel_dist_th: float= None,
               dist_th: float = None,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param rel_dist_th: Relative distance threshold based on GT box depth for a match. Specify either this or rel_dist_th
    :param dist_th: Distance threshold based on GT box depth for a match. Specify either this or dist_th.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    assert ~((rel_dist_th is not None) and (dist_fcn is not None))
    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to relative/absolute threshold we have a match!
        if rel_dist_th is not None:
            is_match = (match_gt_idx is not None) and (min_dist < (gt_boxes[pred_box.sample_token][match_gt_idx].translation[1] * rel_dist_th) or min_dist < 0.25)
            if verbose:
                if match_gt_idx and not is_match:
                    print(f"Was not a match because distance is {min_dist} and threshold is {gt_boxes[pred_box.sample_token][match_gt_idx].translation[1] * rel_dist_th}")
                if is_match:
                    print(f"Is a match because distance is {min_dist} and threshold is {gt_boxes[pred_box.sample_token][match_gt_idx].translation[1] * rel_dist_th}")
        elif dist_th is not None:
            is_match = min_dist < dist_th
        else:
            raise ValueError("Specify either 'rel_dist_th' or 'dist_th'.")

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.
