import numpy as np
from numba import jit
from scipy.spatial.distance import cdist, euclidean

@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    wh = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
    return (wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh))


def centroid_distance(det, tck):
    return euclidean(det, tck)


class Metric(object):
    """
    Defining metric with different options for metrics defined in metrics_dict.
    The matching threshold for each metric is defined in threshold_dict.
    Threshold_dict values should be negative if large value is better metric for the metric
    and positive otherwise.
    """

    metrics_dict = {"iou": iou, "centroids": centroid_distance}
    threshold_dict = {"iou": -0.3, "centroids": 50}

    def __init__(self, metric_str):
        self.metric = Metric.metrics_dict[metric_str]
        self.threshold = Metric.threshold_dict[metric_str]

    def distance(self, detections, trackers):

        return cdist(detections, trackers, self.metric)





