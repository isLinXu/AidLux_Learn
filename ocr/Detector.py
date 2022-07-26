import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class DetectorDecoder:
    def __init__(self, thresh=0.2, box_thresh=0.3, max_candidates=1000, unclip_ratio=1.5, min_box_size=3):
        self.min_size = min_box_size
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, height, width):
        segmentation = self.binarize(pred)
        boxes, scores = self.boxes_from_bitmap(pred,segmentation, width, height)

        if len(boxes) > 0:
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
            boxes, scores = boxes[idx], boxes[idx]
        else:
            boxes, scores = [], []

        save_index = []
        for box_id, box in enumerate(boxes):
            #if box[0][1] > 480:
            save_index.append(box_id)
        boxes_list = np.take(boxes, save_index, axis=0)
        return boxes_list

    def binarize(self, pred):
        return pred > self.thresh

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        assert len(bitmap.shape) == 2
        height, width = bitmap.shape
        label_img = (bitmap * 255).astype(np.uint8)
        tmpTe = cv2.findContours(label_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(tmpTe) == 3:  # opencv版本兼容性
            _, contours, hierarchy = tmpTe
        elif len(tmpTe) == 2:  # opencv版本兼容性
            contours, hierarchy = tmpTe

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)

            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)

        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        # bo = cv2.boxPoints(bounding_box)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        # 计算轮廓所包含的面积
        th_ss = cv2.contourArea(contour)
        # 计算轮廓的周长
        # th_ss = cv2.arcLength(contour, True)

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        # return box, min(bounding_box[1])
        return box, th_ss

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
