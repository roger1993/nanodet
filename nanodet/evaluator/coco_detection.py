# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import copy
import io
import itertools
import json
import logging
import os
import warnings

import numpy as np
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

logger = logging.getLogger("NanoDet")


def xyxy2xywh(bbox):
    """change bbox to coco format.

    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class CocoDetectionEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, "coco_api")
        self.class_names = dataset.class_names
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]

    def evaluate(self, results, save_dir, rank=-1):
        results_json = self._results2json(results)
        if len(results_json) == 0:
            warnings.warn(
                "Detection result is empty! Please check whether "
                "training set is too small (need to increase val_interval "
                "in config and train more epochs). Or check annotation "
                "correctness."
            )
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[key] = 0
            return empty_eval_results

        save_path = os.path.join(save_dir, f"results{rank}.json")
        self._save_result_json(results_json, save_path)
        coco_eval = self._get_coco_eval(save_path)
        self._logging_eval_result(coco_eval)
        self._logging_per_class_ap(coco_eval)
        eval_result = self._get_eval_result(coco_eval)
        return eval_result

    def _results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score,
                    )
                    json_results.append(detection)
        return json_results

    @staticmethod
    def _save_result_json(results_json, save_path):
        with open(save_path, "w") as f:
            json.dump(results_json, f)

    def _get_coco_eval(self, json_path):
        coco_dets = self.coco_api.loadRes(json_path)
        coco_eval = COCOeval(
            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox"
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        return coco_eval

    def _get_eval_result(self, coco_eval):
        aps = coco_eval.stats[:6]
        eval_results = {}
        for metric_name, ap in zip(self.metric_names, aps):
            eval_results[metric_name] = ap
        return eval_results

    def _logging_per_class_ap(self, coco_eval):
        per_class_ap50s, per_class_maps = self._get_per_class_infos(coco_eval)
        table = self._generate_table(per_class_ap50s, per_class_maps)
        logger.info("\n%s", table)

    def _generate_table(self, per_class_ap50s, per_class_maps):
        headers = ["class", "AP50", "mAP"]
        colums = 6
        num_cols = min(colums, len(self.class_names) * len(headers))
        flatten_results = []
        for name, ap50, mAP in zip(self.class_names, per_class_ap50s, per_class_maps):
            flatten_results += [name, ap50, mAP]

        row_pair = itertools.zip_longest(
            *[flatten_results[i::num_cols] for i in range(num_cols)]
        )
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair,
            tablefmt="pipe",
            floatfmt=".1f",
            headers=table_headers,
            numalign="left",
        )

        return table

    def _get_per_class_infos(self, coco_eval):
        per_class_ap50s = []
        per_class_maps = []
        precisions = coco_eval.eval["precision"]
        # dimension of precisions: [TxRxKxAxM]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(self.class_names) == precisions.shape[2]

        for idx in range(len(self.class_names)):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision_50 = precisions[0, :, idx, 0, -1]
            precision_50 = precision_50[precision_50 > -1]
            ap50 = np.mean(precision_50) if precision_50.size else float("nan")
            per_class_ap50s.append(float(ap50 * 100))

            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_maps.append(float(ap * 100))
        return per_class_ap50s, per_class_maps

    @staticmethod
    def _logging_eval_result(coco_eval):
        # use logger to log coco eval results
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        logger.info("\n%s", redirect_string.getvalue())
