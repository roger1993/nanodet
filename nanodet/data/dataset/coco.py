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
import os
from collections import defaultdict
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from .base import BaseDataset


class CocoDataset(BaseDataset):
    def get_data_info(self) -> List[Dict[str, Any]]:
        self.coco_api = COCO(self.ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.class_names = [cat["name"] for cat in self.cats]
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def get_train_data(self, idx: int) -> Dict[str, Any]:
        img_info = self._get_per_img_info(idx)
        img = self._get_img_data(img_info["file_name"])
        ann = self._get_img_annotation(idx)
        meta = self._build_data(img_info, img, ann)
        return meta

    def get_val_data(self, idx: int) -> Dict[str, Any]:
        # TODO: support TTA
        return self.get_train_data(idx)

    def _build_data(
        self, img_info: Dict[str, Any], img: np.ndarray, ann: Dict[str, Any]
    ) -> Dict[str, Any]:
        meta = {
            "img": img,
            "img_info": img_info,
            "gt_bboxes": ann["bboxes"],
            "gt_labels": ann["labels"],
        }
        if self.use_instance_mask:
            meta["gt_masks"] = ann["masks"]
        if self.use_keypoint:
            meta["gt_keypoints"] = ann["keypoints"]
        input_size = self.input_size
        if self.multi_scale:
            input_size = self.get_random_size(self.multi_scale, input_size)
        meta = self.pipeline(meta, input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        return meta

    def _get_img_data(self, file_name: str) -> np.ndarray:
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"image {image_path} read failed.")
            raise FileNotFoundError("Cant load image! Please check image path!")
        return img

    def _get_per_img_info(self, idx: int) -> Dict[str, Any]:
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id_ = img_info["id"]
        if not isinstance(id_, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id_}
        return info

    def _get_img_annotation(self, idx: int) -> Dict[str, Any]:
        anns = self._get_anns(idx)
        parse_result = self._parse_anns(anns)
        annotation = self._build_annotation(parse_result)
        return annotation

    def _get_anns(self, idx: int) -> List[Dict[str, Any]]:
        ann_ids = self.coco_api.getAnnIds([self.img_ids[idx]])
        anns = self.coco_api.loadAnns(ann_ids)
        return anns

    def _parse_anns(self, anns: List[Dict[str, Any]]) -> Dict[str, Any]:
        parse_result = self._get_parse_result(anns)
        convert_result = self._convert_parse_result(parse_result)
        return convert_result

    def _get_parse_result(self, anns: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        parse_result: Dict[str, List[Any]] = defaultdict(list)
        for ann in anns:
            if self._is_valid(ann):
                x1, y1, w, h = ann["bbox"]
                bbox = [x1, y1, x1 + w, y1 + h]
                if ann.get("iscrowd", False):
                    parse_result["gt_bboxes_ignore"].append(bbox)
                else:
                    parse_result["gt_bboxes"].append(bbox)
                    parse_result["gt_labels"].append(self.cat2label[ann["category_id"]])
                    if self.use_instance_mask:
                        parse_result["gt_masks"].append(self.coco_api.annToMask(ann))
                    if self.use_keypoint:
                        parse_result["gt_keypoints"].append(ann["keypoints"])
        return parse_result

    def _is_valid(self, ann: Dict[str, Any]) -> bool:
        # get keep flag
        keep_flag = not ann.get("ignore", False)

        # get valid area flag
        w, h = ann["bbox"][2:]
        valid_area_flag = not (ann["area"] <= 0 or w < 1 or h < 1)

        # get contrain flag
        contain_flag = ann["category_id"] in self.cat_ids

        return keep_flag & valid_area_flag & contain_flag

    @staticmethod
    def _convert_parse_result(parse_result: Dict[str, Any]) -> Dict[str, Any]:
        if parse_result["gt_bboxes"]:
            parse_result["gt_bboxes"] = np.array(
                parse_result["gt_bboxes"], dtype=np.float32
            )
            parse_result["gt_labels"] = np.array(
                parse_result["gt_labels"], dtype=np.int64
            )
        else:
            parse_result["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
            parse_result["gt_labels"] = np.array([], dtype=np.int64)
        if parse_result["gt_bboxes_ignore"]:
            parse_result["gt_bboxes_ignore"] = np.array(
                parse_result["gt_bboxes_ignore"], dtype=np.float32
            )
        else:
            parse_result["gt_bboxes_ignore"] = np.zeros((0, 4), dtype=np.float32)
        return parse_result

    def _build_annotation(self, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        annotation = {
            "bboxes": parse_result["gt_bboxes"],
            "labels": parse_result["gt_labels"],
            "bboxes_ignore": parse_result["gt_bboxes_ignore"],
        }
        annotation["masks"] = parse_result["gt_masks"]
        annotation["keypoints"] = (
            np.array(parse_result["gt_keypoints"], dtype=np.float32)
            if parse_result["gt_keypoints"]
            else np.zeros((0, 51), dtype=np.float32)
        )
        if self.use_instance_mask:
            annotation["masks"] = parse_result["gt_masks"]
        if self.use_keypoint:
            if parse_result["gt_keypoints"]:
                annotation["keypoints"] = np.array(
                    parse_result["gt_keypoints"], dtype=np.float32
                )
            else:
                annotation["keypoints"] = np.zeros((0, 51), dtype=np.float32)
        return annotation
