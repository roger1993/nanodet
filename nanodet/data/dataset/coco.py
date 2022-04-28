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
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from .base import BaseDataset


def get_anns(coco_api, img_id) -> List[Dict[str, Any]]:
    ann_ids = coco_api.getAnnIds([img_id])
    anns = coco_api.loadAnns(ann_ids)
    return anns


def is_valid(ann: Dict[str, Any], cat_ids: List[str]) -> bool:
    # get keep flag
    keep_flag = not ann.get("ignore", False)

    # get valid area flag
    w, h = ann["bbox"][2:]
    valid_area_flag = not (ann["area"] <= 0 or w < 1 or h < 1)

    # get contrain flag
    contain_flag = ann["category_id"] in cat_ids

    return keep_flag & valid_area_flag & contain_flag


def get_parse_result(
    anns,
    cat_ids,
    cat2label,
    use_instance_mask,
    coco_api,
    use_keypoint,
) -> Tuple[List, List, List, List, List]:
    gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, gt_keypoints = (
        [],
        [],
        [],
        [],
        [],
    )

    for ann in anns:
        if is_valid(ann, cat_ids):
            x1, y1, w, h = ann["bbox"]
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(cat2label[ann["category_id"]])
                if use_instance_mask:
                    gt_masks.append(coco_api.annToMask(ann))
                if use_keypoint:
                    gt_keypoints.append(ann["keypoints"])

    return gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, gt_keypoints


def convert_parse_result(
    gt_bboxes: List, gt_labels: List, gt_bboxes_ignore: List
) -> Tuple[np.array, np.array, np.array]:
    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)
    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
    return gt_bboxes, gt_labels, gt_bboxes_ignore


def parse_anns(
    anns,
    cat_ids,
    cat2label,
    use_instance_mask,
    coco_api,
    use_keypoint,
) -> Tuple[np.array, np.array, np.array, List, List]:

    (
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore,
        gt_masks,
        gt_keypoints,
    ) = get_parse_result(
        anns,
        cat_ids,
        cat2label,
        use_instance_mask,
        coco_api,
        use_keypoint,
    )
    gt_bboxes, gt_labels, gt_bboxes_ignore = convert_parse_result(
        gt_bboxes, gt_labels, gt_bboxes_ignore
    )
    return gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, gt_keypoints


def build_annotation(
    gt_bboxes: np.array,
    gt_labels: np.array,
    gt_bboxes_ignore: np.array,
    gt_masks: List,
    gt_keypoints: List,
):
    annotation = {
        "bboxes": gt_bboxes,
        "labels": gt_labels,
        "bboxes_ignore": gt_bboxes_ignore,
    }
    annotation["masks"] = gt_masks
    annotation["keypoints"] = (
        np.array(gt_keypoints, dtype=np.float32)
        if gt_keypoints
        else np.zeros((0, 51), dtype=np.float32)
    )
    return annotation


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
        img_info = self.get_per_img_info(idx)
        img = self.get_img_data(img_info["file_name"])
        ann = self.get_img_annotation(idx)
        meta = self.build_data(img_info, img, ann)
        return meta

    def get_val_data(self, idx: int) -> Dict[str, Any]:
        # TODO: support TTA
        return self.get_train_data(idx)

    def build_data(
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

    def get_img_data(self, file_name: str) -> np.ndarray:
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"image {image_path} read failed.")
            raise FileNotFoundError("Cant load image! Please check image path!")
        return img

    def get_per_img_info(self, idx: int) -> Dict[str, Any]:
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id_ = img_info["id"]
        if not isinstance(id_, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id_}
        return info

    def get_img_annotation(self, idx: int) -> Dict[str, Any]:
        anns = get_anns(self.coco_api, self.img_ids[idx])
        gt_attributes = parse_anns(
            anns,
            self.cat_ids,
            self.cat2label,
            self.use_instance_mask,
            self.coco_api,
            self.use_keypoint,
        )
        annotation = build_annotation(*gt_attributes)
        return annotation
