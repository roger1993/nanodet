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
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pycocotools.coco import COCO

from .coco import CocoDataset


class CocoXML(COCO):
    """Constructor of Microsoft COCO helper class for reading and visualizing annotations.

    Parameters
    ----------
    annotation : dict
        dict which contain annotation info.
    """

    def __init__(self, annotation):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset = annotation
        self.createIndex()


class XMLDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super().__init__(**kwargs)

    def get_data_info(self) -> List[Dict[str, Any]]:
        coco_dict = self._xml_to_coco(self.ann_path)
        self.coco_api = CocoXML(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def _xml_to_coco(self, ann_path: str):
        ann_file_names = self._get_file_list(ann_path, file_type=".xml")
        image_info, annotations = [], []
        categories = [
            {"supercategory": supercat, "id": idx + 1, "name": supercat}
            for idx, supercat in enumerate(self.class_names)
        ]
        ann_id = 1
        for idx, xml_name in enumerate(ann_file_names, start=1):
            root = ET.parse(os.path.join(ann_path, xml_name)).getroot()
            info = self._parse_info(root, idx)
            image_info.append(info)
            for node in root.findall("object"):
                ann = self._get_anno(
                    node,
                    categories,
                    info,
                )
                if ann is not None:
                    ann.update({"image_id": idx, "id": ann_id})
                    annotations.append(ann)
                    ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        return coco_dict

    @staticmethod
    def _get_file_list(path, file_type: str = ".xml") -> List[str]:
        # each img has its own annotation file.
        file_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                path_ = os.path.join(maindir, filename)
                ext = os.path.splitext(path_)[1]
                if ext == file_type:
                    file_names.append(filename)
        return file_names

    @staticmethod
    def _parse_info(root, idx) -> Dict[str, Any]:
        file_name = root.find("filename").text
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        info = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": idx + 1,
        }
        return info

    def _get_anno(self, node, categories, info) -> Optional[Dict[str, Any]]:
        category = node.find("name").text
        if category not in self.class_names:
            return None

        xmin = int(node.find("bndbox").find("xmin").text)
        ymin = int(node.find("bndbox").find("ymin").text)
        xmax = int(node.find("bndbox").find("xmax").text)
        ymax = int(node.find("bndbox").find("ymax").text)
        w = xmax - xmin
        h = ymax - ymin
        if w < 0 or h < 0:
            return None
        coco_box = [
            max(xmin, 0),
            max(ymin, 0),
            min(w, info["width"]),
            min(h, info["height"]),
        ]
        cat_id = None
        for cat in categories:
            if category == cat["name"]:
                cat_id = cat["id"]
                break
        ann = {
            "bbox": coco_box,
            "category_id": cat_id,
            "iscrowd": 0,
            "area": coco_box[2] * coco_box[3],
        }
        return ann
