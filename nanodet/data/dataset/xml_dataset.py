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

import logging
import os
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from collections import defaultdict

from pycocotools.coco import COCO

from .coco import CocoDataset


def get_file_list(path, file_type: str = ".xml") -> List[str]:
    """_summary_

    Parameters
    ----------
    path : _type_
        _description_
    file_type : str, optional
        _description_, by default ".xml"

    Returns
    -------
    List[str]
        _description_
    """
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            path_ = os.path.join(maindir, filename)
            ext = os.path.splitext(path_)[1]
            if ext == file_type:
                file_names.append(filename)
    return file_names


def parse_info(root, idx) -> Dict[str, Any]:
    """_summary_

    Parameters
    ----------
    root : _type_
        _description_
    idx : _type_
        _description_

    Returns
    -------
    Dict[str, Any]
        _description_
    """
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


def get_anno(
    node, class_names, categories, width, height, image_id, ann_id
) -> Optional[Dict[str, Any]]:
    category = node.find("name").text
    if category not in class_names:
        return None
    for cat in categories:
        if category == cat["name"]:
            cat_id = cat["id"]
            break
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
        min(w, width),
        min(h, height),
    ]
    ann = {
        "image_id": image_id + 1,
        "bbox": coco_box,
        "category_id": cat_id,
        "iscrowd": 0,
        "id": ann_id,
        "area": coco_box[2] * coco_box[3],
    }
    return ann


class CocoXML(COCO):
    """Constructor of Microsoft COCO helper class for reading and visualizing annotations.

    Parameters
    ----------
    annotation : _type_
        _description_
    """

    def __init__(self, annotation):
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset = annotation
        self.createIndex()


class XMLDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        super().__init__(**kwargs)
        self.class_names = class_names

    def xml_to_coco(self, ann_path: str):
        """convert xml annotations to coco_api

        Parameters
        ----------
        ann_path : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        ann_file_names = get_file_list(ann_path, file_type=".xml")
        image_info, annotations = [], []
        categories = [
            {"supercategory": supercat, "id": idx + 1, "name": supercat}
            for idx, supercat in enumerate(self.class_names)
        ]
        ann_id = 1
        for idx, xml_name in enumerate(ann_file_names):
            root = ET.parse(os.path.join(ann_path, xml_name)).getroot()
            info = parse_info(root, idx)
            image_info.append(info)
            for node in root.findall("object"):
                ann = get_anno(
                    node,
                    self.class_names,
                    categories,
                    info["width"],
                    info["height"],
                    idx,
                    ann_id,
                )
                if ann is not None:
                    annotations.append(ann)
                    ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        return coco_dict

    def get_data_info(self) -> List[Dict[str, Any]]:
        """_summary_

        Returns
        -------
        List[Dict[str, Any]]
            _description_
        """
        coco_dict = self.xml_to_coco(self.ann_path)
        self.coco_api = CocoXML(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
