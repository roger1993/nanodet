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
import random
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from nanodet.data.transform import Pipeline


# TODO: use dataclass to warpper multiple attributes.
class BaseDataset(Dataset, metaclass=ABCMeta):
    """A base class of detection dataset. Referring from MMDetection. A dataset should have images, annotations and
    preprocessing pipelines NanoDet use [xmin, ymin, xmax, ymax] format for box and.

    [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
    instance masks should decode into binary masks for each instance like
    {
        'bbox': [xmin,ymin,xmax,ymax],
        'mask': mask
    }
    segmentation mask should decode into binary masks for each class.

    Parameters
    ----------
    img_path : str
        image data folder.
    ann_path : str
        annotation file path or folder.
    input_size : Tuple[int, int]
        input img size.
    pipeline : Dict
        pipeline config.
    keep_ratio : bool, optional
        keep img ratio, by default True
    use_instance_mask : bool, optional
        load instance segmentation data, by default False
    use_seg_mask : bool, optional
        load semantic segmentation data, by default False
    use_keypoint : bool, optional
        load pose keypoint data, by default False
    load_mosaic : bool, optional
        using mosaic data augmentation from yolov4, by default False
    mode : str, optional
        'train' or 'val' or 'test', by default "train"
    multi_scale : Optional[Tuple[float, float]], optional
        Multi-scale factor range, by default None
    """

    def __init__(
        self,
        img_path: str,
        ann_path: str,
        input_size: Tuple[int, int],
        pipeline: Dict,
        keep_ratio: bool = True,
        use_instance_mask: bool = False,
        use_seg_mask: bool = False,
        use_keypoint: bool = False,
        load_mosaic: bool = False,
        mode: str = "train",
        multi_scale: Optional[Tuple[float, float]] = None,
    ):
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.pipeline = Pipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.multi_scale = multi_scale
        self.mode = mode
        self.data_info = self.get_data_info()

    def __len__(self) -> int:
        return len(self.data_info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode in ["val", "test"]:
            return self.get_val_data(idx)
        while True:
            data = self.get_train_data(idx)
            if data is None:
                idx = self.get_another_id()
                continue
            return data

    @staticmethod
    def get_random_size(
        scale_range: Tuple[float, float], image_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get random image shape by multi-scale factor and image_size.
        Args:
            scale_range (Tuple[float, float]): Multi-scale factor range.
                Format in [(width, height), (width, height)]
            image_size (Tuple[int, int]): Image size. Format in (width, height).

        Returns:
            Tuple[int, int]
        """
        scale_factor = random.uniform(*scale_range)
        width = int(image_size[0] * scale_factor)
        height = int(image_size[1] * scale_factor)
        return width, height

    @abstractmethod
    def get_data_info(self) -> List[Dict[str, Any]]:
        """Genreate a list of img info dict. Each img info dict should like bellow:
        {
            'coco_url': 'http://images.cocoda...000009.jpg',
            'date_captured': '2013-11-19 20:40:11',
            'file_name': '000000000009.jpg',
            'flickr_url': 'http://farm5.staticf...b8d6_z.jpg',
            'height': 480,
            'id': 9,
            'license': 3,
            'width': 640
        }

        Returns:
            List[Dict[str, Any]]: list of img info dict.
        """

    @abstractmethod
    def get_train_data(self, idx: int) -> Dict[str, Any]:
        """Generate train data for given `idx` img. bellow is a sample data generate by
        `CocoDataset`:
        {
            'img': tensor([[[ 0.4047,  ...1.8996]]]),
            'img_info': {
                'file_name': '000000000139.jpg',
                'height': 426,
                'width': 640,
                'id': 139
            },
            'gt_bboxes': array([[144.29268, ...),
            'gt_labels': array([58, 62, 62, ...]),
            'warp_matrix': array([[0.73154684, ...]])
        }
        * img: tensor of img data.
        * img_info: dict stores img related information.
        * gt_bboxes: ground truth bounding box coordinate value.
        * gt_labels: ground truth label.
        * warp_matrix: warp matrix.

        Parameters
        ----------
        idx : int
            index value.

        Returns
        -------
        Dict[str, Any]
            train data info for a given `idx`.
        """

    @abstractmethod
    def get_val_data(self, idx: int) -> Dict[str, Any]:
        """Please refer to `get_train_data`."""

    def get_another_id(self) -> int:
        """Random generate another index value.

        Returns
        -------
        int
            index value.
        """
        return np.random.random_integers(0, len(self.data_info) - 1)
