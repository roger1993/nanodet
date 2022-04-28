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

import copy

from .coco import CocoDataset
from .xml_dataset import XMLDataset


def build_dataset(cfg, mode):
    dataset_map = {
        "coco": CocoDataset,
        "xml_dataset": XMLDataset,
        "CocoDataset": CocoDataset,
        "XMLDataset": XMLDataset,
    }

    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")

    error_message = (
        f"Unsupported dataset name: {name}, "
        f"currently supported datasets: {list(dataset_map)}"
    )
    assert name in dataset_map, error_message
    assert mode in ["train", "val", "test"], f"Unknown mode: {mode}"

    return dataset_map[name](mode=mode, **dataset_cfg)
