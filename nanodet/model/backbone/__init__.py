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

from .custom_csp import CustomCspNet
from .efficientnet_lite import EfficientNetLite
from .ghostnet import GhostNet
from .mobilenetv2 import MobileNetV2
from .repvgg import RepVGG
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2


def build_backbone(cfg):
    support_backbone = {
        "ResNet": ResNet,
        "ShuffleNetV2": ShuffleNetV2,
        "GhostNet": GhostNet,
        "MobileNetV2": MobileNetV2,
        "EfficientNetLite": EfficientNetLite,
        "CustomCspNet": CustomCspNet,
        "RepVGG": RepVGG,
    }
    support_backbone_str = ",".join(support_backbone)
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    error_message = f"Unknown backbone {name}. Currently supported backbones are: {support_backbone_str}"
    assert name in support_backbone, error_message
    return support_backbone[name](**backbone_cfg)
