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

from .ema import ExpMovingAverager


def build_weight_averager(cfg, device="cpu"):
    support_weight_averager = {
        "ExpMovingAverager": ExpMovingAverager,
    }
    support_weight_averager_str = ",".join(support_weight_averager)
    weight_averager_cfg = copy.deepcopy(cfg)
    name = weight_averager_cfg.pop("name")
    error_message = f"Unknown weight_averager {name}. Currently supported heads are: {support_weight_averager_str}"
    assert name in support_weight_averager, error_message
    return support_weight_averager[name](**weight_averager_cfg, device=device)
