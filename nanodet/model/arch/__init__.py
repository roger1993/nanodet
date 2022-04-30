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

from .nanodet_plus import NanoDetPlus
from .one_stage_detector import OneStageDetector


def build_model(model_cfg):
    support_model = {"OneStageDetector": OneStageDetector, "NanoDetPlus": NanoDetPlus}
    support_model_str = ",".join(support_model)
    model_cfg = copy.deepcopy(model_cfg)
    name = model_cfg.arch.pop("name")
    error_message = (
        f"Unknown model {name}. Currently supported models are: {support_model_str}"
    )
    assert name in support_model, error_message
    return support_model[name](**model_cfg.arch)
