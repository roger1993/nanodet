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

import time

import torch
import torch.nn as nn

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


class OneStageDetector(nn.Module):
    def __init__(
        self,
        backbone,
        fpn=None,
        head=None,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone)
        if fpn is not None:
            self.fpn = build_fpn(fpn)
        if head is not None:
            self.head = build_head(head)
        self.epoch = 0

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, "fpn"):
            x = self.fpn(x)
        if hasattr(self, "head"):
            x = self.head(x)
        return x

    def inference(self, meta):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta["img"])
            torch.cuda.synchronize()
            time2 = time.time()
            print(f"forward time: {time2 - time1:.3f}s", end=" | ")
            results = self.head.post_process(preds, meta)
            torch.cuda.synchronize()
            print(f"decode time: {time.time() - time2:.3f}s", end=" | ")
        return results

    def forward_train(self, gt_meta):
        preds = self(gt_meta["img"])
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states

    def set_epoch(self, epoch):
        self.epoch = epoch
