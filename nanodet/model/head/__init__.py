import copy

from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead
from .nanodet_plus_head import NanoDetPlusHead
from .simple_conv_head import SimpleConvHead


def build_head(cfg):
    support_head = {
        "GFLHead": GFLHead,
        "NanoDetHead": NanoDetHead,
        "NanoDetPlusHead": NanoDetPlusHead,
        "SimpleConvHead": SimpleConvHead,
    }
    support_head_str = ",".join(support_head)
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    error_message = (
        f"Unknown head {name}. Currently supported heads are: {support_head_str}"
    )
    assert name in support_head, error_message
    return support_head[name](**head_cfg)
