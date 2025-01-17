import math
import random
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


def get_flip_matrix(prob=0.5):
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


def get_perspective_matrix(perspective=0.0):
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    return P


def get_rotation_matrix(degree=0.0):
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
    return R


def get_scale_matrix(ratio=(1, 1)):
    Scl = np.eye(3)
    scale = random.uniform(*ratio)
    Scl[0, 0] *= scale
    Scl[1, 1] *= scale
    return Scl


def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
    Str = np.eye(3)
    Str[0, 0] *= random.uniform(*width_ratio)
    Str[1, 1] *= random.uniform(*height_ratio)
    return Str


def get_shear_matrix(degree):
    Sh = np.eye(3)
    Sh[0, 1] = math.tan(random.uniform(-degree, degree) * math.pi / 180)
    Sh[1, 0] = math.tan(random.uniform(-degree, degree) * math.pi / 180)
    return Sh


def get_translate_matrix(translate, width, height):
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    return T


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """Get resize matrix for resizing raw img to input size.

    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)

    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2
        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio
        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C

    Rs[0, 0] *= d_w / r_w
    Rs[1, 1] *= d_h / r_h
    return Rs


def warp_and_resize(
    meta: Dict,
    warp_kwargs: Dict,
    dst_shape: Tuple[int, int],
    keep_ratio: bool = True,
):
    def use_strategy():
        return random.randint(0, 1)

    warp_strategies: Dict[str, Callable] = {
        "perspective": get_perspective_matrix,
        "scale": get_scale_matrix,
        "stretch": get_stretch_matrix,
        "rotation": get_rotation_matrix,
        "shear": get_shear_matrix,
        "flip": get_flip_matrix,
    }
    raw_img = meta["img"]
    height = raw_img.shape[0]
    width = raw_img.shape[1]
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2
    for strategy, warp_func in warp_strategies.items():
        if (strategy in warp_kwargs) & use_strategy():
            if isinstance(warp_kwargs[strategy], Iterable):
                C = warp_func(*warp_kwargs[strategy]) @ C
            else:
                C = warp_func(warp_kwargs[strategy]) @ C

    if "translate" in warp_kwargs and random.randint(0, 1):
        T = get_translate_matrix(warp_kwargs["translate"], width, height)
    else:
        T = get_translate_matrix(0, width, height)
    M = T @ C

    ResizeM = get_resize_matrix((width, height), dst_shape, keep_ratio)
    M = ResizeM @ M

    img = cv2.warpPerspective(raw_img, M, dsize=tuple(dst_shape))
    meta["img"] = img
    meta["warp_matrix"] = M

    if "gt_bboxes" in meta:
        boxes = meta["gt_bboxes"]
        meta["gt_bboxes"] = warp_boxes(boxes, M, dst_shape[0], dst_shape[1])
    if "gt_masks" in meta:
        for i, mask in enumerate(meta["gt_masks"]):
            meta["gt_masks"][i] = cv2.warpPerspective(mask, M, dsize=tuple(dst_shape))
    return meta


def warp_boxes(boxes, M, width, height):
    if n := len(boxes):
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    return boxes


def get_minimum_dst_shape(
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    divisible: Optional[int] = None,
) -> Tuple[int, int]:
    """Calculate minimum dst shape."""
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape
    if src_w / src_h < dst_w / dst_h:
        ratio = dst_h / src_h
    else:
        ratio = dst_w / src_w
    dst_w = int(ratio * src_w)
    dst_h = int(ratio * src_h)
    if divisible and divisible > 0:
        dst_w = max(divisible, int((dst_w + divisible - 1) // divisible * divisible))
        dst_h = max(divisible, int((dst_h + divisible - 1) // divisible * divisible))
    return dst_w, dst_h


class ShapeTransform:
    """Shape transforms including resize, random perspective, random scale, random stretch, random rotation, random
    shear, random translate, and random flip.

    Args:
        keep_ratio: Whether to keep aspect ratio of the image.
        divisible: Make image height and width is divisible by a number.
        perspective: Random perspective factor.
        scale: Random scale ratio.
        stretch: Width and height stretch ratio range.
        rotation: Random rotate degree.
        shear: Random shear degree.
        translate: Random translate ratio.
        flip: Random flip probability.
    """

    def __init__(
        self,
        keep_ratio: bool,
        divisible: int = 0,
        perspective: float = 0.0,
        scale: Tuple[int, int] = (1, 1),
        stretch: Tuple = ((1, 1), (1, 1)),
        rotation: float = 0.0,
        shear: float = 0.0,
        translate: float = 0.0,
        flip: float = 0.0,
        **kwargs,
    ):
        self.keep_ratio = keep_ratio
        self.divisible = divisible
        self.perspective = perspective
        self.scale_ratio = scale
        self.stretch_ratio = stretch
        self.rotation_degree = rotation
        self.shear_degree = shear
        self.flip_prob = flip
        self.translate_ratio = translate

    def __call__(self, meta_data, dst_shape):
        raw_img = meta_data["img"]
        dst_shape, warp_matrix = self._get_warp_matrix(raw_img, dst_shape)
        img = cv2.warpPerspective(raw_img, warp_matrix, dsize=tuple(dst_shape))

        # update meta data
        meta_data["img"] = img
        meta_data["warp_matrix"] = warp_matrix
        if "gt_bboxes" in meta_data:
            boxes = meta_data["gt_bboxes"]
            meta_data["gt_bboxes"] = warp_boxes(
                boxes, warp_matrix, dst_shape[0], dst_shape[1]
            )
        if "gt_masks" in meta_data:
            for i, mask in enumerate(meta_data["gt_masks"]):
                meta_data["gt_masks"][i] = cv2.warpPerspective(
                    mask, warp_matrix, dsize=tuple(dst_shape)
                )
        return meta_data

    def _get_warp_matrix(self, raw_img, dst_shape):
        height = raw_img.shape[0]
        width = raw_img.shape[1]
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2
        P = get_perspective_matrix(self.perspective)
        C = P @ C
        Scl = get_scale_matrix(self.scale_ratio)
        C = Scl @ C
        Str = get_stretch_matrix(*self.stretch_ratio)
        C = Str @ C
        R = get_rotation_matrix(self.rotation_degree)
        C = R @ C
        Sh = get_shear_matrix(self.shear_degree)
        C = Sh @ C
        F = get_flip_matrix(self.flip_prob)
        C = F @ C
        T = get_translate_matrix(self.translate_ratio, width, height)
        M = T @ C
        if self.keep_ratio:
            dst_shape = get_minimum_dst_shape(
                (width, height), dst_shape, self.divisible
            )
        ResizeM = get_resize_matrix((width, height), dst_shape, self.keep_ratio)
        M = ResizeM @ M
        return dst_shape, M
