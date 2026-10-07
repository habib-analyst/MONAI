# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of "vanilla" transforms for spatial operations.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from itertools import zip_longest
from typing import Any, cast

import numpy as np
import torch

from monai.config import USE_COMPILED, DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import BoxMode, StandardMode
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor, get_spatial_ndim
from monai.data.utils import AFFINE_TOL, affine_to_spacing, compute_shape_offset, iter_patch, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.networks.utils import meshgrid_ij
from monai.transforms.croppad.array import CenterSpatialCrop, ResizeWithPadOrCrop
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.functional import (
    _compiled_unsupported,
    affine_func,
    convert_box_to_points,
    convert_points_to_box,
    flip,
    orientation,
    resize,
    rotate,
    rotate90,
    spatial_resample,
    zoom,
)
from monai.transforms.traits import MultiSampleTrait
from monai.transforms.transform import LazyTransform, Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import (
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
    resolves_modes,
    scale_affine,
)
from monai.transforms.utils_pytorch_numpy_unification import argsort, argwhere, linalg_inv, moveaxis
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    SpaceKeys,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)
from monai.utils.deprecate_utils import deprecated_arg_default
from monai.utils.enums import GridPatchSort, PatchKeys, TraceKeys, TransformBackends
from monai.utils.misc import ImageMetaKey as Key
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, get_equivalent_dtype, get_torch_dtype_from_string

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = [
    "SpatialResample",
    "ResampleToMatch",
    "Spacing",
    "Orientation",
    "Flip",
    "GridDistortion",
    "GridSplit",
    "GridPatch",
    "RandGridPatch",
    "Resize",
    "Rotate",
    "Zoom",
    "Rotate90",
    "RandRotate90",
    "RandRotate",
    "RandFlip",
    "RandGridDistortion",
    "RandAxisFlip",
    "RandZoom",
    "AffineGrid",
    "RandAffineGrid",
    "RandDeformGrid",
    "Resample",
    "Affine",
    "RandAffine",
    "Rand2DElastic",
    "Rand3DElastic",
    "RandSimulateLowResolution",
]

RandRange = Sequence[tuple[float, float] | float] | float | None


class SpatialResample(InvertibleTransform, LazyTransform):
    """
    Resample input image from the orientation/spacing defined by ``src_affine`` affine matrix into
    the ones specified by ``dst_affine`` affine matrix.

    Internally this transform computes the affine transform matrix from ``src_affine`` to ``dst_affine``,
    by ``xform = linalg.solve(src_affine, dst_affine)``, and call ``monai.transforms.Affine`` with ``xform``.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY, TransformBackends.CUPY]

    def __init__(
        self,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        lazy: bool = False,
    ):
        """
        Args:
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        LazyTransform.__init__(self, lazy=lazy)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        dst_affine: torch.Tensor | None = None,
        spatial_size: Sequence[int] | torch.Tensor | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: input image to be resampled. It currently supports channel-first arrays with
                at most three spatial dimensions.
            dst_affine: destination affine matrix. Defaults to ``None``, which means the same as `img.affine`.
                the shape should be `(r+1, r+1)` where `r` is the spatial rank of ``img``.
                when `dst_affine` and `spatial_size` are None, the input will be returned without resampling,
                but the data type will be `float32`.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined,
                the transform will compute a spatial size automatically containing the previous field of view.
                if `spatial_size` is ``-1`` are the transform will use the corresponding input img size.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                Defaults to ``None``, effectively using the value of `self.align_corners`.
            dtype: data type for resampling computation. Defaults to ``self.dtype`` or
                ``np.float64`` (for best precision). If ``None``, use the data type of input data.
                To be compatible with other modules, the output data type is always `float32`.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        The spatial rank is determined by the smallest among ``img.ndim -1``, ``len(src_affine) - 1``, and ``3``.

        When both ``monai.config.USE_COMPILED`` and ``align_corners`` are set to ``True``,
        MONAI's resampling implementation will be used.
        Set `dst_affine` and `spatial_size` to `None` to turn off the resampling step.
        """
        # get dtype as torch (e.g., torch.float64)
        dtype_pt = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        align_corners = align_corners if align_corners is not None else self.align_corners
        mode = mode if mode is not None else self.mode
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        lazy_ = self.lazy if lazy is None else lazy
        return spatial_resample(
            img,
            dst_affine,
            spatial_size,
            mode,
            padding_mode,
            align_corners,
            dtype_pt,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # Create inverse transform
        kw_args = transform[TraceKeys.EXTRA_INFO]
        # need to convert dtype from string back to torch.dtype
        kw_args["dtype"] = get_torch_dtype_from_string(kw_args["dtype"])
        # source becomes destination
        kw_args["dst_affine"] = kw_args.pop("src_affine")
        kw_args["spatial_size"] = transform[TraceKeys.ORIG_SIZE]
        if kw_args.get("align_corners") == TraceKeys.NONE:
            kw_args["align_corners"] = False
        with self.trace_transform(False):
            # we can't use `self.__call__` in case a child class calls this inverse.
            out: torch.Tensor = SpatialResample.__call__(self, data, **kw_args)
        kw_args["src_affine"] = kw_args.get("dst_affine")
        return out


class ResampleToMatch(SpatialResample):
    """
    Resample an image to match given metadata. The affine matrix will be aligned,
    and the size of the output image will match.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    def __call__(  # type: ignore
        self,
        img: torch.Tensor,
        img_dst: torch.Tensor,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: input image to be resampled to match ``img_dst``. It currently supports channel-first arrays with
                at most three spatial dimensions.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                Defaults to ``None``, effectively using the value of `self.align_corners`.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype`` or
                ``np.float64`` (for best precision). If ``None``, use the data type of input data.
                To be compatible with other modules, the output data type is always `float32`.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Raises:
            ValueError: When the affine matrix of the source image is not invertible.
        Returns:
            Resampled input tensor or MetaTensor.
        """
        if img_dst is None:
            raise RuntimeError("`img_dst` is missing.")
        dst_affine = img_dst.peek_pending_affine() if isinstance(img_dst, MetaTensor) else torch.eye(4)
        lazy_ = self.lazy if lazy is None else lazy
        img = super().__call__(
            img=img,
            dst_affine=dst_affine,
            spatial_size=img_dst.peek_pending_shape() if isinstance(img_dst, MetaTensor) else img_dst.shape[1:],
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=dtype,
            lazy=lazy_,
        )
        if not lazy_:
            if isinstance(img, MetaTensor):
                img.affine = dst_affine
                if isinstance(img_dst, MetaTensor):
                    original_fname = img.meta.get(Key.FILENAME_OR_OBJ, "resample_to_match_source")
                    img.meta = deepcopy(img_dst.meta)
                    img.meta[Key.FILENAME_OR_OBJ] = original_fname  # keep the original name, the others are overwritten
        else:
            if isinstance(img, MetaTensor) and isinstance(img_dst, MetaTensor):
                original_fname = img.meta.get(Key.FILENAME_OR_OBJ, "resample_to_match_source")
                meta_dict = deepcopy(img_dst.meta)
                for k in ("affine", "spatial_shape"):  # keys that don't copy from img_dst in lazy evaluation
                    meta_dict.pop(k, None)
                img.meta.update(meta_dict)
                img.meta[Key.FILENAME_OR_OBJ] = original_fname  # keep the original name, the others are overwritten
        return img


class Spacing(InvertibleTransform, LazyTransform):
    """
    Resample input image into the specified `pixdim`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = SpatialResample.backend

    def __init__(
        self,
        pixdim: Sequence[float] | float | np.ndarray,
        diagonal: bool = False,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        scale_extent: bool = False,
        recompute_affine: bool = False,
        min_pixdim: Sequence[float] | float | np.ndarray | None = None,
        max_pixdim: Sequence[float] | float | np.ndarray | None = None,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
                items of the pixdim sequence map to the spatial dimensions of input image, if length
                of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
                if shorter, will pad with the last value. For example, for 3D image if pixdim is [1.0, 2.0] it
                will be padded to [1.0, 2.0, 2.0]
                if the components of the `pixdim` are non-positive values, the transform will use the
                corresponding components of the original pixdim, which is computed from the `affine`
                matrix of input image.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            scale_extent: whether the scale is computed based on the spacing or the full extent of voxels,
                default False. The option is ignored if output spatial size is specified when calling this transform.
                See also: :py:func:`monai.data.utils.compute_shape_offset`. When this is True, `align_corners`
                should be `True` because `compute_shape_offset` already provides the corner alignment shift/scaling.
            recompute_affine: whether to recompute affine based on the output shape. The affine computed
                analytically does not reflect the potential quantization errors in terms of the output shape.
                Set this flag to True to recompute the output affine based on the actual pixdim. Default to ``False``.
            min_pixdim: minimal input spacing to be resampled. If provided, input image with a larger spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.
            max_pixdim: maximal input spacing to be resampled. If provided, input image with a smaller spacing than this
                value will be kept in its original spacing (not be resampled to `pixdim`). Set it to `None` to use the
                value of `pixdim`. Default to `None`.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        LazyTransform.__init__(self, lazy=lazy)
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.min_pixdim = np.array(ensure_tuple(min_pixdim), dtype=np.float64)
        self.max_pixdim = np.array(ensure_tuple(max_pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.scale_extent = scale_extent
        self.recompute_affine = recompute_affine

        for mn, mx in zip(self.min_pixdim, self.max_pixdim):
            if (not np.isnan(mn)) and (not np.isnan(mx)) and ((mx < mn) or (mn < 0)):
                raise ValueError(f"min_pixdim {self.min_pixdim} must be positive, smaller than max {self.max_pixdim}.")

        self.sp_resample = SpatialResample(
            mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype=dtype, lazy=lazy
        )

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.sp_resample.lazy = val

    def __call__(
        self,
        data_array: torch.Tensor,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike = None,
        scale_extent: bool | None = None,
        output_spatial_shape: Sequence[int] | np.ndarray | int | None = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"self.mode"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"self.padding_mode"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                Defaults to ``None``, effectively using the value of `self.align_corners`.
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            scale_extent: whether the scale is computed based on the spacing or the full extent of voxels,
                The option is ignored if output spatial size is specified when calling this transform.
                See also: :py:func:`monai.data.utils.compute_shape_offset`. When this is True, `align_corners`
                should be `True` because `compute_shape_offset` already provides the corner alignment shift/scaling.
            output_spatial_shape: specify the shape of the output data_array. This is typically useful for
                the inverse of `Spacingd` where sometimes we could not compute the exact shape due to the quantization
                error with the affine.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``pixdim`` is nonpositive.

        Returns:
            data tensor or MetaTensor (resampled into `self.pixdim`).

        """
        original_spatial_shape = (
            data_array.peek_pending_shape() if isinstance(data_array, MetaTensor) else data_array.shape[1:]
        )
        sr = len(original_spatial_shape)
        if sr <= 0:
            raise ValueError(f"data_array must have at least one spatial dimension, got {original_spatial_shape}.")
        affine_: np.ndarray
        input_affine = data_array.peek_pending_affine() if isinstance(data_array, MetaTensor) else None
        if input_affine is None:
            warnings.warn("`data_array` is not of type MetaTensor, assuming affine to be identity.")
            # default to identity
            input_affine = np.eye(sr + 1, dtype=np.float64)
        affine_ = to_affine_nd(sr, convert_data_type(input_affine, np.ndarray)[0])

        out_d = self.pixdim[:sr].copy()
        if out_d.size < sr:
            out_d = np.append(out_d, [out_d[-1]] * (sr - out_d.size))

        orig_d = affine_to_spacing(affine_, sr, out_d.dtype)
        for idx, (_d, mn, mx) in enumerate(
            zip_longest(orig_d, self.min_pixdim[:sr], self.max_pixdim[:sr], fillvalue=np.nan)
        ):
            target = out_d[idx]
            mn = target if np.isnan(mn) else min(mn, target)
            mx = target if np.isnan(mx) else max(mx, target)
            if mn > mx:
                raise ValueError(f"min_pixdim is larger than max_pixdim at dim {idx}: min {mn} max {mx} out {target}.")
            out_d[idx] = _d if (mn - AFFINE_TOL) <= _d <= (mx + AFFINE_TOL) else target

        if not align_corners and scale_extent:
            warnings.warn("align_corners=False is not compatible with scale_extent=True.")

        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        scale_extent = self.scale_extent if scale_extent is None else scale_extent
        output_shape, offset = compute_shape_offset(original_spatial_shape, affine_, new_affine, scale_extent)
        new_affine[:sr, -1] = offset[:sr]

        actual_shape = list(output_shape) if output_spatial_shape is None else output_spatial_shape
        lazy_ = self.lazy if lazy is None else lazy
        data_array = self.sp_resample(
            data_array,
            dst_affine=torch.as_tensor(new_affine),
            spatial_size=actual_shape,  # type: ignore
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=dtype,
            lazy=lazy_,
        )
        if self.recompute_affine and isinstance(data_array, MetaTensor):
            if lazy_:
                raise NotImplementedError("recompute_affine is not supported with lazy evaluation.")
            ac = align_corners if align_corners is not None else self.sp_resample.align_corners
            a = scale_affine(original_spatial_shape, actual_shape, align_corners=ac)
            data_array.affine = convert_to_dst_type(a, affine_)[0]  # type: ignore
        return data_array

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return self.sp_resample.inverse(data)


class Orientation(InvertibleTransform, LazyTransform):
    """
    Change the input image's orientation into the specified based on `axcodes`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    @deprecated_arg_default(
        name="labels",
        old_default=(("L", "R"), ("P", "A"), ("I", "S")),
        new_default=None,
        msg_suffix=(
            "Default value changed to None meaning that the transform now uses the 'space' of a "
            "meta-tensor, if applicable, to determine appropriate axis labels."
        ),
    )
    def __init__(
        self,
        axcodes: str | None = None,
        as_closest_canonical: bool = False,
        labels: Sequence[tuple[str, str]] | None = None,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                If ``None``, an appropriate value is chosen depending on the
                value of the ``"space"`` metadata item of a metatensor: if
                ``"space"`` is ``"LPS"``, the value used is ``(('R', 'L'),
                ('A', 'P'), ('I', 'S'))``, if ``"space"`` is ``"RPS"`` or the
                input is not a meta-tensor or has no ``"space"`` item, the
                value ``(('L', 'R'), ('P', 'A'), ('I', 'S'))`` is used. If not
                ``None``, the provided value is always used and the ``"space"``
                metadata item (if any) of the input is ignored.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        Raises:
            ValueError: When ``axcodes=None`` and ``as_closest_canonical=True``. Incompatible values.

        See Also: `nibabel.orientations.ornt2axcodes`.

        """
        LazyTransform.__init__(self, lazy=lazy)
        if axcodes is None and not as_closest_canonical:
            raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
        if axcodes is not None and as_closest_canonical:
            warnings.warn("using as_closest_canonical=True, axcodes ignored.")
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels

    def __call__(self, data_array: torch.Tensor, lazy: bool | None = None) -> torch.Tensor:
        """
        If input type is `MetaTensor`, original affine is extracted with `data_array.affine`.
        If input type is `torch.Tensor`, original affine is assumed to be identity.

        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``axcodes`` spatiality differs from ``data_array``.

        Returns:
            data_array [reoriented in `self.axcodes`]. Output type will be `MetaTensor`
                unless `get_track_meta() == False`, in which case it will be
                `torch.Tensor`.

        """
        spatial_shape = data_array.peek_pending_shape() if isinstance(data_array, MetaTensor) else data_array.shape[1:]
        sr = len(spatial_shape)
        if sr <= 0:
            raise ValueError(f"data_array must have at least one spatial dimension, got {spatial_shape}.")
        affine_: np.ndarray
        affine_np: np.ndarray
        labels = self.labels
        if isinstance(data_array, MetaTensor):
            affine_np, *_ = convert_data_type(data_array.peek_pending_affine(), np.ndarray)
            affine_ = to_affine_nd(sr, affine_np)

            # Set up "labels" such that LPS tensors are handled correctly by default
            if (
                self.labels is None
                and "space" in data_array.meta
                and SpaceKeys(data_array.meta["space"]) == SpaceKeys.LPS
            ):
                labels = (("R", "L"), ("A", "P"), ("I", "S"))  # value for LPS

        else:
            warnings.warn("`data_array` is not of type `MetaTensor, assuming affine to be identity.")
            # default to identity
            affine_np = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)

        src = nib.io_orientation(affine_)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            if self.axcodes is None:
                raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
            if sr < len(self.axcodes):
                warnings.warn(
                    f"axcodes ('{self.axcodes}') length is smaller than number of input spatial dimensions D={sr}.\n"
                    f"{self.__class__.__name__}: spatial shape = {spatial_shape}, channels = {data_array.shape[0]},"
                    "please make sure the input is in the channel-first format."
                )
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=labels)
            if len(dst) < sr:
                raise ValueError(
                    f"axcodes must match data_array spatially, got axcodes={len(self.axcodes)}D data_array={sr}D"
                )
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        lazy_ = self.lazy if lazy is None else lazy
        return orientation(data_array, affine_np, spatial_ornt, lazy=lazy_, transform_info=self.get_transform_info())

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # Create inverse transform
        orig_affine = transform[TraceKeys.EXTRA_INFO]["original_affine"]
        labels = self.labels

        # Set up "labels" such that LPS tensors are handled correctly by default
        if (
            isinstance(data, MetaTensor)
            and self.labels is None
            and "space" in data.meta
            and SpaceKeys(data.meta["space"]) == SpaceKeys.LPS
        ):
            labels = (("R", "L"), ("A", "P"), ("I", "S"))  # value for LPS

        orig_axcodes = nib.orientations.aff2axcodes(orig_affine, labels=labels)
        inverse_transform = Orientation(axcodes=orig_axcodes, as_closest_canonical=False, labels=labels)
        # Apply inverse
        with inverse_transform.trace_transform(False):
            data = inverse_transform(data)

        return data


class Flip(InvertibleTransform, LazyTransform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    See `torch.flip` documentation for additional details:
    https://pytorch.org/docs/stable/generated/torch.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, spatial_axis: Sequence[int] | int | None = None, lazy: bool = False) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.spatial_axis = spatial_axis

    def __call__(self, img: torch.Tensor, lazy: bool | None = None) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ])
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        lazy_ = self.lazy if lazy is None else lazy
        return flip(img, self.spatial_axis, lazy=lazy_, transform_info=self.get_transform_info())  # type: ignore

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        self.pop_transform(data)
        flipper = Flip(spatial_axis=self.spatial_axis)
        with flipper.trace_transform(False):
            return flipper(data)


class Resize(InvertibleTransform, LazyTransform):
    """
    Resize the input image to given spatial size (with scaling, not cropping/padding).
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        spatial_size: Sequence[int] | int,
        size_mode: str = "all",
        mode: str = InterpolateMode.AREA,
        align_corners: bool | None = None,
        anti_aliasing: bool = False,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
        dtype: DtypeLike | torch.dtype = torch.float32,
        lazy: bool = False,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.size_mode = look_up_option(size_mode, ["all", "longest"])
        self.spatial_size = spatial_size
        self.mode = mode
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        align_corners: bool | None = None,
        anti_aliasing: bool | None = None,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
        dtype: DtypeLike | torch.dtype = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            anti_aliasing: bool, optional
                Whether to apply a Gaussian filter to smooth the image prior
                to downsampling. It is crucial to filter when downsampling
                the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
            anti_aliasing_sigma: {float, tuple of floats}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1. For the up-size case, s < 1, no
                anti-aliasing is performed prior to rescaling.
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

        """
        anti_aliasing = self.anti_aliasing if anti_aliasing is None else anti_aliasing
        anti_aliasing_sigma = self.anti_aliasing_sigma if anti_aliasing_sigma is None else anti_aliasing_sigma

        input_ndim = get_spatial_ndim(img)
        if self.size_mode == "all":
            output_ndim = len(ensure_tuple(self.spatial_size))
            if output_ndim > input_ndim:
                input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
                img = img.reshape(input_shape)
                if isinstance(img, MetaTensor):
                    img.spatial_ndim = output_ndim
            elif output_ndim < input_ndim:
                raise ValueError(
                    "len(spatial_size) must be greater or equal to img spatial dimensions, "
                    f"got spatial_size={output_ndim} img={input_ndim}."
                )
            _sp = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
            sp_size = fall_back_tuple(self.spatial_size, _sp)
        else:  # for the "longest" mode
            img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
            if not isinstance(self.spatial_size, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
            scale = self.spatial_size / max(img_size)
            sp_size = tuple(int(round(s * scale)) for s in img_size)

        _mode = self.mode if mode is None else mode
        _align_corners = self.align_corners if align_corners is None else align_corners
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        lazy_ = self.lazy if lazy is None else lazy
        return resize(  # type: ignore
            img,
            tuple(int(_s) for _s in sp_size),
            _mode,
            _align_corners,
            _dtype,
            input_ndim,
            anti_aliasing,
            anti_aliasing_sigma,
            lazy_,
            self.get_transform_info(),
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        orig_size = transform[TraceKeys.ORIG_SIZE]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        dtype = transform[TraceKeys.EXTRA_INFO]["dtype"]
        xform = Resize(
            spatial_size=orig_size,
            mode=mode,
            align_corners=None if align_corners == TraceKeys.NONE else align_corners,
            dtype=dtype,
        )
        with xform.trace_transform(False):
            data = xform(data)
        for _ in range(transform[TraceKeys.EXTRA_INFO]["new_dim"]):
            data = data.squeeze(-1)  # remove the additional dims
        return data


class Rotate(InvertibleTransform, LazyTransform):
    """
    Rotates an input image by given angle using :py:class:`monai.networks.layers.AffineTransform`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        rotate_order: for 3D inputs, the order in which the axes are rotated about, following the convention of
            :py:func:`scipy.spatial.transform.Rotation.from_euler`. See
            :py:func:`monai.transforms.utils.create_rotate`. Defaults to ``"XYZ"`` (the legacy behaviour).
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        angle: Sequence[float] | float,
        keep_size: bool = True,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike | torch.dtype = torch.float32,
        lazy: bool = False,
        rotate_order: str = "XYZ",
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.angle = angle
        self.keep_size = keep_size
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype
        self.rotate_order = rotate_order

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Raises:
            ValueError: When ``img`` spatially is not one of [2D, 3D].

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        _mode = mode or self.mode
        _padding_mode = padding_mode or self.padding_mode
        _align_corners = self.align_corners if align_corners is None else align_corners
        im_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        output_shape = im_shape if self.keep_size else None
        lazy_ = self.lazy if lazy is None else lazy
        return rotate(  # type: ignore
            img,
            self.angle,
            output_shape,
            _mode,
            _padding_mode,
            _align_corners,
            _dtype,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
            rotate_order=self.rotate_order,
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        fwd_rot_mat = transform[TraceKeys.EXTRA_INFO]["rot_mat"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        dtype = transform[TraceKeys.EXTRA_INFO]["dtype"]
        inv_rot_mat = linalg_inv(convert_to_numpy(fwd_rot_mat))

        _, _m, _p, _ = resolves_modes(mode, padding_mode)
        xform = AffineTransform(
            normalized=False,
            mode=_m,
            padding_mode=_p,
            align_corners=False if align_corners == TraceKeys.NONE else align_corners,
            reverse_indexing=True,
        )
        img_t: torch.Tensor = convert_data_type(data, MetaTensor, dtype=dtype)[0]
        transform_t, *_ = convert_to_dst_type(inv_rot_mat, img_t)
        sp_size = transform[TraceKeys.ORIG_SIZE]
        out: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=sp_size).float().squeeze(0)
        out = convert_to_dst_type(out, dst=data, dtype=out.dtype)[0]
        if isinstance(out, MetaTensor):
            affine = convert_to_tensor(out.peek_pending_affine(), track_meta=False)
            # Use affine matrix shape directly (not spatial_ndim) because the affine may be
            # larger than the spatial dimensions (e.g., 4x4 for 2D data), and we need to match
            # the actual affine matrix rank being composed
            mat = to_affine_nd(len(affine) - 1, transform_t)
            out.affine @= convert_to_dst_type(mat, affine)[0]
        return out


class Zoom(InvertibleTransform, LazyTransform):
    """
    Zooms an ND image using :py:class:`torch.nn.functional.interpolate`.
    For details, please see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

    Different from :py:class:`monai.transforms.resize`, this transform takes scaling factors
    as input, and provides an option of preserving the input spatial size.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"edge"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (padding/slicing if needed), default is True.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        zoom: Sequence[float] | float,
        mode: str = InterpolateMode.AREA,
        padding_mode: str = NumpyPadMode.EDGE,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = torch.float32,
        keep_size: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.zoom = zoom
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype
        self.keep_size = keep_size
        self.kwargs = kwargs

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
                ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"edge"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        _zoom = ensure_tuple_rep(self.zoom, get_spatial_ndim(img))
        _mode = self.mode if mode is None else mode
        _padding_mode = padding_mode or self.padding_mode
        _align_corners = self.align_corners if align_corners is None else align_corners
        _dtype = get_equivalent_dtype(dtype or self.dtype or img.dtype, torch.Tensor)
        lazy_ = self.lazy if lazy is None else lazy
        return zoom(  # type: ignore
            img,
            _zoom,
            self.keep_size,
            _mode,
            _padding_mode,
            _align_corners,
            _dtype,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
            **self.kwargs,
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        if transform[TraceKeys.EXTRA_INFO]["do_padcrop"]:
            orig_size = transform[TraceKeys.ORIG_SIZE]
            pad_or_crop = ResizeWithPadOrCrop(spatial_size=orig_size, mode="edge")
            padcrop_xform = transform[TraceKeys.EXTRA_INFO]["padcrop"]
            padcrop_xform[TraceKeys.EXTRA_INFO]["pad_info"][TraceKeys.ID] = TraceKeys.NONE
            padcrop_xform[TraceKeys.EXTRA_INFO]["crop_info"][TraceKeys.ID] = TraceKeys.NONE
            # this uses inverse because spatial_size // 2 in the forward pass of center crop may cause issues
            data = pad_or_crop.inverse_transform(data, padcrop_xform)  # type: ignore
        # Create inverse transform
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        dtype = transform[TraceKeys.EXTRA_INFO]["dtype"]
        inverse_transform = Resize(spatial_size=transform[TraceKeys.ORIG_SIZE])
        # Apply inverse
        with inverse_transform.trace_transform(False):
            out = inverse_transform(
                data, mode=mode, align_corners=None if align_corners == TraceKeys.NONE else align_corners, dtype=dtype
            )
        return out


class Rotate90(InvertibleTransform, LazyTransform):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    See `torch.rot90` for additional details:
    https://pytorch.org/docs/stable/generated/torch.rot90.html#torch-rot90.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, k: int = 1, spatial_axes: tuple[int, int] = (0, 1), lazy: bool = False) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                If axis is negative it counts from the last to the first axis.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        LazyTransform.__init__(self, lazy=lazy)
        self.k = (4 + (k % 4)) % 4  # 0, 1, 2, 3
        spatial_axes_: tuple[int, int] = ensure_tuple(spatial_axes)
        if len(spatial_axes_) != 2:
            raise ValueError(f"spatial_axes must be 2 numbers to define the plane to rotate, got {spatial_axes_}.")
        self.spatial_axes = spatial_axes_

    def __call__(self, img: torch.Tensor, lazy: bool | None = None) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        axes = map_spatial_axes(img.ndim, self.spatial_axes)
        lazy_ = self.lazy if lazy is None else lazy
        return rotate90(img, axes, self.k, lazy=lazy_, transform_info=self.get_transform_info())  # type: ignore

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        return self.inverse_transform(data, transform)

    def inverse_transform(self, data: torch.Tensor, transform) -> torch.Tensor:
        axes = transform[TraceKeys.EXTRA_INFO]["axes"]
        k = transform[TraceKeys.EXTRA_INFO]["k"]
        inv_k = 4 - k % 4
        xform = Rotate90(k=inv_k, spatial_axes=axes)
        with xform.trace_transform(False):
            return xform(data)


class RandRotate90(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Rotate90.backend

    def __init__(
        self, prob: float = 0.1, max_k: int = 3, spatial_axes: tuple[int, int] = (0, 1), lazy: bool = False
    ) -> None:
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
        """
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._rand_k = self.R.randint(self.max_k) + 1

    def __call__(self, img: torch.Tensor, randomize: bool = True, lazy: bool | None = None) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """

        if randomize:
            self.randomize()

        lazy_ = self.lazy if lazy is None else lazy
        if self._do_transform:
            xform = Rotate90(self._rand_k, self.spatial_axes, lazy=lazy_)
            out = xform(img)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta())

        self.push_transform(out, replace=True, lazy=lazy_)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        rotate_xform = xform_info[TraceKeys.EXTRA_INFO]
        return Rotate90().inverse_transform(data, rotate_xform)


class RandRotate(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    Randomly rotate the input arrays.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y). only work for 3D data.
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z). only work for 3D data.
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Rotate.backend

    def __init__(
        self,
        range_x: tuple[float, float] | float = 0.0,
        range_y: tuple[float, float] | float = 0.0,
        range_z: tuple[float, float] | float = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike | torch.dtype = np.float32,
        lazy: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.keep_size = keep_size
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ):
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``float32``.
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        if randomize:
            self.randomize()

        lazy_ = self.lazy if lazy is None else lazy
        if self._do_transform:
            ndim = len(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:])
            rotator = Rotate(
                angle=self.x if ndim == 2 else (self.x, self.y, self.z),
                keep_size=self.keep_size,
                mode=mode or self.mode,
                padding_mode=padding_mode or self.padding_mode,
                align_corners=self.align_corners if align_corners is None else align_corners,
                dtype=dtype or self.dtype or img.dtype,
                lazy=lazy_,
            )
            out = rotator(img)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        self.push_transform(out, replace=True, lazy=lazy_)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        return Rotate(0).inverse_transform(data, xform_info[TraceKeys.EXTRA_INFO])


class RandFlip(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1, spatial_axis: Sequence[int] | int | None = None, lazy: bool = False) -> None:
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis, lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def __call__(self, img: torch.Tensor, randomize: bool = True, lazy: bool | None = None) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        if randomize:
            self.randomize(None)
        lazy_ = self.lazy if lazy is None else lazy
        out = self.flipper(img, lazy=lazy_) if self._do_transform else img
        out = convert_to_tensor(out, track_meta=get_track_meta())
        self.push_transform(out, replace=True, lazy=lazy_)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        if not transform[TraceKeys.DO_TRANSFORM]:
            return data
        data.applied_operations.append(transform[TraceKeys.EXTRA_INFO])  # type: ignore
        return self.flipper.inverse(data)


class RandAxisFlip(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    Randomly select a spatial axis and flip along it.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        prob: Probability of flipping.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1, lazy: bool = False) -> None:
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self._axis: int | None = None
        self.flipper = Flip(spatial_axis=self._axis)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def randomize(self, data: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._axis = self.R.randint(get_spatial_ndim(data))

    def __call__(self, img: torch.Tensor, randomize: bool = True, lazy: bool | None = None) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ])
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        if randomize:
            self.randomize(data=img)

        lazy_ = self.lazy if lazy is None else lazy
        if self._do_transform:
            self.flipper.spatial_axis = self._axis
            out = self.flipper(img, lazy=lazy_)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta())
        self.push_transform(out, replace=True, lazy=lazy_)
        return out

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        if not transform[TraceKeys.DO_TRANSFORM]:
            return data
        flipper = Flip(spatial_axis=transform[TraceKeys.EXTRA_INFO][TraceKeys.EXTRA_INFO]["axes"])
        with flipper.trace_transform(False):
            return flipper(data)


class RandZoom(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    Randomly zooms input arrays with given probability within given zoom range.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (pad if needed), default is True.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Zoom.backend

    def __init__(
        self,
        prob: float = 0.1,
        min_zoom: Sequence[float] | float = 0.9,
        max_zoom: Sequence[float] | float = 1.1,
        mode: str = InterpolateMode.AREA,
        padding_mode: str = NumpyPadMode.EDGE,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = torch.float32,
        keep_size: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.min_zoom = ensure_tuple(min_zoom)
        self.max_zoom = ensure_tuple(max_zoom)
        if len(self.min_zoom) != len(self.max_zoom):
            raise ValueError(
                f"min_zoom and max_zoom must have same length, got {len(self.min_zoom)} and {len(self.max_zoom)}."
            )
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype
        self.keep_size = keep_size
        self.kwargs = kwargs

        self._zoom: Sequence[float] = [1.0]

    def randomize(self, img: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        _sp = get_spatial_ndim(img)
        self._zoom = [self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom)]
        if len(self._zoom) == 1:
            # to keep the spatial shape ratio, use same random zoom factor for all dims
            self._zoom = ensure_tuple_rep(self._zoom[0], _sp)
        elif len(self._zoom) == 2 and _sp > 2:
            # if 2 zoom factors provided for 3D data, use the first factor for H and W dims, second factor for D dim
            self._zoom = ensure_tuple_rep(self._zoom[0], _sp - 1) + ensure_tuple(self._zoom[-1])

    def __call__(
        self,
        img: torch.Tensor,
        mode: str | None = None,
        padding_mode: str | None = None,
        align_corners: bool | None = None,
        dtype: DtypeLike | torch.dtype = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``,
                ``"area"``}, the interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data.
            randomize: whether to execute `randomize()` function first, default to True.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        # match the spatial image dim
        if randomize:
            self.randomize(img=img)

        lazy_ = self.lazy if lazy is None else lazy
        if not self._do_transform:
            out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        else:
            xform = Zoom(
                self._zoom,
                keep_size=self.keep_size,
                mode=mode or self.mode,
                padding_mode=padding_mode or self.padding_mode,
                align_corners=self.align_corners if align_corners is None else align_corners,
                dtype=dtype or self.dtype,
                lazy=lazy_,
                **self.kwargs,
            )
            out = xform(img)
        self.push_transform(out, replace=True, lazy=lazy_)
        return out  # type: ignore

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        return Zoom(self._zoom).inverse_transform(data, xform_info[TraceKeys.EXTRA_INFO])


class AffineGrid(LazyTransform):
    """
    Affine transforms on the coordinates.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
            Defaults to no rotation.
        shear_params: shearing factors for affine matrix, take a 3D affine as example::

            [
                [1.0, params[0], params[1], 0.0],
                [params[2], 1.0, params[3], 0.0],
                [params[4], params[5], 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

            a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
        translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
            pixel/voxel relative to the center of the input image. Defaults to no translation.
        scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
            a tuple of 3 floats for 3D. Defaults to `1.0`.
        dtype: data type for the grid computation. Defaults to ``float32``.
            If ``None``, use the data type of input data (if `grid` is provided).
        device: device on which the tensor will be allocated, if a new grid is generated.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        affine: If applied, ignore the params (`rotate_params`, etc.) and use the
            supplied matrix. Should be square with each side = num of image spatial
            dimensions + 1.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        rotate_order: for 3D inputs, the order in which the axes are rotated about when building the
            rotation from ``rotate_params``, following the convention of
            :py:func:`scipy.spatial.transform.Rotation.from_euler`. See
            :py:func:`monai.transforms.utils.create_rotate`. Defaults to ``"XYZ"`` (the legacy behaviour).
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        align_corners: bool = False,
        affine: NdarrayOrTensor | None = None,
        lazy: bool = False,
        rotate_order: str = "XYZ",
    ) -> None:
        LazyTransform.__init__(self, lazy=lazy)
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.device = device
        _dtype = get_equivalent_dtype(dtype, torch.Tensor)
        self.dtype = _dtype if _dtype in (torch.float16, torch.float64, None) else torch.float32
        self.align_corners = align_corners
        self.affine = affine
        self.rotate_order = rotate_order

    def __call__(
        self, spatial_size: Sequence[int] | None = None, grid: torch.Tensor | None = None, lazy: bool | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        The grid can be initialized with a `spatial_size` parameter, or provided directly as `grid`.
        Therefore, either `spatial_size` or `grid` must be provided.
        When initialising from `spatial_size`, the backend "torch" will be used.

        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        lazy_ = self.lazy if lazy is None else lazy
        _device: torch.device | None

        if not lazy_:
            if grid is None:  # create grid from spatial_size
                if spatial_size is None:
                    raise ValueError("Incompatible values: grid=None and spatial_size=None.")
                grid_ = create_grid(spatial_size, device=self.device, backend="torch", dtype=self.dtype)
            else:
                grid_ = grid
            _dtype = self.dtype or grid_.dtype
            grid_: torch.Tensor = convert_to_tensor(grid_, dtype=_dtype, track_meta=get_track_meta())  # type: ignore
            _device = torch.device(grid_.device)  # type: ignore
            spatial_dims = len(grid_.shape) - 1
        else:
            _device = self.device  # type: ignore[assignment]
            spatial_dims = len(spatial_size)  # type: ignore
        _b = TransformBackends.TORCH
        affine: torch.Tensor
        if self.affine is None:
            affine = torch.eye(spatial_dims + 1, device=_device)
            if self.rotate_params:
                affine @= create_rotate(spatial_dims, self.rotate_params, device=_device, backend=_b, rotate_order=self.rotate_order)  # type: ignore[assignment]
            if self.shear_params:
                affine @= create_shear(spatial_dims, self.shear_params, device=_device, backend=_b)  # type: ignore[assignment]
            if self.translate_params:
                affine @= create_translate(spatial_dims, self.translate_params, device=_device, backend=_b)  # type: ignore[assignment]
            if self.scale_params:
                affine @= create_scale(spatial_dims, self.scale_params, device=_device, backend=_b)  # type: ignore[assignment]
        else:
            affine = self.affine  # type: ignore
        affine = to_affine_nd(spatial_dims, affine)
        if lazy_:
            return None, affine

        affine = convert_to_tensor(affine, device=grid_.device, dtype=grid_.dtype, track_meta=False)  # type: ignore
        if self.align_corners:
            sc = create_scale(
                spatial_dims, [max(d, 2) / (max(d, 2) - 1) for d in grid_.shape[1:]], device=_device, backend=_b
            )
            sc = convert_to_dst_type(sc, affine)[0]
            grid_ = ((affine @ sc) @ grid_.view((grid_.shape[0], -1))).view([-1] + list(grid_.shape[1:]))
        else:
            grid_ = (affine @ grid_.view((grid_.shape[0], -1))).view([-1] + list(grid_.shape[1:]))
        return grid_, affine  # type: ignore[return-value]


class RandAffineGrid(Randomizable, LazyTransform):
    """
    Generate randomised affine grid.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = AffineGrid.backend

    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        translate_relative: bool = False,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
