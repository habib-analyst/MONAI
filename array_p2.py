                `uniform[rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxels to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            translate_relative: if True, `translate_range` values are interpreted as fractions of the
                corresponding spatial dimension size (e.g. 0.5 allows translating up to half the image
                size), instead of absolute voxels. Defaults to False.
            device: device to store the output grid data.
            dtype: data type for the grid computation. Defaults to ``np.float32``.
                If ``None``, use the data type of input data (if `grid` is provided).
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`

        """
        LazyTransform.__init__(self, lazy=lazy)
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)
        self.translate_relative = translate_relative

        self.rotate_params: list[float] | None = None
        self.shear_params: list[float] | None = None
        self.translate_params: list[float] | None = None
        self.scale_params: list[float] | None = None

        self.device = device
        self.dtype = dtype
        self.affine: torch.Tensor | None = torch.eye(4, dtype=torch.float64)

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError(f"If giving range as [min,max], should have 2 elements per dim, got {f}.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Any | None = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(
        self,
        spatial_size: Sequence[int] | None = None,
        grid: NdarrayOrTensor | None = None,
        randomize: bool = True,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            randomize: boolean as to whether the grid parameters governing the grid should be randomized.
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        if randomize:
            self.randomize()
        lazy_ = self.lazy if lazy is None else lazy
        translate_params = self.translate_params
        if self.translate_relative and translate_params is not None:
            # interpret the sampled parameters as fractions of the spatial dims
            sp_size = spatial_size if spatial_size is not None else (tuple(grid.shape[1:]) if grid is not None else None)
            if sp_size is not None:
                translate_params = [p * d for p, d in zip(translate_params, sp_size)]
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=translate_params,
            scale_params=self.scale_params,
            device=self.device,
            dtype=self.dtype,
            lazy=lazy_,
        )
        if lazy_:  # return the affine only, don't construct the grid
            self.affine = affine_grid(spatial_size, grid)[1]  # type: ignore
            return None  # type: ignore
        _grid: torch.Tensor
        _grid, self.affine = affine_grid(spatial_size, grid)  # type: ignore
        return _grid

    def get_transformation_matrix(self) -> torch.Tensor | None:
        """Get the most recently applied transformation matrix"""
        return self.affine


class RandDeformGrid(Randomizable, Transform):
    """
    Generate random deformation grid.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self, spacing: Sequence[float] | float, magnitude_range: tuple[float, float], device: torch.device | None = None
    ) -> None:
        """
        Args:
            spacing: spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range: the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            device: device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range

        self.rand_mag = 1.0
        self.random_offset: np.ndarray
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        self.random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size))).astype(np.float32, copy=False)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]) -> torch.Tensor:
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing, device=self.device, backend="torch")
        self.randomize(control_grid.shape[1:])
        _offset, *_ = convert_to_dst_type(self.rand_mag * self.random_offset, control_grid)
        control_grid[: len(spatial_size)] += _offset
        return control_grid  # type: ignore


class Resample(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        norm_coords: bool = True,
        device: torch.device | None = None,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://monai.readthedocs.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://monai.readthedocs.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            norm_coords: whether to normalize the coordinates from `[-(size-1)/2, (size-1)/2]` to
                `[0, size - 1]` (for ``monai/csrc`` implementation) or
                `[-1, 1]` (for torch ``grid_sample`` implementation) to be compatible with the underlying
                resampling API.
            device: device on which the tensor will be allocated.
            align_corners: Defaults to False.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.

        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm_coords = norm_coords
        self.device = device
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        grid: torch.Tensor | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        dtype: DtypeLike = None,
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
                if ``norm_coords`` is True, the grid values must be in `[-(size-1)/2, (size-1)/2]`.
                if ``USE_COMPILED=True`` and ``norm_coords=False``, grid values must be in `[0, size-1]`.
                if ``USE_COMPILED=False`` and ``norm_coords=False``, grid values must be in `[-1, 1]`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://monai.readthedocs.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://monai.readthedocs.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                To be compatible with other modules, the output data type is always `float32`.
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

        See also:
            :py:const:`monai.config.USE_COMPILED`
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if grid is None:
            return img

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        _dtype = dtype or self.dtype or img.dtype
        _align_corners = self.align_corners if align_corners is None else align_corners
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=_dtype, device=_device)
        sr = min(len(img_t.peek_pending_shape() if isinstance(img_t, MetaTensor) else img_t.shape[1:]), 3)
        _use_compiled = USE_COMPILED and not _compiled_unsupported(img_t.device)
        backend, _interp_mode, _padding_mode, _ = resolves_modes(
            self.mode if mode is None else mode,
            self.padding_mode if padding_mode is None else padding_mode,
            backend=None,
            use_compiled=_use_compiled,
        )

        if _use_compiled or backend == TransformBackends.NUMPY:
            grid_t, *_ = convert_to_dst_type(grid[:sr], img_t, dtype=grid.dtype, wrap_sequence=True)
            if isinstance(grid, torch.Tensor) and grid_t.data_ptr() == grid.data_ptr():
                grid_t = grid_t.clone(memory_format=torch.contiguous_format)
            for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                _dim = max(2, dim)
                t = (_dim - 1) / 2.0
                if self.norm_coords:
                    grid_t[i] = ((_dim - 1) / _dim) * grid_t[i] + t if _align_corners else grid_t[i] + t
                elif _align_corners:
                    grid_t[i] = ((_dim - 1) / _dim) * (grid_t[i] + 0.5)
            if _use_compiled and backend == TransformBackends.TORCH:  # compiled is using torch backend param name
                grid_t = moveaxis(grid_t, 0, -1)  # type: ignore
                out = grid_pull(
                    img_t.unsqueeze(0),
                    grid_t.unsqueeze(0).to(img_t),
                    bound=_padding_mode,
                    extrapolate=True,
                    interpolation=_interp_mode,
                )[0]
            elif backend == TransformBackends.NUMPY:
                is_cuda = img_t.is_cuda
                img_np = (convert_to_cupy if is_cuda else convert_to_numpy)(img_t, wrap_sequence=True)
                grid_np, *_ = convert_to_dst_type(grid_t, img_np, dtype=grid_t.dtype, wrap_sequence=True)
                _map_coord = (cupy_ndi if is_cuda else np_ndi).map_coordinates
                out = (cupy if is_cuda else np).stack(
                    [_map_coord(c, grid_np, order=_interp_mode, mode=_padding_mode) for c in img_np]
                )
                out = convert_to_dst_type(out, img_t)[0]
            else:
                # Fallback to PyTorch grid_sample when compiled extension is unsupported.
                # Convert grid coordinates from compiled convention [0, size-1] to PyTorch [-1, 1]
                for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                    _dim = max(2, dim)
                    grid_t[i] = (grid_t[i] * 2.0 / _dim) - 1.0
                grid_t = moveaxis(grid_t, 0, -1)  # type: ignore
                out = torch.nn.functional.grid_sample(
                    img_t.unsqueeze(0),
                    grid_t.unsqueeze(0),
                    mode=_interp_mode,
                    padding_mode=_padding_mode,
                    align_corners=None if _align_corners == TraceKeys.NONE else _align_corners,  # type: ignore
                )[0]
        else:
            grid_t = moveaxis(grid[list(range(sr - 1, -1, -1))], 0, -1)  # type: ignore
            grid_t = convert_to_dst_type(grid_t, img_t, wrap_sequence=True)[0].unsqueeze(0)
            if isinstance(grid, torch.Tensor) and grid_t.data_ptr() == grid.data_ptr():
                grid_t = grid_t.clone(memory_format=torch.contiguous_format)
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[sr + 1 : 0 : -1]):
                    grid_t[0, ..., i] *= 2.0 / max(2, dim)
            out = torch.nn.functional.grid_sample(
                img_t.unsqueeze(0),
                grid_t,
                mode=_interp_mode,
                padding_mode=_padding_mode,
                align_corners=None if _align_corners == TraceKeys.NONE else _align_corners,  # type: ignore
            )[0]
        out_val, *_ = convert_to_dst_type(out, dst=img, dtype=np.float32)
        return out_val


class Affine(InvertibleTransform, LazyTransform):
    """
    Transform ``img`` given the affine parameters.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Note:
        This transform assumes that the origin of the coordinate system is at the spatial center
        of the image. When applying transformations (rotation, scaling, etc.), they are performed
        relative to this center point. If you need transformations around a different origin,
        you may need to compose this transform with translation operations or adjust your affine
        matrix accordingly.
    """

    backend = list(set(AffineGrid.backend) & set(Resample.backend))

    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        affine: NdarrayOrTensor | None = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        normalized: bool = False,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        align_corners: bool = False,
        image_only: bool = False,
        lazy: bool = False,
        rotate_order: str = "XYZ",
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

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
            affine: If applied, ignore the params (`rotate_params`, etc.) and use the
                supplied matrix. Should be square with each side = num of image spatial
                dimensions + 1.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            normalized: indicates whether the provided `affine` matrix already includes coordinate
                normalization. Set to ``True`` if your affine matrix is designed to work with normalized
                coordinates (e.g., from image processing libraries that use normalized coordinate systems).
                Set to ``False`` (default) if your affine matrix works with pixel/voxel coordinates centered
                at the image center. When ``False``, MONAI will automatically apply the necessary coordinate
                transformations. Most users should use the default ``False``.
                See also: :py:func:`monai.networks.utils.normalize_transform`.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``float32``.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.
            align_corners: Defaults to False.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            image_only: if True return only the image volume, otherwise return (image, affine).
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False
            rotate_order: for 3D inputs, the order in which the axes are rotated about when building the rotation
                from ``rotate_params``, following the convention of
                :py:func:`scipy.spatial.transform.Rotation.from_euler`. See
                :py:func:`monai.transforms.utils.create_rotate`. Defaults to ``"XYZ"`` (the legacy behaviour).
        """
        LazyTransform.__init__(self, lazy=lazy)
        self.affine_grid = AffineGrid(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            affine=affine,
            dtype=dtype,
            align_corners=align_corners,
            device=device,
            lazy=lazy,
            rotate_order=rotate_order,
        )
        self.image_only = image_only
        self.norm_coord = not normalized
        self.resampler = Resample(norm_coords=self.norm_coord, device=device, dtype=dtype, align_corners=align_corners)
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self.affine_grid.lazy = val
        self._lazy = val

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        lazy: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, NdarrayOrTensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
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
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img_size)
        lazy_ = self.lazy if lazy is None else lazy
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        grid, affine = self.affine_grid(spatial_size=sp_size, lazy=lazy_)

        return affine_func(  # type: ignore
            img,
            affine,
            grid,
            self.resampler,
            sp_size,
            _mode,
            _padding_mode,
            True,
            self.image_only,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
        )

    @classmethod
    def compute_w_affine(cls, spatial_rank, mat, img_size, sp_size, align_corners: bool = False):
        """
        Compute the affine matrix for transforming image coordinates, accounting for
        center-based coordinate system.

        This function adjusts the provided affine transformation matrix to work with images
        where transformations are applied relative to the image center rather than the origin.
        It composes the input matrix with translation operations that shift between
        corner-based and center-based coordinate systems.

        Args:
            spatial_rank: number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
            mat: the base affine transformation matrix to be adjusted.
            img_size: spatial dimensions of the input image.
            sp_size: spatial dimensions of the output (transformed) image.
            align_corners: if True, align the corners of the initial and transformed volumes.

        Returns:
            The adjusted affine matrix that can be applied to image coordinates.
        """
        r = int(spatial_rank)
        mat = to_affine_nd(r, mat)
        shift_1 = create_translate(r, [float(d - 1) / 2 for d in img_size[:r]])
        shift_2 = create_translate(r, [-float(d - 1) / 2 for d in sp_size[:r]])
        mat = convert_data_type(mat, np.ndarray)[0]
        if align_corners:
            # Keep lazy world-affine consistent with eager sampling:
            # x_in = T_in @ S_in^-1 @ A_centered @ S_out @ T_out^-1 @ x_out
            src_scale = create_scale(r, [(max(float(d), 2.0) - 1.0) / max(float(d), 2.0) for d in img_size[:r]])
            dst_scale = create_scale(r, [max(float(d), 2.0) / (max(float(d), 2.0) - 1.0) for d in sp_size[:r]])
            src_scale = convert_data_type(src_scale, np.ndarray)[0]
            dst_scale = convert_data_type(dst_scale, np.ndarray)[0]
            mat = shift_1 @ src_scale @ mat @ dst_scale @ shift_2
        else:
            mat = shift_1 @ mat @ shift_2
        return mat

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        orig_size = transform[TraceKeys.ORIG_SIZE]
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
        inv_affine = linalg_inv(convert_to_numpy(fwd_affine))
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]

        affine_grid = AffineGrid(affine=inv_affine, align_corners=align_corners)
        grid, _ = affine_grid(orig_size)
        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode, align_corners=align_corners)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        affine = convert_data_type(out.peek_pending_affine(), torch.Tensor)[0]
        # Use affine matrix shape directly (not spatial_ndim) to ensure matrix composition compatibility
        # when affine is larger than spatial dimensions (e.g., 4x4 for 2D data)
        xform, *_ = convert_to_dst_type(
            Affine.compute_w_affine(len(affine) - 1, inv_affine, data.shape[1:], orig_size), affine
        )
        out.affine @= xform
        return out


class RandAffine(RandomizableTransform, InvertibleTransform, LazyTransform):
    """
    Random affine transform.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    backend = Affine.backend

    def __init__(
        self,
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        translate_relative: bool = False,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: torch.device | None = None,
        lazy: bool = False,
    ) -> None:
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            translate_relative: if True, `translate_range` values are interpreted as fractions of the
                corresponding spatial dimension size (e.g. 0.5 allows translating up to half the image
                size), instead of absolute voxels. Defaults to False.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``bilinear``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``reflection``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        Note:
            The affine transformations in MONAI use a 'backward mapping' (image-to-grid) logic.
            This can be counter-intuitive:
            - Translation: A positive value shifts the image in the negative direction.
            - Scaling: Positive scale_range values decrease the image size; values in [-1, 0) increase it.
            - Rotation: The direction (CW/CCW) may vary depending on the axis.

        """
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            translate_relative=translate_relative,
            device=device,
            lazy=lazy,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.cache_grid = cache_grid
        self._cached_grid = self._init_identity_cache(lazy)
        self.mode = mode
        self.padding_mode: str = padding_mode

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.rand_affine_grid.lazy = val

    def _init_identity_cache(self, lazy: bool):
        """
        Create cache of the identity grid if cache_grid=True and spatial_size is known.
        """
        if lazy:
            return None
        if self.spatial_size is None:
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size, please specify 'spatial_size'."
                )
            return None
        _sp_size = ensure_tuple(self.spatial_size)
        _ndim = len(_sp_size)
        if _sp_size != fall_back_tuple(_sp_size, [1] * _ndim) or _sp_size != fall_back_tuple(_sp_size, [2] * _ndim):
            # dynamic shape because it falls back to different outcomes
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size "
                    f"'spatial_size={self.spatial_size}', please specify 'spatial_size'."
                )
            return None
        return create_grid(spatial_size=_sp_size, device=self.rand_affine_grid.device, backend="torch")

    def get_identity_grid(self, spatial_size: Sequence[int], lazy: bool):
        """
        Return a cached or new identity grid depends on the availability.

        Args:
            spatial_size: non-dynamic spatial size
        """
        if lazy:
            return None
        ndim = len(spatial_size)
        if spatial_size != fall_back_tuple(spatial_size, [1] * ndim) or spatial_size != fall_back_tuple(
            spatial_size, [2] * ndim
        ):
            raise RuntimeError(f"spatial_size should not be dynamic, got {spatial_size}.")
        return (
            create_grid(spatial_size=spatial_size, device=self.rand_affine_grid.device, backend="torch")
            if self._cached_grid is None
            else self._cached_grid
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAffine:
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
        grid=None,
        lazy: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
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
            randomize: whether to execute `randomize()` function first, default to True.
            grid: precomputed grid to be used (mainly to accelerate `RandAffined`).
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.
        """
        if randomize:
            self.randomize()
        # if not doing transform and spatial size doesn't change, nothing to do
        # except convert to float and device
        ori_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, ori_size)
        do_resampling = self._do_transform or (sp_size != ensure_tuple(ori_size))
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        lazy_ = self.lazy if lazy is None else lazy
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if lazy_:
            if self._do_transform:
                if grid is None:
                    self.rand_affine_grid(sp_size, randomize=randomize, lazy=True)
                affine = self.rand_affine_grid.get_transformation_matrix()
            else:
                affine = convert_to_dst_type(torch.eye(len(sp_size) + 1), img, dtype=self.rand_affine_grid.dtype)[0]
        else:
            if grid is None:
                grid = self.get_identity_grid(sp_size, lazy_)
                if self._do_transform:
                    grid = self.rand_affine_grid(grid=grid, randomize=randomize, lazy=lazy_)
            affine = self.rand_affine_grid.get_transformation_matrix()
        return affine_func(  # type: ignore
            img,
            affine,
            grid,
            self.resampler,
            sp_size,
            _mode,
            _padding_mode,
            do_resampling,
            True,
            lazy=lazy_,
            transform_info=self.get_transform_info(),
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        # if transform was not performed nothing to do.
        if not transform[TraceKeys.EXTRA_INFO]["do_resampling"]:
            return data
        orig_size = transform[TraceKeys.ORIG_SIZE]
        orig_size = fall_back_tuple(orig_size, data.shape[1:])
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        inv_affine = linalg_inv(convert_to_numpy(fwd_affine))
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]
        affine_grid = AffineGrid(affine=inv_affine)
        grid, _ = affine_grid(orig_size)

        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        affine = convert_data_type(out.peek_pending_affine(), torch.Tensor)[0]
        # Use affine matrix shape directly (not spatial_ndim) to ensure matrix composition compatibility
        # when affine is larger than spatial dimensions (e.g., 4x4 for 2D data)
        xform, *_ = convert_to_dst_type(
            Affine.compute_w_affine(len(affine) - 1, inv_affine, data.shape[1:], orig_size), affine
        )
        out.affine @= xform
        return out


class Rand2DElastic(RandomizableTransform):
    """
    Random elastic deformation and affine in 2D.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Resample.backend

    def __init__(
        self,
        spacing: tuple[float, float] | float,
        magnitude_range: tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: tuple[int, int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            spacing : distance in between the control points.
            magnitude_range: the random offsets will be generated from ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D) for affine matrix, take a 2D affine as example::

                    [
                        [1.0, params[0], 0.0],
                        [params[1], 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        RandomizableTransform.__init__(self, prob)
        self.deform_grid = RandDeformGrid(spacing=spacing, magnitude_range=magnitude_range, device=device)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
            lazy=False,
        )
        self.resampler = Resample(device=device)

        self.device = device
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand2DElastic:
        self.deform_grid.set_random_state(seed, state)
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def set_device(self, device):
        self.deform_grid.device = device
        self.rand_affine_grid.device = device
        self.resampler.device = device
        self.device = device

    def randomize(self, spatial_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.deform_grid.randomize(spatial_size)
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: tuple[int, int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W),
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
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
            randomize: whether to execute `randomize()` function first, default to True.
        """
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(spatial_size=sp_size)

        if self._do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = self.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(
                recompute_scale_factor=True,
                input=grid.unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            _device = img.device if isinstance(img, torch.Tensor) else self.device
            grid = cast(torch.Tensor, create_grid(spatial_size=sp_size, device=_device, backend="torch"))
        out: torch.Tensor = self.resampler(
            img,
            grid,
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        return out


class Rand3DElastic(RandomizableTransform):
    """
    Random elastic deformation and affine in 3D.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Resample.backend

    def __init__(
        self,
        sigma_range: tuple[float, float],
        magnitude_range: tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: tuple[int, int, int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 6 floats for 3D) for affine matrix, take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        RandomizableTransform.__init__(self, prob)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
            lazy=False,
        )
        self.resampler = Resample(device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode
        self.device = device

        self.rand_offset: np.ndarray
        self.magnitude = 1.0
        self.sigma = 1.0

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand3DElastic:
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def set_device(self, device):
        self.rand_affine_grid.device = device
        self.resampler.device = device
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32, copy=False)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: tuple[int, int, int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W, D),
            spatial_size: specifying spatial 3D output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
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
            randomize: whether to execute `randomize()` function first, default to True.
        """
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(grid_size=sp_size)

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        grid = create_grid(spatial_size=sp_size, device=_device, backend="torch")
        if self._do_transform:
            if self.rand_offset is None:
                raise RuntimeError("rand_offset is not initialized.")
            gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=_device)
            offset = torch.as_tensor(self.rand_offset, device=_device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        out: torch.Tensor = self.resampler(
            img,
            grid,  # type: ignore
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        return out


class GridDistortion(Transform):
    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        num_cells: tuple[int] | int,
        distort_steps: Sequence[Sequence[float]],
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
    ) -> None:
        """
        Grid distortion transform. Refer to:
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py

        Args:
            num_cells: number of grid cells on each dimension.
            distort_steps: This argument is a list of tuples, where each tuple contains the distort steps of the
                corresponding dimensions (in the order of H, W[, D]). The length of each tuple equals to `num_cells + 1`.
                Each value in the tuple represents the distort step of the related cell.
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
            device: device on which the tensor will be allocated.

        """
        self.resampler = Resample(mode=mode, padding_mode=padding_mode, device=device)
        self.num_cells = num_cells
        self.distort_steps = distort_steps
        self.device = device

    def __call__(
        self,
        img: torch.Tensor,
        distort_steps: Sequence[Sequence] | None = None,
        mode: str | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            distort_steps: This argument is a list of tuples, where each tuple contains the distort steps of the
                corresponding dimensions (in the order of H, W[, D]). The length of each tuple equals to `num_cells + 1`.
                Each value in the tuple represents the distort step of the related cell.
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

        """
        distort_steps = self.distort_steps if distort_steps is None else distort_steps
        if len(img.shape) != len(distort_steps) + 1:
            raise ValueError("the spatial size of `img` does not match with the length of `distort_steps`")

        all_ranges = []
        _sp = get_spatial_ndim(img)
        num_cells = ensure_tuple_rep(self.num_cells, _sp)
        if isinstance(img, MetaTensor) and img.pending_operations:
            warnings.warn("MetaTensor img has pending operations, transform may return incorrect results.")
        for dim_idx, dim_size in enumerate(img.shape[1 : 1 + _sp]):
            dim_distort_steps = distort_steps[dim_idx]
            ranges = torch.zeros(dim_size, dtype=torch.float32)
            cell_size = dim_size // num_cells[dim_idx]
            prev = 0
            for idx in range(num_cells[dim_idx] + 1):
                start = int(idx * cell_size)
                end = start + cell_size
                if end > dim_size:
                    end = dim_size
                    cur = dim_size
                else:
                    cur = prev + cell_size * dim_distort_steps[idx]
                ranges[start:end] = torch.linspace(prev, cur, end - start)
                prev = cur
            ranges = ranges - (dim_size - 1.0) / 2.0
            all_ranges.append(ranges)

        coords = meshgrid_ij(*all_ranges)
        grid = torch.stack([*coords, torch.ones_like(coords[0])])

        return self.resampler(img, grid=grid, mode=mode, padding_mode=padding_mode)


class RandGridDistortion(RandomizableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        num_cells: tuple[int] | int = 5,
        prob: float = 0.1,
        distort_limit: tuple[float, float] | float = (-0.03, 0.03),
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
    ) -> None:
        """
        Random grid distortion transform. Refer to:
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py

        Args:
            num_cells: number of grid cells on each dimension.
            prob: probability of returning a randomized grid distortion transform. Defaults to 0.1.
            distort_limit: range to randomly distort.
                If single number, distort_limit is picked from (-distort_limit, distort_limit).
                Defaults to (-0.03, 0.03).
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
            device: device on which the tensor will be allocated.

        """
        RandomizableTransform.__init__(self, prob)
        self.num_cells = num_cells
        if isinstance(distort_limit, (int, float)):
            self.distort_limit = (min(-distort_limit, distort_limit), max(-distort_limit, distort_limit))
        else:
            self.distort_limit = (min(distort_limit), max(distort_limit))
        self.distort_steps: Sequence[Sequence[float]] = ((1.0,),)
        self.grid_distortion = GridDistortion(
            num_cells=num_cells, distort_steps=self.distort_steps, mode=mode, padding_mode=padding_mode, device=device
        )

    def randomize(self, spatial_shape: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return
        self.distort_steps = tuple(
            tuple(1.0 + self.R.uniform(low=self.distort_limit[0], high=self.distort_limit[1], size=n_cells + 1))
            for n_cells in ensure_tuple_rep(self.num_cells, len(spatial_shape))
        )

    def __call__(
        self, img: torch.Tensor, mode: str | None = None, padding_mode: str | None = None, randomize: bool = True
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
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
            randomize: whether to shuffle the random factors using `randomize()`, default to True.
        """
        if randomize:
            if isinstance(img, MetaTensor) and img.pending_operations:
                warnings.warn("MetaTensor img has pending operations, transform may return incorrect results.")
            self.randomize(img.shape[1:])
        if not self._do_transform:
            return convert_to_tensor(img, track_meta=get_track_meta())  # type: ignore
        return self.grid_distortion(img, distort_steps=self.distort_steps, mode=mode, padding_mode=padding_mode)


class GridSplit(Transform, MultiSampleTrait):
    """
    Split the image into patches based on the provided grid in 2D.

    Args:
        grid: a tuple define the shape of the grid upon which the image is split. Defaults to (2, 2)
        size: a tuple or an integer that defines the output patch sizes.
            If it's an integer, the value will be repeated for each dimension.
            The default is None, where the patch size will be inferred from the grid shape.

    Example:
        Given an image (torch.Tensor or numpy.ndarray) with size of (3, 10, 10) and a grid of (2, 2),
        it will return a Tensor or array with the size of (4, 3, 5, 5).
        Here, if the `size` is provided, the returned shape will be (4, 3, size, size)

    Note: This transform currently support only image with two spatial dimensions.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, grid: tuple[int, int] = (2, 2), size: int | tuple[int, int] | None = None):
        # Grid size
        self.grid = grid

        # Patch size
        self.size = None if size is None else ensure_tuple_rep(size, len(self.grid))

    def __call__(
        self, image: NdarrayOrTensor, size: int | tuple[int, int] | np.ndarray | None = None
    ) -> list[NdarrayOrTensor]:
        input_size = self.size if size is None else ensure_tuple_rep(size, len(self.grid))

        if self.grid == (1, 1) and input_size is None:
            return [image]
        if isinstance(image, MetaTensor) and image.pending_operations:
            warnings.warn("MetaTensor img has pending operations, transform may return incorrect results.")
        split_size, steps = self._get_params(image.shape[1:], input_size)
        patches: list[NdarrayOrTensor]
        as_strided_func: Callable
        if isinstance(image, torch.Tensor):
            as_strided_func = torch.as_strided
            c_stride, x_stride, y_stride = image.stride()
        elif isinstance(image, np.ndarray):
            as_strided_func = np.lib.stride_tricks.as_strided
            c_stride, x_stride, y_stride = image.strides
        else:
            raise ValueError(f"Input type [{type(image)}] is not supported.")

        x_step, y_step = steps
        n_channels = image.shape[0]
        strided_image = as_strided_func(
            image,
            (*self.grid, n_channels, split_size[0], split_size[1]),
            (x_stride * x_step, y_stride * y_step, c_stride, x_stride, y_stride),
        )
        # Flatten the first two dimensions
        strided_image = strided_image.reshape(-1, *strided_image.shape[2:])
        # Make a list of contiguous patches
        if isinstance(image, torch.Tensor):
            patches = [p.contiguous() for p in strided_image]
        elif isinstance(image, np.ndarray):
            patches = [np.ascontiguousarray(p) for p in strided_image]

        return patches

    def _get_params(self, image_size: Sequence[int] | np.ndarray, size: Sequence[int] | np.ndarray | None = None):
        """
        Calculate the size and step required for splitting the image
        Args:
            The size of the input image
        """
        if size is None:
            # infer each sub-image size from the image size and the grid
            size = tuple(image_size[i] // self.grid[i] for i in range(len(self.grid)))

        if any(size[i] > image_size[i] for i in range(len(self.grid))):
            raise ValueError(f"The image size ({image_size})is smaller than the requested split size ({size})")

        steps = tuple(
            (image_size[i] - size[i]) // (self.grid[i] - 1) if self.grid[i] > 1 else image_size[i]
            for i in range(len(self.grid))
        )

        return size, steps


class GridPatch(Transform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps.
    It can sort the patches and return all or a subset of them.

    Args:
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        offset: offset of starting position in the array, default is 0 for each dimension.
        num_patches: number of patches (or maximum number of patches) to return.
            If the requested number of patches is greater than the number of available patches,
            padding will be applied to provide exactly `num_patches` patches unless `threshold` is set.
            When `threshold` is set, this value is treated as the maximum number of patches.
            Defaults to None, which does not limit number of the patches.
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
            lowest values (`"min"`), or in their default order (`None`). Default to None.
        threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
            Defaults to no filtering.
        pad_mode: the  mode for padding the input image by `patch_size` to include patches that cross boundaries.
            Available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function.
            Defaults to `None`, which means no padding will be applied.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            requires pytorch >= 1.10 for best compatibility.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    Returns:
        MetaTensor: the extracted patches as a single tensor (with patch dimension as the first dimension),
            with following metadata:

            - `PatchKeys.LOCATION`: the starting location of the patch in the image,
            - `PatchKeys.COUNT`: total number of patches in the image,
            - "spatial_shape": spatial size of the extracted patch, and
            - "offset": the amount of offset for the patches in the image (starting position of the first patch)

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        patch_size: Sequence[int],
        offset: Sequence[int] | None = None,
        num_patches: int | None = None,
        overlap: Sequence[float] | float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str | None = None,
        **pad_kwargs,
    ):
        self.patch_size = ensure_tuple(patch_size)
        self.offset = ensure_tuple(offset) if offset else (0,) * len(self.patch_size)
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs
        self.overlap = overlap
        self.num_patches = num_patches
        self.sort_fn = sort_fn.lower() if sort_fn else None
        self.threshold = threshold

    def filter_threshold(self, image_np: NdarrayOrTensor, locations: np.ndarray) -> tuple[NdarrayOrTensor, np.ndarray]:
        """
        Filter the patches and their locations according to a threshold.

        Args:
            image_np: a numpy.ndarray or torch.Tensor representing a stack of patches.
            locations: a numpy.ndarray representing the stack of location of each patch.

        Returns:
            tuple[NdarrayOrTensor, numpy.ndarray]:  tuple of filtered patches and locations.
        """
        n_dims = len(image_np.shape)
        idx = argwhere(image_np.sum(tuple(range(1, n_dims))) < self.threshold).reshape(-1)  # type: ignore[operator]
        idx_np = convert_data_type(idx, np.ndarray)[0]
        return image_np[idx], locations[idx_np]

    def filter_count(self, image_np: NdarrayOrTensor, locations: np.ndarray) -> tuple[NdarrayOrTensor, np.ndarray]:
        """
        Sort the patches based on the sum of their intensity, and just keep `self.num_patches` of them.

        Args:
            image_np: a numpy.ndarray or torch.Tensor representing a stack of patches.
            locations: a numpy.ndarray representing the stack of location of each patch.
        """
        if self.sort_fn is None:
            image_np = image_np[: self.num_patches]
            locations = locations[: self.num_patches]
        elif self.num_patches is not None:
            n_dims = len(image_np.shape)
            if self.sort_fn == GridPatchSort.MIN:
                idx = argsort(image_np.sum(tuple(range(1, n_dims))))
            elif self.sort_fn == GridPatchSort.MAX:
                idx = argsort(-image_np.sum(tuple(range(1, n_dims))))
            else:
                raise ValueError(f'`sort_fn` should be either "min", "max", or None! {self.sort_fn} provided!')
            idx = idx[: self.num_patches]
            idx_np = convert_data_type(idx, np.ndarray)[0]
            image_np = image_np[idx]
            locations = locations[idx_np]
        return image_np, locations

    def __call__(self, array: NdarrayOrTensor) -> MetaTensor:
        """
        Extract the patches (sweeping the entire image in a row-major sliding-window manner with possible overlaps).

        Args:
            array: a input image as `numpy.ndarray` or `torch.Tensor`

        Return:
            MetaTensor: the extracted patches as a single tensor (with patch dimension as the first dimension),
                with defined `PatchKeys.LOCATION` and `PatchKeys.COUNT` metadata.
        """
        # create the patch iterator which sweeps the image row-by-row
        patch_iterator = iter_patch(
            array,
            patch_size=(None,) + self.patch_size,  # expand to have the channel dim
            start_pos=(0,) + self.offset,  # expand to have the channel dim
            overlap=self.overlap,
            copy_back=False,
            mode=self.pad_mode,
            **self.pad_kwargs,
        )
        patches = list(zip(*patch_iterator))
        patched_image: NdarrayOrTensor
        patched_image = np.stack(patches[0]) if isinstance(array, np.ndarray) else torch.stack(patches[0])
        locations = np.stack(patches[1])[:, 1:, 0]  # only keep the starting location

        # Apply threshold filtering
        if self.threshold is not None:
            patched_image, locations = self.filter_threshold(patched_image, locations)

        # Apply count filtering
        if self.num_patches:
            # Limit number of patches
            patched_image, locations = self.filter_count(patched_image, locations)
            # Pad the patch list to have the requested number of patches
            if self.threshold is None:
                padding = self.num_patches - len(patched_image)
                if padding > 0:
                    # pad constant patches to the end of the first dim
                    constant_values = self.pad_kwargs.get("constant_values", 0)
                    padding_shape = (padding, *list(patched_image.shape)[1:])
                    constant_padding: NdarrayOrTensor
                    if isinstance(patched_image, np.ndarray):
                        constant_padding = np.full(padding_shape, constant_values, dtype=patched_image.dtype)
                        patched_image = np.concatenate([patched_image, constant_padding], axis=0)
                    else:
                        constant_padding = torch.full(
                            padding_shape,
                            constant_values,
                            dtype=patched_image.dtype,
                            layout=patched_image.layout,
                            device=patched_image.device,
                        )
                        patched_image = torch.cat([patched_image, constant_padding], dim=0)
                    locations = np.pad(locations, [[0, padding], [0, 0]], constant_values=0)

        # Convert to MetaTensor
        metadata = array.meta if isinstance(array, MetaTensor) else MetaTensor.get_default_meta()
        metadata[PatchKeys.LOCATION] = locations.T
        metadata[PatchKeys.COUNT] = len(locations)
        metadata["spatial_shape"] = np.tile(np.array(self.patch_size), (len(locations), 1)).T
        metadata["offset"] = self.offset
        output = MetaTensor(x=patched_image, meta=metadata)
        output.is_batch = True

        return output


class RandGridPatch(GridPatch, RandomizableTransform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps,
    and with random offset for the minimal corner of the image, (0,0) for 2D and (0,0,0) for 3D.
    It can sort the patches and return all or a subset of them.

    Args:
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        min_offset: the minimum range of offset to be selected randomly. Defaults to 0.
        max_offset: the maximum range of offset to be selected randomly.
            Defaults to image size modulo patch size.
        num_patches: number of patches (or maximum number of patches) to return.
            If the requested number of patches is greater than the number of available patches,
            padding will be applied to provide exactly `num_patches` patches unless `threshold` is set.
            When `threshold` is set, this value is treated as the maximum number of patches.
            Defaults to None, which does not limit number of the patches.
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
            lowest values (`"min"`), in random ("random"), or in their default order (`None`). Default to None.
        threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
            Defaults to no filtering.
        pad_mode: the  mode for padding the input image by `patch_size` to include patches that cross boundaries.
            Available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function.
            Defaults to `None`, which means no padding will be applied.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            requires pytorch >= 1.10 for best compatibility.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    Returns:
        MetaTensor: the extracted patches as a single tensor (with patch dimension as the first dimension),
            with following metadata:

            - `PatchKeys.LOCATION`: the starting location of the patch in the image,
            - `PatchKeys.COUNT`: total number of patches in the image,
            - "spatial_shape": spatial size of the extracted patch, and
            - "offset": the amount of offset for the patches in the image (starting position of the first patch)

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        patch_size: Sequence[int],
        min_offset: Sequence[int] | int | None = None,
        max_offset: Sequence[int] | int | None = None,
        num_patches: int | None = None,
        overlap: Sequence[float] | float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str | None = None,
        **pad_kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            offset=(),
            num_patches=num_patches,
            overlap=overlap,
            sort_fn=sort_fn,
            threshold=threshold,
            pad_mode=pad_mode,
            **pad_kwargs,
        )
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.num_patches = num_patches
        self.sort_fn = sort_fn

    def randomize(self, array):
        if self.min_offset is None:
            min_offset = (0,) * len(self.patch_size)
        else:
            min_offset = ensure_tuple_rep(self.min_offset, len(self.patch_size))
        if self.max_offset is None:
            max_offset = tuple(s % p for s, p in zip(array.shape[1:], self.patch_size))
        else:
            max_offset = ensure_tuple_rep(self.max_offset, len(self.patch_size))

        self.offset = tuple(self.R.randint(low=low, high=high + 1) for low, high in zip(min_offset, max_offset))

    def filter_count(self, image_np: NdarrayOrTensor, locations: np.ndarray) -> tuple[NdarrayOrTensor, np.ndarray]:
        if self.sort_fn == GridPatchSort.RANDOM:
            idx = self.R.permutation(image_np.shape[0])
            idx = idx[: self.num_patches]
            idx_np = convert_data_type(idx, np.ndarray)[0]
            image_np = image_np[idx]  # type: ignore[index]
            locations = locations[idx_np]
            return image_np, locations
        elif self.sort_fn not in (None, GridPatchSort.MIN, GridPatchSort.MAX):
            raise ValueError(f'`sort_fn` should be either "min", "max", "random" or None! {self.sort_fn} provided!')
        return super().filter_count(image_np, locations)

    def __call__(self, array: NdarrayOrTensor, randomize: bool = True):
        if randomize:
            self.randomize(array)
        return super().__call__(array)


class RandSimulateLowResolution(RandomizableTransform):
    """
    Random simulation of low resolution corresponding to nnU-Net's SimulateLowResolutionTransform
    (https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/resample_transforms.py#L23)
    First, the array/tensor is resampled at lower resolution as determined by the zoom_factor which is uniformly sampled
    from the `zoom_range`. Then, the array/tensor is resampled at the original resolution.
    """

    backend = Affine.backend

    def __init__(
        self,
        prob: float = 0.1,
        downsample_mode: InterpolateMode | str = InterpolateMode.NEAREST,
        upsample_mode: InterpolateMode | str = InterpolateMode.TRILINEAR,
        zoom_range: Sequence[float] = (0.5, 1.0),
        align_corners=False,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            prob: probability of performing this augmentation
            downsample_mode: interpolation mode for downsampling operation
            upsample_mode: interpolation mode for upsampling operation
            zoom_range: range from which the random zoom factor for the downsampling and upsampling operation is
            sampled. It determines the shape of the downsampled tensor.
            align_corners: This only has an effect when downsample_mode or upsample_mode  is 'linear', 'bilinear',
                'bicubic' or 'trilinear'. Default: False
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            device: device on which the tensor will be allocated.

        """
        RandomizableTransform.__init__(self, prob)

        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.zoom_range = zoom_range
        self.align_corners = align_corners
        self.device = device
        self.zoom_factor = 1.0

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        self.zoom_factor = self.R.uniform(self.zoom_range[0], self.zoom_range[1])
        if not self._do_transform:
            return None

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            randomize: whether to execute `randomize()` function first, defaults to True.
        """
        if randomize:
            self.randomize()

        if self._do_transform:
            input_shape = img.shape[1:]
            # Clamp each axis to at least 1 so F.interpolate never sees a zero-sized dimension.
            target_shape = tuple(max(1, int(np.round(s * self.zoom_factor))) for s in input_shape)

            # Use F.interpolate directly on a plain tensor to avoid mutating the global
            # set_track_meta flag, which is not thread-safe (see GitHub issue #8409).
            img_t = convert_to_tensor(img, track_meta=False)
            # F.interpolate requires float input and a batch dimension; cast matches
            # the default dtype=float32 that Resize uses internally.
            img_float = img_t.unsqueeze(0).to(dtype=torch.float32)

            downsample_mode = str(self.downsample_mode)
            upsample_mode = str(self.upsample_mode)
            # align_corners is only valid for linear/bilinear/bicubic/trilinear modes
            _align_corners_modes = {"linear", "bilinear", "bicubic", "trilinear"}
            downsample_align_corners = self.align_corners if downsample_mode in _align_corners_modes else None
            upsample_align_corners = self.align_corners if upsample_mode in _align_corners_modes else None

            img_downsampled = torch.nn.functional.interpolate(
                img_float, size=target_shape, mode=downsample_mode, align_corners=downsample_align_corners
            )
            img_upsampled_t = torch.nn.functional.interpolate(
                img_downsampled, size=input_shape, mode=upsample_mode, align_corners=upsample_align_corners
            ).squeeze(0)

            # copy metadata from original image to down-and-upsampled image,
            # respecting the caller's get_track_meta() setting.
            img_upsampled = cast(torch.Tensor, convert_to_tensor(img_upsampled_t, track_meta=get_track_meta()))
            if isinstance(img_upsampled, MetaTensor):
                img_upsampled.copy_meta_from(img)

            return img_upsampled

        else:
            return img


class ConvertBoxToPoints(Transform):
    """
    Converts an axis-aligned bounding box to points. It can automatically convert the boxes to the points based on the box mode.
    Bounding boxes of the shape (N, C) for N boxes. C is [x1, y1, x2, y2] for 2D or [x1, y1, z1, x2, y2, z2] for 3D for each box.
    Return shape will be (N, 4, 2) for 2D or (N, 8, 3) for 3D.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mode: str | BoxMode | type[BoxMode] | None = None) -> None:
        """
        Args:
            mode: the mode of the box, can be a string, a BoxMode instance or a BoxMode class. Defaults to StandardMode.
        """
        super().__init__()
        self.mode = StandardMode if mode is None else mode

    def __call__(self, data: Any):
        data = convert_to_tensor(data, track_meta=get_track_meta())
        points = convert_box_to_points(data, mode=self.mode)
        return convert_to_dst_type(points, data)[0]


class ConvertPointsToBoxes(Transform):
    """
    Converts points to an axis-aligned bounding box.
    Points representing the corners of the bounding box. Shape (N, 8, 3) for the 8 corners of a 3D cuboid or
    (N, 4, 2) for the 4 corners of a 2D rectangle.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Any):
        data = convert_to_tensor(data, track_meta=get_track_meta())
        box = convert_points_to_box(data)
        return convert_to_dst_type(box, data)[0]
