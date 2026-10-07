    """

    backend = Rand3DElastic.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma_range: tuple[float, float],
        magnitude_range: tuple[float, float],
        spatial_size: tuple[int, int, int] | int | None = None,
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: Sequence[tuple[float, float] | float] | float | None = None,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
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
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_3d_elastic = Rand3DElastic(
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=1.0,  # because probability controlled by this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            device=device,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Rand3DElasticd:
        self.rand_3d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)

        if first_key == ():
            out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)
        if isinstance(d[first_key], MetaTensor) and d[first_key].pending_operations:  # type: ignore
            warnings.warn(f"data['{first_key}'] has pending operations, transform may return incorrect results.")
        sp_size = fall_back_tuple(self.rand_3d_elastic.spatial_size, d[first_key].shape[1:])

        # all the keys share the same random elastic factor
        self.rand_3d_elastic.randomize(sp_size)

        device = self.rand_3d_elastic.device
        if device is None and isinstance(d[first_key], torch.Tensor):
            device = d[first_key].device
            self.rand_3d_elastic.set_device(device)
        grid = create_grid(spatial_size=sp_size, device=device, backend="torch")
        if self._do_transform:
            gaussian = GaussianFilter(spatial_dims=3, sigma=self.rand_3d_elastic.sigma, truncated=3.0).to(device)
            offset = torch.as_tensor(self.rand_3d_elastic.rand_offset, device=device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.rand_3d_elastic.magnitude
            grid = self.rand_3d_elastic.rand_affine_grid(grid=grid)

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.rand_3d_elastic.resampler(d[key], grid, mode=mode, padding_mode=padding_mode)  # type: ignore
        return d


class Flipd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Flip.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_axis: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            d[key] = self.flipper(d[key], lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.flipper.inverse(d[key])
        return d


class RandFlipd(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = Flip.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Sequence[int] | int | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = Flip(spatial_axis=spatial_axis, lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandFlipd:
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        self.randomize(None)

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key], lazy=lazy_)
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if not xform[TraceKeys.DO_TRANSFORM]:
                continue
            with self.flipper.trace_transform(False):
                d[key] = self.flipper(d[key])
        return d


class RandAxisFlipd(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAxisFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = RandAxisFlip.backend

    def __init__(
        self, keys: KeysCollection, prob: float = 0.1, allow_missing_keys: bool = False, lazy: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.flipper = RandAxisFlip(prob=1.0, lazy=lazy)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.flipper.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAxisFlipd:
        super().set_random_state(seed, state)
        self.flipper.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            return d

        self.randomize(None)

        # all the keys share the same random selected axis
        self.flipper.randomize(d[first_key])

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key], randomize=False, lazy=lazy_)
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.flipper.inverse(d[key])
        return d


class Rotated(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rotate`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        angle: Rotation angle(s) in radians.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
            It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        rotate_order: for 3D inputs, the order in which the axes are rotated about, following the convention of
            :py:func:`scipy.spatial.transform.Rotation.from_euler`. See
            :py:func:`monai.transforms.utils.create_rotate`. Defaults to ``"XYZ"`` (the legacy behaviour).
    """

    backend = Rotate.backend

    def __init__(
        self,
        keys: KeysCollection,
        angle: Sequence[float] | float,
        keep_size: bool = True,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        rotate_order: str = "XYZ",
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.rotator = Rotate(angle=angle, keep_size=keep_size, lazy=lazy, rotate_order=rotate_order)

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rotator.lazy = val
        self._lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            d[key] = self.rotator(
                d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype=dtype, lazy=lazy_
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.rotator.inverse(d[key])
        return d


class RandRotated(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate`
    Randomly rotates the input arrays.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
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
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
            It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
    """

    backend = RandRotate.backend

    def __init__(
        self,
        keys: KeysCollection,
        range_x: tuple[float, float] | float = 0.0,
        range_y: tuple[float, float] | float = 0.0,
        range_z: tuple[float, float] | float = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.BORDER,
        align_corners: Sequence[bool] | bool = False,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_rotate = RandRotate(
            range_x=range_x, range_y=range_y, range_z=range_z, prob=1.0, keep_size=keep_size, lazy=lazy
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rand_rotate.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandRotated:
        super().set_random_state(seed, state)
        self.rand_rotate.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        self.randomize(None)

        # all the keys share the same random rotate angle
        self.rand_rotate.randomize()
        lazy_ = self.lazy if lazy is None else lazy

        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            if self._do_transform:
                d[key] = self.rand_rotate(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=dtype,
                    randomize=False,
                    lazy=lazy_,
                )
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_rotate.inverse(d[key])
        return d


class Zoomd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Zoom`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
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
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Zoom.backend

    def __init__(
        self,
        keys: KeysCollection,
        zoom: Sequence[float] | float,
        mode: SequenceStr = InterpolateMode.AREA,
        padding_mode: SequenceStr = NumpyPadMode.EDGE,
        align_corners: Sequence[bool | None] | bool | None = None,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy=lazy)

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.zoomer = Zoom(zoom=zoom, keep_size=keep_size, lazy=lazy, **kwargs)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.zoomer.lazy = val
        self._lazy = val

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            d[key] = self.zoomer(
                d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype=dtype, lazy=lazy_
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.zoomer.inverse(d[key])
        return d


class RandZoomd(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):
    """
    Dict-based version :py:class:`monai.transforms.RandZoom`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: Keys to pick data for transformation.
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
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
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
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``float32``.
            If None, use the data type of input data.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not.
            Defaults to False
        kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    backend = RandZoom.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        min_zoom: Sequence[float] | float = 0.9,
        max_zoom: Sequence[float] | float = 1.1,
        mode: SequenceStr = InterpolateMode.AREA,
        padding_mode: SequenceStr = NumpyPadMode.EDGE,
        align_corners: Sequence[bool | None] | bool | None = None,
        dtype: Sequence[DtypeLike | torch.dtype] | DtypeLike | torch.dtype = np.float32,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **kwargs,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_zoom = RandZoom(
            prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, keep_size=keep_size, lazy=lazy, **kwargs
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool):
        self.rand_zoom.lazy = val
        self._lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandZoomd:
        super().set_random_state(seed, state)
        self.rand_zoom.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)

        # all the keys share the same random zoom factor
        self.rand_zoom.randomize(d[first_key])
        lazy_ = self.lazy if lazy is None else lazy

        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            if self._do_transform:
                d[key] = self.rand_zoom(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=dtype,
                    randomize=False,
                    lazy=lazy_,
                )
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            xform = self.pop_transform(d[key])
            if xform[TraceKeys.DO_TRANSFORM]:
                d[key].applied_operations.append(xform[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_zoom.inverse(d[key])
        return d


class GridDistortiond(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GridDistortion`.
    """

    backend = GridDistortion.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_cells: tuple[int] | int,
        distort_steps: list[tuple],
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
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
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.grid_distortion = GridDistortion(num_cells=num_cells, distort_steps=distort_steps, device=device)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.grid_distortion(d[key], mode=mode, padding_mode=padding_mode)
        return d


class RandGridDistortiond(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandGridDistortion`.
    """

    backend = RandGridDistortion.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_cells: tuple[int] | int = 5,
        prob: float = 0.1,
        distort_limit: tuple[float, float] | float = (-0.03, 0.03),
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
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
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_grid_distortion = RandGridDistortion(
            num_cells=num_cells, prob=1.0, distort_limit=distort_limit, device=device
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandGridDistortiond:
        super().set_random_state(seed, state)
        self.rand_grid_distortion.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            out: dict[Hashable, torch.Tensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out = convert_to_tensor(d, track_meta=get_track_meta())
            return out
        if isinstance(d[first_key], MetaTensor) and d[first_key].pending_operations:  # type: ignore
            warnings.warn(f"data['{first_key}'] has pending operations, transform may return incorrect results.")
        self.rand_grid_distortion.randomize(d[first_key].shape[1:])

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.rand_grid_distortion(d[key], mode=mode, padding_mode=padding_mode, randomize=False)
        return d


class GridSplitd(MapTransform, MultiSampleTrait):
    """
    Split the image into patches based on the provided grid in 2D.

    Args:
        keys: keys of the corresponding items to be transformed.
        grid: a tuple define the shape of the grid upon which the image is split. Defaults to (2, 2)
        size: a tuple or an integer that defines the output patch sizes,
            or a dictionary that define it separately for each key, like {"image": 3, "mask", (2, 2)}.
            If it's an integer, the value will be repeated for each dimension.
            The default is None, where the patch size will be inferred from the grid shape.
        allow_missing_keys: don't raise exception if key is missing.

    Note: This transform currently support only image with two spatial dimensions.
    """

    backend = GridSplit.backend

    def __init__(
        self,
        keys: KeysCollection,
        grid: tuple[int, int] = (2, 2),
        size: int | tuple[int, int] | dict[Hashable, int | tuple[int, int] | None] | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.grid = grid
        self.size = size if isinstance(size, dict) else {key: size for key in self.keys}
        self.splitter = GridSplit(grid=grid)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> list[dict[Hashable, NdarrayOrTensor]]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        n_outputs = np.prod(self.grid)
        output: list[dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(n_outputs)]
        for key in self.key_iterator(d):
            result = self.splitter(d[key], self.size[key])
            for i in range(n_outputs):
                output[i][key] = result[i]
        return output


class GridPatchd(MapTransform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps.
    It can sort the patches and return all or a subset of them.

    Args:
        keys: keys of the corresponding items to be transformed.
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        offset: starting position in the array, default is 0 for each dimension.
            np.random.randint(0, patch_size, 2) creates random start between 0 and `patch_size` for a 2D image.
        num_patches: number of patches (or maximum number of patches) to return.
            If the requested number of patches is greater than the number of available patches,
            padding will be applied to provide exactly `num_patches` patches unless `threshold` is set.
            When `threshold` is set, this value is treated as the maximum number of patches.
            Defaults to None, which does not limit number of the patches.
        overlap: amount of overlap between patches in each dimension. Default to 0.0.
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
        allow_missing_keys: don't raise exception if key is missing.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    Returns:
        dictionary, contains the all the original key/value with the values for `keys`
            replaced by the patches, a MetaTensor with following metadata:

            - `PatchKeys.LOCATION`: the starting location of the patch in the image,
            - `PatchKeys.COUNT`: total number of patches in the image,
            - "spatial_shape": spatial size of the extracted patch, and
            - "offset": the amount of offset for the patches in the image (starting position of the first patch)
    """

    backend = GridPatch.backend

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        offset: Sequence[int] | None = None,
        num_patches: int | None = None,
        overlap: float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str | None = None,
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ):
        super().__init__(keys, allow_missing_keys)
        self.patcher = GridPatch(
            patch_size=patch_size,
            offset=offset,
            num_patches=num_patches,
            overlap=overlap,
            sort_fn=sort_fn,
            threshold=threshold,
            pad_mode=pad_mode,
            **pad_kwargs,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.patcher(d[key])
        return d


class RandGridPatchd(RandomizableTransform, MapTransform, MultiSampleTrait):
    """
    Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps,
    and with random offset for the minimal corner of the image, (0,0) for 2D and (0,0,0) for 3D.
    It can sort the patches and return all or a subset of them.

    Args:
        keys: keys of the corresponding items to be transformed.
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        min_offset: the minimum range of starting position to be selected randomly. Defaults to 0.
        max_offset: the maximum range of starting position to be selected randomly.
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
        allow_missing_keys: don't raise exception if key is missing.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    Returns:
        dictionary, contains the all the original key/value with the values for `keys`
            replaced by the patches, a MetaTensor with following metadata:

            - `PatchKeys.LOCATION`: the starting location of the patch in the image,
            - `PatchKeys.COUNT`: total number of patches in the image,
            - "spatial_shape": spatial size of the extracted patch, and
            - "offset": the amount of offset for the patches in the image (starting position of the first patch)

    """

    backend = RandGridPatch.backend

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        min_offset: Sequence[int] | int | None = None,
        max_offset: Sequence[int] | int | None = None,
        num_patches: int | None = None,
        overlap: float = 0.0,
        sort_fn: str | None = None,
        threshold: float | None = None,
        pad_mode: str | None = None,
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.patcher = RandGridPatch(
            patch_size=patch_size,
            min_offset=min_offset,
            max_offset=max_offset,
            num_patches=num_patches,
            overlap=overlap,
            sort_fn=sort_fn,
            threshold=threshold,
            pad_mode=pad_mode,
            **pad_kwargs,
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandGridPatchd:
        super().set_random_state(seed, state)
        self.patcher.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        # All the keys share the same random noise
        for key in self.key_iterator(d):
            self.patcher.randomize(d[key])
            break
        for key in self.key_iterator(d):
            d[key] = self.patcher(d[key], randomize=False)
        return d


class RandSimulateLowResolutiond(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandSimulateLowResolution`.
    Random simulation of low resolution corresponding to nnU-Net's SimulateLowResolutionTransform
    (https://github.com/MIC-DKFZ/batchgenerators/blob/7651ece69faf55263dd582a9f5cbd149ed9c3ad0/batchgenerators/transforms/resample_transforms.py#L23)
    First, the array/tensor is resampled at lower resolution as determined by the zoom_factor which is uniformly sampled
    from the `zoom_range`. Then, the array/tensor is resampled at the original resolution.
    """

    backend = RandAffine.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        downsample_mode: InterpolateMode | str = InterpolateMode.NEAREST,
        upsample_mode: InterpolateMode | str = InterpolateMode.TRILINEAR,
        zoom_range=(0.5, 1.0),
        align_corners=False,
        allow_missing_keys: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            prob: probability of performing this augmentation
            downsample_mode: interpolation mode for downsampling operation
            upsample_mode: interpolation mode for upsampling operation
            zoom_range: range from which the random zoom factor for the downsampling and upsampling operation is
            sampled. It determines the shape of the downsampled tensor.
            align_corners: This only has an effect when downsample_mode or upsample_mode  is 'linear', 'bilinear',
                'bicubic' or 'trilinear'. Default: False
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            allow_missing_keys: don't raise exception if key is missing.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.zoom_range = zoom_range
        self.align_corners = align_corners
        self.device = device

        self.sim_lowres_tfm = RandSimulateLowResolution(
            prob=1.0,  # probability is handled by dictionary class
            downsample_mode=self.downsample_mode,
            upsample_mode=self.upsample_mode,
            zoom_range=self.zoom_range,
            align_corners=self.align_corners,
            device=self.device,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandSimulateLowResolutiond:
        super().set_random_state(seed, state)
        self.sim_lowres_tfm.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be transformed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)

        for key in self.key_iterator(d):
            # do the transform
            if self._do_transform:
                d[key] = self.sim_lowres_tfm(d[key])  # type: ignore
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
        return d


class ConvertBoxToPointsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertBoxToPoints`.
    """

    backend = ConvertBoxToPoints.backend

    def __init__(
        self,
        keys: KeysCollection,
        point_key="points",
        mode: str | BoxMode | type[BoxMode] | None = StandardMode,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            point_key: key to store the point data.
            mode: the mode of the input boxes. Defaults to StandardMode.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.point_key = point_key
        self.converter = ConvertBoxToPoints(mode=mode)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            data[self.point_key] = self.converter(d[key])
        return data


class ConvertPointsToBoxesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertPointsToBoxes`.
    """

    def __init__(self, keys: KeysCollection, box_key="box", allow_missing_keys: bool = False):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            box_key: key to store the box data.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
        self.converter = ConvertPointsToBoxes()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            data[self.box_key] = self.converter(d[key])
        return data


SpatialResampleD = SpatialResampleDict = SpatialResampled
ResampleToMatchD = ResampleToMatchDict = ResampleToMatchd
SpacingD = SpacingDict = Spacingd
OrientationD = OrientationDict = Orientationd
Rotate90D = Rotate90Dict = Rotate90d
RandRotate90D = RandRotate90Dict = RandRotate90d
ResizeD = ResizeDict = Resized
AffineD = AffineDict = Affined
RandAffineD = RandAffineDict = RandAffined
Rand2DElasticD = Rand2DElasticDict = Rand2DElasticd
Rand3DElasticD = Rand3DElasticDict = Rand3DElasticd
FlipD = FlipDict = Flipd
RandFlipD = RandFlipDict = RandFlipd
GridDistortionD = GridDistortionDict = GridDistortiond
RandGridDistortionD = RandGridDistortionDict = RandGridDistortiond
RandAxisFlipD = RandAxisFlipDict = RandAxisFlipd
RotateD = RotateDict = Rotated
RandRotateD = RandRotateDict = RandRotated
ZoomD = ZoomDict = Zoomd
RandZoomD = RandZoomDict = RandZoomd
GridSplitD = GridSplitDict = GridSplitd
GridPatchD = GridPatchDict = GridPatchd
RandGridPatchD = RandGridPatchDict = RandGridPatchd
RandSimulateLowResolutionD = RandSimulateLowResolutionDict = RandSimulateLowResolutiond
ConvertBoxToPointsD = ConvertBoxToPointsDict = ConvertBoxToPointsd
ConvertPointsToBoxesD = ConvertPointsToBoxesDict = ConvertPointsToBoxesd
