# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Multi-aperture meta-optic camera readout."""

import torch

from ..base import DeepObj
from .psf import conv_psf


def _center_crop(image, size):
    """Center-crop an image batch to ``size`` without changing batch or channels."""
    target_h, target_w = size
    image_h, image_w = image.shape[-2:]
    if target_h > image_h or target_w > image_w:
        raise ValueError(
            f"feature_size {size} exceeds rendered image size {(image_h, image_w)}."
        )
    top = (image_h - target_h) // 2
    left = (image_w - target_w) // 2
    return image[..., top : top + target_h, left : left + target_w]


class MetaOpticChannelArray(DeepObj):
    """Tiled multi-aperture camera with independently specified channel PSFs.

    This model follows the parallel metalens-array architecture used for
    multichannel optical convolution. Each aperture forms a feature image on a
    separate detector ROI. The detector image is the physical, non-negative
    accumulation of all aperture images. Signed feature extraction is performed
    after readout, typically with positive/negative channel pairs.

    The PSFs may come from any DeepLens simulation, including `Pixel2D`,
    `DiffractiveLens`, or a separately calibrated optical model.

    Attributes:
        psfs (torch.Tensor): Per-aperture PSFs, shape [N, C, ks, ks].
        channel_rois (torch.Tensor): Top-left detector coordinates for every
            channel, shape [N, 2].
        detector_shape (tuple): Full detector size as (H, W). [pixel]
        feature_size (tuple): Common feature-image size as (H, W). [pixel]
        channel_signs (torch.Tensor): Post-readout channel signs, shape [N].
        channel_pairs (tuple of tuple): Optional positive/negative channel index
            pairs used for signed feature extraction.
    """

    def __init__(
        self,
        psfs,
        channel_rois,
        feature_size,
        detector_shape=None,
        channel_signs=None,
        channel_pairs=None,
        device="cpu",
    ):
        """Initialize a tiled multi-aperture meta-optic camera.

        Args:
            psfs (torch.Tensor): Per-aperture PSFs, shape [N, C, ks, ks].
            channel_rois (sequence): Top-left detector coordinates for each
                channel, shape [N, 2], ordered as (row, col).
            feature_size (tuple): Common feature-image size as (H, W).
            detector_shape (tuple or None, optional): Full detector size as
                (H, W). When None, it is inferred from the channel ROIs.
            channel_signs (torch.Tensor or None, optional): Post-readout channel
                signs, shape [N]. Defaults to all positive.
            channel_pairs (sequence or None, optional): Pairs of channel indices
                `(positive, negative)` used to form signed features. Defaults to
                None, in which case every signed channel is a feature.
            device (str, optional): Device for all tensors. Defaults to "cpu".
        """
        super().__init__()

        if not torch.is_tensor(psfs) or psfs.ndim != 4:
            raise ValueError("psfs must have shape [N, C, ks, ks].")
        if psfs.shape[-2] != psfs.shape[-1]:
            raise ValueError("PSF kernels must be square.")
        if len(feature_size) != 2 or any(int(v) <= 0 for v in feature_size):
            raise ValueError("feature_size must contain two positive integers.")

        self.psfs = psfs.to(device=device)
        self.channel_rois = torch.as_tensor(channel_rois, dtype=torch.long)
        self.feature_size = (int(feature_size[0]), int(feature_size[1]))
        num_channels = self.psfs.shape[0]
        if self.channel_rois.shape != (num_channels, 2):
            raise ValueError(
                f"channel_rois must have shape {(num_channels, 2)}, got "
                f"{tuple(self.channel_rois.shape)}."
            )
        if bool((self.channel_rois < 0).any()):
            raise ValueError("channel_rois cannot contain negative coordinates.")

        if detector_shape is None:
            bottom = int((self.channel_rois[:, 0] + self.feature_size[0]).max())
            right = int((self.channel_rois[:, 1] + self.feature_size[1]).max())
            detector_shape = (bottom, right)
        if len(detector_shape) != 2 or any(int(v) <= 0 for v in detector_shape):
            raise ValueError("detector_shape must contain two positive integers.")
        self.detector_shape = (int(detector_shape[0]), int(detector_shape[1]))
        if (
            self.channel_rois[:, 0].max().item() + self.feature_size[0]
            > self.detector_shape[0]
            or self.channel_rois[:, 1].max().item() + self.feature_size[1]
            > self.detector_shape[1]
        ):
            raise ValueError("Every channel ROI must fit inside detector_shape.")
        self._validate_nonoverlapping_rois()

        if channel_signs is None:
            channel_signs = torch.ones(num_channels)
        self.channel_signs = torch.as_tensor(channel_signs).reshape(-1)
        if self.channel_signs.numel() != num_channels:
            raise ValueError(
                f"channel_signs must contain {num_channels} values, got "
                f"{self.channel_signs.numel()}."
            )

        if channel_pairs is None:
            self.channel_pairs = ()
        else:
            self.channel_pairs = tuple(
                tuple(int(v) for v in pair) for pair in channel_pairs
            )
        for pos, neg in self.channel_pairs:
            if pos < 0 or pos >= num_channels or neg < 0 or neg >= num_channels:
                raise ValueError(f"Invalid channel pair {(pos, neg)}.")

        self.to(device)

    @classmethod
    def from_metasurface_array(
        cls,
        metasurface_array,
        channel_rois,
        feature_size=None,
        wvln=None,
        upsample_factor=1,
        detector_shape=None,
        channel_signs=None,
        channel_pairs=None,
        device="cpu",
    ):
        """Build a receiver from a `Pixel2DMetasurfaceArray`.

        Args:
            metasurface_array (Pixel2DMetasurfaceArray): Phase-domain
                multi-aperture metasurface.
            channel_rois (sequence): Top-left detector coordinates for every
                aperture.
            feature_size (tuple or None, optional): Feature-image size. Defaults
                to the computed PSF size.
            wvln (float or None, optional): Wavelength in micrometres. When
                None, uses the design wavelength.
            upsample_factor (int, optional): ASM field upsampling factor.
                Defaults to 1.
            detector_shape (tuple or None, optional): Full detector size.
            channel_signs (torch.Tensor or None, optional): Post-readout signs.
            channel_pairs (sequence or None, optional): Positive/negative channel
                pairs.
            device (str, optional): Device for receiver tensors. Defaults to
                "cpu".

        Returns:
            receiver (MetaOpticChannelArray): Receiver containing differentiable
                PSFs generated from the aperture phase maps.
        """
        psfs = metasurface_array.compute_channel_psfs(
            wvln=wvln,
            upsample_factor=upsample_factor,
            normalize=True,
        )
        if feature_size is None:
            feature_size = psfs.shape[-2:]
        return cls(
            psfs=psfs,
            channel_rois=channel_rois,
            feature_size=feature_size,
            detector_shape=detector_shape,
            channel_signs=channel_signs,
            channel_pairs=channel_pairs,
            device=device,
        )

    def _validate_nonoverlapping_rois(self):
        """Reject ROIs that physically overlap on the shared detector."""
        feature_h, feature_w = self.feature_size
        for i in range(self.channel_rois.shape[0]):
            top_i, left_i = self.channel_rois[i].tolist()
            bottom_i, right_i = top_i + feature_h, left_i + feature_w
            for j in range(i + 1, self.channel_rois.shape[0]):
                top_j, left_j = self.channel_rois[j].tolist()
                bottom_j, right_j = top_j + feature_h, left_j + feature_w
                overlap = (
                    top_i < bottom_j
                    and top_j < bottom_i
                    and left_i < right_j
                    and left_j < right_i
                )
                if overlap:
                    raise ValueError(
                        f"Channel ROIs {i} and {j} overlap on the detector."
                    )

    def forward(self, scene):
        """Render the full physical detector image from a scene.

        Args:
            scene (torch.Tensor): Image batch in raw space, shape
                ``[B, C, H, W]``.

        Returns:
            detector (torch.Tensor): Non-negative tiled detector image, shape
                ``[B, C, detector_H, detector_W]``.
        """
        if scene.ndim != 4:
            raise ValueError(f"scene must have shape [B, C, H, W], got {scene.shape}.")
        if scene.shape[1] != self.psfs.shape[1]:
            raise ValueError(
                f"scene channels ({scene.shape[1]}) must match PSF channels "
                f"({self.psfs.shape[1]})."
            )

        detector = scene.new_zeros(
            scene.shape[0],
            scene.shape[1],
            self.detector_shape[0],
            self.detector_shape[1],
        )
        for idx in range(self.psfs.shape[0]):
            channel_image = conv_psf(scene, self.psfs[idx])
            feature = _center_crop(channel_image, self.feature_size)
            top, left = self.channel_rois[idx].tolist()
            detector[
                ...,
                top : top + self.feature_size[0],
                left : left + self.feature_size[1],
            ] += feature
        return detector

    def _extract_signed_channels(self, detector):
        """Extract signed feature maps from a detector image."""
        features = []
        for idx in range(self.psfs.shape[0]):
            top, left = self.channel_rois[idx].tolist()
            feature = detector[
                ...,
                top : top + self.feature_size[0],
                left : left + self.feature_size[1],
            ]
            features.append(feature * self.channel_signs[idx])
        return torch.stack(features, dim=-3)

    def forward_features(self, scene, reduce=None):
        """Render and extract signed channel features.

        Args:
            scene (torch.Tensor): Image batch in raw space, shape [B, C, H, W].
            reduce (str or None, optional): When `"mean"`, average each feature
                map over its spatial dimensions. Defaults to None.

        Returns:
            features (torch.Tensor): Signed channel features. Shape is
                ``[B, C, N, feature_H, feature_W]``, or
                ``[B, C, N]`` when `reduce="mean"`.
        """
        detector = self.forward(scene)
        return self.extract_features(detector, reduce=reduce)

    def extract_features(self, detector, reduce=None):
        """Extract signed features from a detector image.

        Args:
            detector (torch.Tensor): Physical detector image, shape
                [B, C, H, W].
            reduce (str or None, optional): When `"mean"`, average each feature
                map over its spatial dimensions. Defaults to None.

        Returns:
            features (torch.Tensor): Signed channel features.
        """
        features = self._extract_signed_channels(detector)

        if self.channel_pairs:
            paired = [
                features[..., pos, :, :] - features[..., neg, :, :]
                for pos, neg in self.channel_pairs
            ]
            features = torch.stack(paired, dim=-3)

        if reduce is None:
            return features
        if reduce == "mean":
            return features.mean(dim=(-2, -1))
        raise ValueError("reduce must be None or 'mean'.")

    def get_optimizer_params(self, lr=0.01):
        """Return an optimizer parameter group for the receiver PSFs.

        This optimizes the channel PSFs directly. A later phase-aware receiver
        can replace this with differentiable phase-to-PSF generation.

        Args:
            lr (float, optional): Learning rate for the PSFs. Defaults to 0.01.

        Returns:
            params (list): One Adam-style parameter group.
        """
        self.psfs.requires_grad = True
        return [{"params": [self.psfs], "lr": lr}]
