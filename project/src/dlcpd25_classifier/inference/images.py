"""Defensive image decoding and canonical RGB conversion."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ImageValidationError

SUPPORTED_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class ImageLimits:
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 40_000_000

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError("image limits must be positive")


ImageSource = Image.Image | str | Path | bytes | bytearray | BinaryIO


def _source_stream(source: ImageSource, limits: ImageLimits) -> BinaryIO:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ImageValidationError("image_missing", "找不到上传的图片，请重新选择文件。")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageValidationError(
                "unsupported_extension", "不支持该文件扩展名，请上传 JPG、PNG、WEBP、BMP 或 TIFF 图片。"
            )
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ImageValidationError("image_unreadable", "无法读取上传的图片，请重新选择文件。") from exc
        if size > limits.max_upload_bytes:
            raise ImageValidationError("file_too_large", "图片文件过大，请压缩后重试。")
        try:
            return path.open("rb")
        except OSError as exc:
            raise ImageValidationError("image_unreadable", "无法读取上传的图片，请重新选择文件。") from exc
    if isinstance(source, (bytes, bytearray)):
        if len(source) > limits.max_upload_bytes:
            raise ImageValidationError("file_too_large", "图片文件过大，请压缩后重试。")
        return io.BytesIO(source)
    if hasattr(source, "read"):
        stream = source
        current = stream.tell() if hasattr(stream, "tell") else None
        payload = stream.read(limits.max_upload_bytes + 1)
        if current is not None and hasattr(stream, "seek"):
            stream.seek(current)
        if len(payload) > limits.max_upload_bytes:
            raise ImageValidationError("file_too_large", "图片文件过大，请压缩后重试。")
        return io.BytesIO(payload)
    raise ImageValidationError("unsupported_input", "无法识别上传内容，请选择有效图片。")


def _check_dimensions(image: Image.Image, limits: ImageLimits) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageValidationError("invalid_dimensions", "图片尺寸无效，请重新选择图片。")
    if width * height > limits.max_image_pixels:
        raise ImageValidationError("pixel_limit_exceeded", "图片像素尺寸过大，请缩小后重试。")


def load_rgb_image(source: ImageSource, limits: ImageLimits | None = None) -> Image.Image:
    """Decode one image, apply EXIF orientation, and return an owned RGB image."""
    active_limits = limits or ImageLimits()
    if isinstance(source, Image.Image):
        try:
            image = source.copy()
            _check_dimensions(image, active_limits)
            image = ImageOps.exif_transpose(image)
            image.load()
            return image.convert("RGB")
        except ImageValidationError:
            raise
        except (OSError, ValueError) as exc:
            raise ImageValidationError("decode_failed", "图片已损坏或无法识别，请重新选择有效图片。") from exc

    stream = _source_stream(source, active_limits)
    try:
        with stream, warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(stream) as decoded:
                _check_dimensions(decoded, active_limits)
                oriented = ImageOps.exif_transpose(decoded)
                oriented.load()
                return oriented.convert("RGB")
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("pixel_limit_exceeded", "图片像素尺寸过大，请缩小后重试。") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("decode_failed", "图片已损坏或无法识别，请重新选择有效图片。") from exc
