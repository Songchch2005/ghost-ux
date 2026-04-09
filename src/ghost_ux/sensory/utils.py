from __future__ import annotations

from io import BytesIO

from PIL import Image


def load_image_from_bytes(payload: bytes) -> Image.Image:
    with Image.open(BytesIO(payload)) as image:
        return image.convert("RGBA")


def save_image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def parse_color(value: str | None) -> tuple[float, float, float] | None:
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith("rgb(") or value.startswith("rgba("):
        numbers = value[value.find("(") + 1 : value.rfind(")")].split(",")
        if len(numbers) < 3:
            return None
        try:
            return tuple(float(numbers[index].strip()) / 255.0 for index in range(3))
        except ValueError:
            return None
    if value.startswith("#") and len(value) in {4, 7}:
        if len(value) == 4:
            value = "#" + "".join(channel * 2 for channel in value[1:])
        try:
            return tuple(int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
        except ValueError:
            return None
    return None


def relative_luminance(rgb: tuple[float, float, float]) -> float:
    def transform(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (transform(channel) for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(text_color: str | None, background_color: str | None) -> float | None:
    foreground = parse_color(text_color)
    background = parse_color(background_color)
    if foreground is None or background is None:
        return None
    lighter = max(relative_luminance(foreground), relative_luminance(background))
    darker = min(relative_luminance(foreground), relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)
