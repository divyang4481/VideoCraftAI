import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy's built-in SlideIn is unstable for full-screen materials in the current processing chain.
    # There will be a situation where "the transition is logically applied, but there is almost no change in the picture."
    # Here it is changed to explicit black background + displacement animation to ensure that the transition effect is visible and the behavior is controllable.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_start = max(clip.duration - t, 0)

    # SlideOut is also changed to explicit displacement to ensure that the end of the clip can slide out of the screen stably.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(t, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Retaining the 20% zoom range of the original design gives a clearly visible sense of Ken Burns' movement even in short clips of around three seconds.
# Scaling stability is ensured by sub-pixel center sampling below, without masking source video encoding flicker by reducing the magnitude of the effect.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Use sub-pixel center cropping to achieve black-edge-free and stable zoom effects.

    You cannot convert the cropping width and height into integers first: when the scaling ratio changes continuously, the integer boundaries will jump at different steps.
    And the half-pixel sampling phase is changed when odd and even sizes are switched, which ultimately manifests as picture jitter.Pillow of EXTENT
    The transformation can directly receive floating point boundaries and complete sub-pixel sampling on the fixed output canvas; left and right, upper and lower boundaries
    It is always symmetrical around the same floating point center, so it is suitable for scenes where the entire video continues to zoom slowly.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # 1x zoom directly returns to the original frame to avoid meaningless resampling causing slight blurring of the first frame.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # Continuous video scaling pays more attention to the consistency of adjacent frames. BICUBIC/LANCZOS Although single frame is sharper,
        # However, high-frequency textures are prone to ringing and brightness flickering when crossing the sampling grid; BILINEAR is softer and
        # A small loss of sharpness can be exchanged for a more stable dynamic look.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Smoothly zooms in from original to 1.2 times."""
    # t is temporarily reserved to maintain a unified call signature with other transition functions; scaling needs to cover the entire clip,
    # Otherwise, the picture will suddenly freeze after a short zoom, which is not suitable for static or low-motion materials.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """throughout the fragment from 1.2 Smoothly zoom out to the original screen."""
    # Consistent with zoomin_transition, t is only used to be compatible with the unified transition calling interface.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)
