"""Tests for vision (PNG), audio (WAV/DSP), and the shared imaging module."""
import io
import math
import wave

import pytest

from cognify.capabilities.audio import AudioCapability, note_name, read_wav
from cognify.capabilities.imaging import (decode_png, encode_apng, encode_png,
                                          has_actl, is_png)
from cognify.capabilities.vision import VisionCapability, color_name
from cognify.core.stimulus import Stimulus


# --------------------------------------------------------------------- helpers
def solid(width, height, rgb):
    return [[rgb] * width for _ in range(height)]


def wav_bytes(samples, rate=22050):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = b"".join(
            int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True)
            for s in samples)
        wf.writeframes(frames)
    return buf.getvalue()


def sine(freq, seconds, rate=22050, amp=0.6):
    return [amp * math.sin(2 * math.pi * freq * i / rate)
            for i in range(int(seconds * rate))]


class TestImaging:
    def test_roundtrip_solid(self):
        rows = solid(10, 6, (200, 30, 40))
        png = encode_png(10, 6, rows)
        assert is_png(png)
        w, h, decoded = decode_png(png)
        assert (w, h) == (10, 6)
        assert decoded[0][0] == (200, 30, 40)
        assert decoded[5][9] == (200, 30, 40)

    def test_roundtrip_gradient(self):
        rows = [[(x * 7 % 256, y * 5 % 256, 128) for x in range(24)]
                for y in range(16)]
        w, h, decoded = decode_png(encode_png(24, 16, rows))
        assert decoded == rows

    def test_mismatched_rows_rejected(self):
        with pytest.raises(ValueError):
            encode_png(10, 5, solid(10, 4, (0, 0, 0)))

    def test_not_png_rejected(self):
        with pytest.raises(ValueError):
            decode_png(b"NOTAPNGFILE........")

    def test_apng_structure(self):
        frames = [(8, 4, solid(8, 4, (255, 0, 0))),
                  (8, 4, solid(8, 4, (0, 255, 0))),
                  (8, 4, solid(8, 4, (0, 0, 255)))]
        apng = encode_apng(frames)
        assert is_png(apng) and has_actl(apng)
        w, h, first = decode_png(apng)          # first frame decodes
        assert (w, h) == (8, 4)
        assert first[0][0] == (255, 0, 0)

    def test_apng_size_mismatch_rejected(self):
        with pytest.raises(ValueError):
            encode_apng([(8, 4, solid(8, 4, (0, 0, 0))),
                         (9, 4, solid(9, 4, (0, 0, 0)))])


class TestColorName:
    def test_basic_colors(self):
        assert color_name(220, 30, 30) == "red"
        assert color_name(30, 200, 60) == "green"
        assert color_name(250, 250, 250) == "white"
        assert color_name(10, 10, 10) == "black"


class TestVision:
    def test_sees_red_ball_on_green_field(self):
        rows = solid(48, 36, (40, 180, 70))
        for y in range(10, 22):
            for x in range(18, 30):
                rows[y][x] = (220, 40, 40)
        cap = VisionCapability()
        p = cap.handle(Stimulus.image(encode_png(48, 36, rows)))
        assert p.modality == "vision"
        colors = {o["color"] for o in p.objects}
        assert "red" in colors
        assert p.brightness > 0.2
        assert "image" in p.description

    def test_brightness_levels(self):
        cap = VisionCapability()
        dark = cap.handle(Stimulus.image(encode_png(16, 16, solid(16, 16, (5, 5, 5)))))
        bright = cap.handle(Stimulus.image(encode_png(16, 16, solid(16, 16, (250, 250, 250)))))
        assert dark.brightness < 0.1 < bright.brightness

    def test_position_report(self):
        rows = solid(48, 36, (30, 30, 30))
        for y in range(2, 10):
            for x in range(2, 10):
                rows[y][x] = (220, 40, 40)      # top-left blob
        p = VisionCapability().handle(Stimulus.image(encode_png(48, 36, rows)))
        obj = next(o for o in p.objects if o["color"] == "red")
        assert obj["position"].startswith("top")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError):
            VisionCapability().handle(Stimulus.image(b"\x00\xff\x00..."))

    def test_can_handle_only_images(self):
        cap = VisionCapability()
        assert cap.can_handle(Stimulus.image(b"\x89PNG..."))
        assert not cap.can_handle(Stimulus.text("look at this"))


class TestAudio:
    def test_note_name(self):
        assert note_name(440.0) == "A4"
        assert note_name(261.63) == "C4"

    def test_wav_roundtrip(self):
        samples = sine(440, 0.5)
        read, rate = read_wav(wav_bytes(samples))
        assert rate == 22050
        assert len(read) == pytest.approx(len(samples), rel=0.01)
        assert max(read) > 0.5

    def test_tone_detection_finds_440(self):
        p = AudioCapability().handle(Stimulus.audio(wav_bytes(sine(440, 1.0))))
        assert p.dominant_hz == pytest.approx(440, abs=15)
        assert p.note == "A4"
        assert "tonal" in p.character

    def test_pitch_classification(self):
        low = AudioCapability().handle(Stimulus.audio(wav_bytes(sine(90, 1.0))))
        high = AudioCapability().handle(Stimulus.audio(wav_bytes(sine(1200, 1.0))))
        assert low.dominant_hz < 200 < high.dominant_hz

    def test_silence_is_quiet(self):
        p = AudioCapability().handle(Stimulus.audio(wav_bytes([0.0] * 22050)))
        assert p.rms < 0.01

    def test_noise_is_noise(self):
        import random
        rng = random.Random(5)
        samples = [rng.uniform(-1, 1) for _ in range(22050)]
        p = AudioCapability().handle(Stimulus.audio(wav_bytes(samples)))
        assert p.character

    def test_invalid_wav_raises(self):
        with pytest.raises(ValueError):
            AudioCapability().handle(Stimulus.audio(b"not a wav file at all"))

    def test_can_handle_only_audio(self):
        cap = AudioCapability()
        assert cap.can_handle(Stimulus.audio(b"RIFF...."))
        assert not cap.can_handle(Stimulus.text("listen"))
