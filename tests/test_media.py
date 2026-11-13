"""Tests for media generation (WAV synthesis + APNG video)."""
import math

import pytest

from cognify.capabilities.audio import read_wav
from cognify.capabilities.imaging import decode_png, has_actl
from cognify.capabilities.media import (MediaGenerationCapability, MediaAsset,
                                        SceneRenderer, melody_from_prompt,
                                        note_freq, parse_mood,
                                        synth_tone, video_from_prompt, wav_bytes)
from cognify.core.stimulus import Stimulus


class TestNoteFreq:
    def test_a4(self):
        assert note_freq("A4") == pytest.approx(440.0, rel=1e-3)

    def test_c4(self):
        assert note_freq("C4") == pytest.approx(261.63, rel=1e-2)

    def test_octave_doubling(self):
        assert note_freq("A5") == pytest.approx(note_freq("A4") * 2, rel=1e-3)

    def test_invalid_note(self):
        with pytest.raises(ValueError):
            note_freq("H13")


class TestAudioSynthesis:
    def test_tone_length(self):
        samples = synth_tone(440, 1.0)
        assert len(samples) == pytest.approx(22050, rel=0.01)

    def test_tone_amplitude_bounded(self):
        samples = synth_tone(440, 0.5, volume=0.9)
        assert max(samples) <= 1.0 and min(samples) >= -1.0

    def test_wav_bytes_parse_back(self):
        samples = synth_tone(330, 0.4)
        read, rate = read_wav(wav_bytes(samples))
        assert rate == 22050
        assert len(read) == pytest.approx(len(samples), rel=0.01)

    def test_melody_asset(self):
        asset = melody_from_prompt("a happy little tune", seed=2)
        assert asset.kind == "audio"
        assert asset.data[:4] == b"RIFF"
        read, _ = read_wav(asset.data)
        assert len(read) > 22050               # at least ~1s of music
        assert asset.metadata["mood"] == "happy"

    def test_mood_parsing(self):
        assert parse_mood("a sad rainy ballad") == "sad"
        assert parse_mood("energetic dance banger") == "energetic"
        assert parse_mood("something calm and soft") == "calm"
        assert parse_mood("mysterious dark cave") == "dark"
        assert parse_mood("plain prompt") == "happy"

    def test_explicit_notes_respected(self):
        asset = melody_from_prompt("play C4 E4 G4 for me", seed=1)
        assert asset.metadata["notes"] == 3


class TestVideoSynthesis:
    def test_video_asset_structure(self):
        asset = video_from_prompt("a happy bouncing ball animation", frames=10)
        assert asset.kind == "video"
        assert asset.metadata["frames"] == 10
        assert asset.metadata["motion"] == "bounce"
        assert has_actl(asset.data)

    def test_video_decodes_first_frame(self):
        asset = video_from_prompt("dark spiral galaxy animation", frames=6)
        w, h, rows = decode_png(asset.data)
        assert (w, h) == (96, 72)
        assert len(rows) == h and len(rows[0]) == w

    def test_motion_selection(self):
        assert video_from_prompt("ocean waves").metadata["motion"] == "wave"
        assert video_from_prompt("night starfield sky").metadata["motion"] == "stars"
        assert video_from_prompt("heartbeat pulse").metadata["motion"] == "pulse"

    def test_frames_vary(self):
        renderer = SceneRenderer("bouncing ball", seed=1)
        f0 = renderer.render_frame(0.0)
        f1 = renderer.render_frame(0.25)   # quarter-period: ball must move
        assert f0 != f1

    def test_frame_pixels_in_range(self):
        renderer = SceneRenderer("calm waves", seed=3)
        frame = renderer.render_frame(0.3)
        for row in frame:
            for (r, g, b) in row:
                assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255


class TestMediaCapability:
    def test_can_handle_prompts(self):
        cap = MediaGenerationCapability()
        assert cap.can_handle(Stimulus.text("generate a happy song"))
        assert cap.can_handle(Stimulus.text("make an animation of a bouncing ball"))
        assert not cap.can_handle(Stimulus.text("what is a bird?"))

    def test_handle_audio_request(self):
        cap = MediaGenerationCapability()
        percept = cap.handle(Stimulus.text("play a calm melody"))
        assert percept is not None
        assert percept.assets[0].kind == "audio"

    def test_handle_video_request(self):
        cap = MediaGenerationCapability()
        percept = cap.handle(Stimulus.text("create an animation of waves"))
        assert percept is not None
        assert percept.assets[0].kind == "video"

    def test_asset_save(self, tmp_path):
        asset = melody_from_prompt("a tune", seed=5)
        path = tmp_path / "out.wav"
        assert asset.save(str(path)) == str(path)
        assert path.stat().st_size == len(asset.data)

    def test_asset_str(self):
        asset = MediaAsset("audio", b"xx", "desc")
        assert "audio" in str(asset) and "desc" in str(asset)
