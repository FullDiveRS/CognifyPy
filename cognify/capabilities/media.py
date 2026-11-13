"""MediaGeneration: synthesizes real WAV audio and animated PNG video."""
from __future__ import annotations

import array
import io
import math
import random
import re
import struct
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.stimulus import Stimulus
from ..util import clamp
from .base import Capability, MediaPercept
from .imaging import encode_apng, encode_png

RATE = 22050

NOTE_INDEX = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5,
              "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8, "a": 9, "a#": 10,
              "bb": 10, "b": 11}
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11, 12],
    "minor": [0, 2, 3, 5, 7, 8, 10, 12],
    "pentatonic": [0, 2, 4, 7, 9, 12, 14, 16],
}
MOODS: Dict[str, Dict[str, Any]] = {
    "happy": {"scale": "major", "bpm": 132, "wave": "triangle", "octave": 4,
              "colors": [(255, 200, 60), (90, 200, 250), (255, 240, 200), (30, 144, 255)], "bg": (30, 60, 120)},
    "sad": {"scale": "minor", "bpm": 66, "wave": "sine", "octave": 3,
            "colors": [(70, 90, 160), (50, 60, 110), (120, 130, 180), (30, 40, 80)], "bg": (12, 14, 30)},
    "calm": {"scale": "pentatonic", "bpm": 84, "wave": "sine", "octave": 4,
             "colors": [(120, 220, 180), (80, 160, 200), (200, 240, 220), (40, 100, 140)], "bg": (16, 40, 52)},
    "energetic": {"scale": "major", "bpm": 160, "wave": "square", "octave": 4,
                  "colors": [(255, 90, 60), (255, 200, 40), (240, 60, 120), (60, 40, 160)], "bg": (40, 10, 40)},
    "dark": {"scale": "minor", "bpm": 100, "wave": "triangle", "octave": 3,
             "colors": [(150, 60, 60), (90, 40, 90), (60, 60, 70), (200, 80, 40)], "bg": (10, 8, 14)},
}
MOOD_WORDS = {
    "happy": ["happy", "joyful", "cheerful", "bright", "sunny", "fun", "playful"],
    "sad": ["sad", "melancholy", "gloomy", "blue", "somber", "rain"],
    "calm": ["calm", "peaceful", "gentle", "soft", "serene", "quiet", "zen"],
    "energetic": ["energetic", "fast", "exciting", "intense", "dance", "rock", "hyper"],
    "dark": ["dark", "scary", "eerie", "mysterious", "spooky", "night"],
}


def note_freq(note: str) -> float:
    """'A4' -> 440.0, 'C4' -> 261.63 (equal temperament, A4 = 440)."""
    m = re.fullmatch(r"([a-g][#b]?)(\d)", note.strip().lower())
    if not m:
        raise ValueError(f"invalid note {note!r}")
    semitones = NOTE_INDEX[m.group(1)] + 12 * (int(m.group(2)) + 1) - 69
    return 440.0 * (2.0 ** (semitones / 12.0))


def parse_mood(prompt: str) -> str:
    p = prompt.lower()
    for mood, words in MOOD_WORDS.items():
        if any(w in p for w in words):
            return mood
    return "happy"


@dataclass
class MediaAsset:
    kind: str                      # "audio" | "video"
    data: bytes
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        with open(path, "wb") as fh:
            fh.write(self.data)
        return path

    def __str__(self) -> str:
        return f"MediaAsset({self.kind}, {len(self.data)} bytes, {self.description})"


# ------------------------------------------------------------------ audio synth
def _waveform(kind: str, phase: float) -> float:
    if kind == "sine":
        return math.sin(phase)
    if kind == "square":
        return 0.6 if math.sin(phase) >= 0 else -0.6
    if kind == "triangle":
        return 2.0 / math.pi * math.asin(math.sin(phase))
    return math.sin(phase)


def synth_tone(freq: float, seconds: float, wave_kind: str = "sine",
               volume: float = 0.5, attack: float = 0.01,
               release: float = 0.08, rate: int = RATE,
               harmonic: float = 0.25) -> List[float]:
    """One note with ADSR-ish envelope and a soft second harmonic."""
    n = int(seconds * rate)
    out: List[float] = []
    for i in range(n):
        t = i / rate
        phase = 2.0 * math.pi * freq * t
        value = _waveform(wave_kind, phase) + harmonic * _waveform(wave_kind, phase * 2)
        env = 1.0
        if t < attack:
            env = t / attack
        elif t > seconds - release:
            env = max(0.0, (seconds - t) / release)
        out.append(clamp(value * env * volume, -1.0, 1.0))
    return out


def synth_melody(notes: List[Tuple[float, float]], wave_kind: str = "sine",
                 volume: float = 0.5, rate: int = RATE) -> List[float]:
    samples: List[float] = []
    for freq, seconds in notes:
        samples += synth_tone(freq, seconds, wave_kind, volume, rate=rate)
    return samples


def wav_bytes(samples: List[float], rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = array.array("h", [max(-32768, min(32767, int(s * 32767))) for s in samples])
        wf.writeframes(frames.tobytes())
    return buf.getvalue()


def melody_from_prompt(prompt: str, seed: int = 7) -> MediaAsset:
    """Turn a text prompt into a short synthesized melody (real WAV bytes)."""
    mood = parse_mood(prompt)
    cfg = MOODS[mood]
    rng = random.Random(seed + len(prompt))
    explicit = re.findall(r"\b([A-G][#b]?\d)\b", prompt)
    if explicit:
        notes = [(note_freq(n), 0.3) for n in explicit]
        scale_name = cfg["scale"]
    else:
        scale_name = cfg["scale"]
        scale = SCALES[scale_name]
        beat = 60.0 / cfg["bpm"]
        degree = 0
        notes = []
        for _ in range(8):
            step = rng.choice((-2, -1, -1, 0, 1, 1, 2))
            degree = max(0, min(len(scale) - 1, degree + step))
            midi = 12 * (cfg["octave"] + 1) + scale[degree]
            freq = 440.0 * (2.0 ** ((midi - 69) / 12))
            notes.append((freq, beat * rng.choice((0.5, 0.5, 1.0))))
    samples = synth_melody(notes, cfg["wave"], volume=0.45)
    duration = len(samples) / RATE
    note_names = ", ".join(_freq_to_note(f) for f, _ in notes[:4])
    desc = (f"A {mood} {len(notes)}-note melody in {scale_name} "
            f"({note_names}...), {duration:.1f}s WAV")
    return MediaAsset("audio", wav_bytes(samples), desc,
                      {"mood": mood, "scale": scale_name, "notes": len(notes),
                       "duration": round(duration, 2)})


def _freq_to_note(freq: float) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    n = round(12 * math.log2(freq / 440.0)) + 69
    return f"{names[n % 12]}{n // 12 - 1}"


# ------------------------------------------------------------------ video synth
class SceneRenderer:
    """Procedurally renders tiny animated scenes from a mood + motion type."""

    def __init__(self, prompt: str, seed: int = 7) -> None:
        p = prompt.lower()
        self.mood = parse_mood(p)
        self.motion = ("bounce" if re.search(r"bounc|ball|jump", p) else
                       "wave" if re.search(r"wave|ocean|sea|water", p) else
                       "spiral" if re.search(r"spiral|galaxy|vortex", p) else
                       "pulse" if re.search(r"pulse|heartbeat|breath|ring", p) else
                       "stars" if re.search(r"star|night|space|sky", p) else "drift")
        self.cfg = MOODS[self.mood]
        self.rng = random.Random(seed + len(prompt))
        self.orbs = [(self.rng.uniform(0.2, 0.8), self.rng.uniform(0.2, 0.8),
                      self.rng.uniform(0.08, 0.18), i % len(self.cfg["colors"]))
                     for i in range(3)]

    def render_frame(self, t: float, width: int = 96, height: int = 72) -> List[List[Tuple[int, int, int]]]:
        rows: List[List[Tuple[int, int, int]]] = []
        br, bg_, bb = self.cfg["bg"]
        colors = self.cfg["colors"]
        for y in range(height):
            row: List[Tuple[int, int, int]] = []
            vy = y / height
            for x in range(width):
                vx = x / width
                r, g, b = self._pixel(vx, vy, t, br, bg_, bb, colors)
                row.append((r, g, b))
            rows.append(row)
        return rows

    def _pixel(self, vx: float, vy: float, t: float, br: int, bg_: int, bb: int,
               colors) -> Tuple[int, int, int]:
        r, g, b = br, bg_, bb
        if self.motion == "bounce":
            bx = 0.15 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * t))
            by = 0.75 - 0.55 * abs(math.sin(2 * math.pi * t * 2))
            d = math.hypot(vx - bx, (vy - by) * 0.75)
            if d < 0.14:
                r, g, b = colors[0]
            elif d < 0.16:
                r, g, b = tuple(min(255, c + 40) for c in (br, bg_, bb))
        elif self.motion == "wave":
            phase = math.sin(6 * math.pi * vx + 2 * math.pi * t)
            surface = 0.5 + 0.12 * phase
            if vy > surface:
                depth = (vy - surface) * 2
                r, g, b = colors[1 if depth < 0.5 else 2]
                r = int(r * (1 - 0.4 * depth))
                g = int(g * (1 - 0.4 * depth))
                b = min(255, int(b * (1 - 0.2 * depth)))
        elif self.motion == "spiral":
            dx, dy = vx - 0.5, vy - 0.5
            ang = math.atan2(dy, dx)
            dist = math.hypot(dx, dy)
            arm = math.sin(6 * ang + 10 * dist - 4 * math.pi * t)
            if arm > 0.4 and dist < 0.48:
                r, g, b = colors[0] if arm > 0.75 else colors[1]
        elif self.motion == "pulse":
            d = math.hypot(vx - 0.5, vy - 0.5)
            ring = (d - (t % 1.0) * 0.5) % 0.25
            if ring < 0.03:
                r, g, b = colors[0]
            elif d < 0.08:
                r, g, b = colors[2]
        elif self.motion == "stars":
            h = hash((int(vx * 24), int(vy * 18))) & 0xFF
            if h > 244:
                tw = 0.5 + 0.5 * math.sin(4 * math.pi * t + h)
                r = g = b = int(120 + 135 * tw)
            elif vy < 0.22:
                r, g, b = colors[1]
        else:  # drift
            for ox, oy, orb_r, ci in self.orbs:
                px = ox + 0.06 * math.sin(2 * math.pi * t + ox * 6)
                py = oy + 0.05 * math.cos(2 * math.pi * t + oy * 6)
                if math.hypot(vx - px, (vy - py) * 0.8) < orb_r:
                    r, g, b = colors[ci]
        return (int(clamp(r, 0, 255)), int(clamp(g, 0, 255)), int(clamp(b, 0, 255)))


def video_from_prompt(prompt: str, frames: int = 16, width: int = 96,
                      height: int = 72, seed: int = 7) -> MediaAsset:
    """Generate an animated PNG 'video' driven by the prompt's mood/motion."""
    renderer = SceneRenderer(prompt, seed)
    encoded: List[Tuple[int, int, List[List[Tuple[int, int, int]]]]] = []
    for i in range(max(2, frames)):
        t = i / max(2, frames)
        encoded.append((width, height, renderer.render_frame(t, width, height)))
    apng = encode_apng(encoded, delay_ms=50)
    desc = (f"A {frames}-frame {renderer.mood} '{renderer.motion}' animation "
            f"({width}x{height} APNG)")
    return MediaAsset("video", apng, desc,
                      {"mood": renderer.mood, "motion": renderer.motion,
                       "frames": len(encoded), "width": width, "height": height})


class MediaGenerationCapability(Capability):
    """Generates audio (WAV) and video (APNG) from text prompts."""

    name = "media_generation"
    description = "Creates melodies (WAV) and tiny animations (APNG) from text."
    phase = "perceive"

    def can_handle(self, stimulus: Stimulus, ctx=None) -> bool:
        p = stimulus.text.lower()
        audio_want = re.search(r"\b(play|generate|make|create|compose|sing|hum)\b", p) and \
            re.search(r"\b(song|melody|tone|music|audio|sound|note)s?\b", p)
        video_want = re.search(r"\b(generate|make|create|render|animate|draw)\b", p) and \
            re.search(r"\b(video|animation|animated|clip|movie|gif|scene)s?\b", p)
        return bool(audio_want or video_want)

    def handle(self, stimulus: Stimulus, ctx=None) -> Optional[MediaPercept]:
        p = stimulus.text
        assets: List[MediaAsset] = []
        video_want = re.search(r"\b(video|animation|animated|clip|movie|gif|scene)s?\b", p, re.I)
        audio_want = re.search(r"\b(song|melody|tone|music|audio|sound|note)s?\b", p, re.I)
        if video_want:
            assets.append(video_from_prompt(p))
        elif audio_want:
            assets.append(melody_from_prompt(p))
        if not assets:
            return None
        if ctx is not None:
            ctx.assets.extend(assets)
        return MediaPercept(
            description=f"I created media: " + "; ".join(a.description for a in assets) + ".",
            confidence=0.85, assets=assets,
            attributes={"assets": len(assets)},
        )
