"""Audio: parses real WAV data and extracts acoustic features.

Uses the stdlib ``wave`` module for decoding and implements Goertzel tone
detection, RMS/ZCR/envelope analysis, and simple acoustic classification —
all in pure Python.
"""
from __future__ import annotations

import array
import io
import math
import wave
from typing import List, Tuple

from ..core.stimulus import Stimulus
from .base import AudioPercept, Capability

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(freq: float) -> str:
    """Nearest equal-temperament note name for a frequency."""
    if freq <= 0:
        return ""
    n = round(12 * math.log2(freq / 440.0)) + 69       # MIDI number
    if n < 0 or n > 127:
        return ""
    return f"{NOTES[n % 12]}{n // 12 - 1}"


def read_wav(data: bytes) -> Tuple[List[float], int]:
    """Decode 8/16/32-bit PCM WAV bytes to mono samples in [-1, 1]."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        nch, sw, rate, n = (wf.getnchannels(), wf.getsampwidth(),
                            wf.getframerate(), wf.getnframes())
        raw = wf.readframes(n)
    if sw == 1:
        samples = [(b - 128) / 128.0 for b in raw]
    elif sw == 2:
        buf = array.array("h")
        buf.frombytes(raw[: (len(raw) // 2) * 2])
        samples = [s / 32768.0 for s in buf]
    elif sw == 4:
        buf = array.array("i")
        buf.frombytes(raw[: (len(raw) // 4) * 4])
        samples = [s / 2147483648.0 for s in buf]
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if nch > 1:
        samples = [sum(samples[i:i + nch]) / nch
                   for i in range(0, len(samples) - nch + 1, nch)]
    return samples, rate


def goertzel(samples: List[float], rate: int, freq: float) -> float:
    """Signal power at a single frequency (Goertzel algorithm)."""
    w = 2.0 * math.pi * freq / rate
    coeff = 2.0 * math.cos(w)
    s1 = s2 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0
    return s1 * s1 + s2 * s2 - coeff * s1 * s2


def _spectrum(samples: List[float], rate: int, bins: int = 40) -> List[Tuple[float, float]]:
    """Log-spaced power spectrum 55..2200 Hz over a middle window."""
    if len(samples) > 8192:
        start = (len(samples) - 8192) // 2
        window = samples[start:start + 8192]
    else:
        window = samples
    # decimate to <= 8 kHz for speed
    factor = max(1, rate // 8000)
    if factor > 1:
        window = [sum(window[i:i + factor]) / factor
                  for i in range(0, len(window) - factor + 1, factor)]
    eff_rate = rate // factor
    out: List[Tuple[float, float]] = []
    for i in range(bins):
        f = 55.0 * (2200.0 / 55.0) ** (i / (bins - 1))
        out.append((f, max(0.0, goertzel(window, eff_rate, f))))
    return out


class AudioCapability(Capability):
    name = "audio"
    description = "Hears WAV audio: loudness, pitch, tone vs speech vs noise."
    phase = "perceive"

    def can_handle(self, stimulus: Stimulus, ctx=None) -> bool:
        return stimulus.data is not None and stimulus.modality.value == "audio"

    def handle(self, stimulus: Stimulus, ctx=None) -> AudioPercept | None:
        data = stimulus.data or b""
        try:
            samples, rate = read_wav(data)
        except (wave.Error, EOFError, ValueError) as exc:
            raise ValueError(f"audio: unsupported format ({exc}); use PCM WAV") from exc
        if not samples:
            raise ValueError("audio: empty clip")

        duration = len(samples) / rate
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        peak = max(abs(s) for s in samples)
        zcr = self._zero_crossings(samples) / len(samples)
        env = self._envelope(samples, rate)
        env_var = self._variance(env)
        spec = _spectrum(samples, rate)
        total_power = sum(p for _, p in spec) or 1e-12
        dom_f, dom_p = max(spec, key=lambda kv: kv[1])
        tonality = dom_p / total_power
        peaks = self._count_peaks(env)

        character = self._classify(tonality, env_var, zcr, peaks)
        loud = ("loud" if rms > 0.25 else "quiet" if rms < 0.05 else "moderately loud")
        pitch = ("high" if dom_f > 800 else "low" if dom_f < 200 else "mid-range")
        desc = (f"A {duration:.1f}s audio clip: {loud}, {pitch}-pitched "
                f"({dom_f:.0f} Hz ≈ {note_name(dom_f) or 'unknown note'}), "
                f"{character}.")

        return AudioPercept(
            description=desc, confidence=0.75,
            duration=round(duration, 3), rms=round(rms, 4),
            dominant_hz=round(dom_f, 1), note=note_name(dom_f),
            zcr=round(zcr, 4), character=character,
            attributes={"peak": round(peak, 3), "tonality": round(tonality, 3),
                        "envelope_variance": round(env_var, 5),
                        "bursts": peaks},
        )

    # ------------------------------------------------------------------ features
    @staticmethod
    def _zero_crossings(samples: List[float]) -> int:
        count = 0
        for a, b in zip(samples, samples[1:]):
            if (a >= 0) != (b >= 0):
                count += 1
        return count

    @staticmethod
    def _envelope(samples: List[float], rate: int, window_ms: int = 40) -> List[float]:
        w = max(1, int(rate * window_ms / 1000))
        env: List[float] = []
        for i in range(0, len(samples) - w + 1, w):
            chunk = samples[i:i + w]
            env.append(math.sqrt(sum(s * s for s in chunk) / w))
        return env or [0.0]

    @staticmethod
    def _variance(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        mean = sum(xs) / len(xs)
        return sum((x - mean) ** 2 for x in xs) / len(xs)

    @staticmethod
    def _count_peaks(env: List[float], threshold: float = 0.05) -> int:
        peaks = 0
        for a, b, c in zip(env, env[1:], env[2:]):
            if b > a and b >= c and b > threshold:
                peaks += 1
        return peaks

    @staticmethod
    def _classify(tonality: float, env_var: float, zcr: float, bursts: int) -> str:
        if tonality > 0.35 and env_var < 0.02:
            return "steady tonal sound (musical or a tone)"
        if env_var > 0.01 and 0.02 < zcr < 0.45:
            return "speech-like, with bursts of energy"
        if bursts > 2:
            return "rhythmic intermittent sound"
        return "noisy or unstructured sound"
