"""The seven capability modules in action."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

from cognify import MindBuilder, Stimulus
from cognify.capabilities.imaging import encode_png
from cognify.capabilities.media import melody_from_prompt, video_from_prompt

mind = (MindBuilder("Sensorium")
        .with_preset("curious_child")
        .enable_vision()            # 1. Vision - real PNG decoding
        .enable_audio()             # 2. Audio  - real WAV analysis (Goertzel DSP)
        .enable_embeddings()        # 3. Embeddings - deterministic vectors
        .enable_multilingual()      # 4. Multilingual - 11-language detection
        .enable_tool_use()          # 5. Tool use - safe function calling
        .enable_code()              # 6. Code - real AST analysis + generation
        .enable_media_generation()  # 7. Media - WAV melodies + APNG animation
        .build())

# --- 1. VISION: draw a tiny scene (red ball on green field) and show it ------
rows = [[(40, 180, 70)] * 48 for _ in range(36)]
for y in range(10, 22):
    for x in range(18, 30):
        rows[y][x] = (220, 40, 40)
reply = mind.process(Stimulus.image(encode_png(48, 36, rows), text="Look!"))
print("VISION:", reply.text)

# --- 2. AUDIO: synthesize a tone, let the mind listen ------------------------
tone = melody_from_prompt("a calm melody").data
reply = mind.process(Stimulus.audio(tone, text="Listen..."))
print("AUDIO: ", reply.text)

# --- 4. MULTILINGUAL: it detects the language and greets in kind -------------
print("LANG:  ", mind.process("Hola, gracias!").text)

# --- 5. TOOLS: the calculator fires automatically on arithmetic --------------
print("TOOLS: ", mind.process("What is 123 * 456?").tool_results)

# --- 6. CODE: real Python AST analysis ---------------------------------------
snippet = "def double(x):\n    return x * 2\n"
print("CODE:  ", mind.process(Stimulus.code(snippet)).text)
print("CODEGEN", mind.process("Write a function that reverses a string").text)

# --- 7. MEDIA GENERATION: real WAV audio + animated PNG video ----------------
os.makedirs("cognify_media", exist_ok=True)
song = melody_from_prompt("a happy energetic melody", seed=3)
film = video_from_prompt("a happy bouncing ball animation", seed=3)
song.save("cognify_media/song.wav")            # playable WAV audio
film.save("cognify_media/film.png")            # animated PNG (open in a browser!)
print(f"MEDIA:  saved song.wav ({len(song.data)} B) and film.png ({len(film.data)} B)")

# --- 3. EMBEDDINGS: deterministic semantic similarity ------------------------
svc = next(c for c in mind.capabilities if c.name == "embeddings")
print("EMBED:  dog/cat similarity =", round(
    svc.similarity("a dog barks", "the cat meows"), 3))
