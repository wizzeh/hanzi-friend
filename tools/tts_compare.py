"""Generate an offline, blind Mandarin listening comparison (see README.md)."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import time
import wave
from xml.sax.saxutils import escape

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "riff-24khz-16bit-mono-pcm"
VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-Xiaoxiao:DragonHDFlashLatestNeural",
    "zh-CN-Xiaochen:DragonHDLatestNeural",
    "zh-CN-Yunxi:DragonHDFlashLatestNeural",
]
# No pronunciation hints are sent: this measures how the app's plain text
# performs today. Contextual readings are scored separately from naturalness.
PROMPTS = [
    ("Characters", "狗", "gǒu — dog"),
    ("Characters", "女", "nǚ — female; listen to the ü vowel"),
    ("Characters", "水", "shuǐ — water"),
    ("Characters", "相", "Ambiguous in isolation: xiāng or xiàng. Either is acceptable here; this does not test selecting a card's reading."),
    ("Counted phrases", "一只狗", "yì zhī gǒu — a dog"),
    ("Counted phrases", "一本书", "yì běn shū — a book"),
    ("Counted phrases", "一杯水", "yì bēi shuǐ — a glass of water"),
    ("Counted phrases", "两个人", "liǎng ge rén — two people; neutral-tone 个"),
    ("Contextual readings", "长大", "zhǎng dà — grow up"),
    ("Contextual readings", "长短", "cháng duǎn — length"),
    ("Contextual readings", "银行", "yín háng — bank"),
    ("Contextual readings", "行走", "xíng zǒu — walk"),
    ("Tone changes", "你好", "nǐ hǎo in dictionary notation; the first third tone changes in connected speech"),
    ("Tone changes", "我很好", "wǒ hěn hǎo in dictionary notation; listen for natural third-tone phrasing"),
    ("Tone changes", "不是", "bú shì — 不 changes before a fourth tone"),
    ("Tone changes", "一起", "yì qǐ — 一 changes before a third tone"),
    ("Sentences", "你今天晚上有空吗？我们一起去吃饭吧。", "An everyday invitation; listen for conversational rhythm."),
    ("Sentences", "这件衣服有点儿贵，能不能便宜一点？", "A polite request; listen for natural phrasing and 儿."),
    ("Sentences", "我本来想坐地铁，可是下雨了，就打车过来了。", "A short explanation; listen for pauses and sentence flow."),
    ("Sentences", "请给我两杯水，一杯热的，一杯凉的。", "A simple order; listen to tones, counting, and contrast."),
]


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_wav(data):
    with wave.open(io.BytesIO(data)) as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
            raise ValueError("Unexpected audio format")
        frames = audio.readframes(audio.getnframes())
        if not frames or len(frames) != audio.getnframes() * 2 or not any(frames):
            raise ValueError("Empty, silent, or truncated audio")


def ssml(text, voice):
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="zh-CN"><voice name="' + escape(voice, {'"': '&quot;'})
            + '">' + escape(text) + '</voice></speak>')


def synthesize(session, text, voice, key, region):
    for attempt in range(3):
        response = session.post(
            f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers={"Ocp-Apim-Subscription-Key": key,
                     "Content-Type": "application/ssml+xml",
                     "X-Microsoft-OutputFormat": FORMAT,
                     "User-Agent": "hanzi-blind-comparison"},
            data=ssml(text, voice).encode("utf-8"), timeout=(10, 90),
        )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
        if not response.ok:
            # Avoid printing provider bodies or request headers containing keys.
            raise RuntimeError(f"Azure returned HTTP {response.status_code} for {voice}")
        validate_wav(response.content)
        return response.content
    raise RuntimeError("Azure retries exhausted")


def normalize(ffmpeg, source, target):
    """Two-pass EBU R128 loudness matching, preserving pace and pitch."""
    command = [ffmpeg, "-hide_banner", "-nostdin", "-i", str(source)]
    analysis = subprocess.run(command + ["-af", "loudnorm=I=-20:TP=-2:LRA=11:print_format=json",
                                         "-f", "null", "-"], capture_output=True, text=True, check=True)
    stats, _ = json.JSONDecoder().raw_decode(analysis.stderr[analysis.stderr.rfind("{"):])
    if not all(math.isfinite(float(stats[k])) for k in
               ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
        raise ValueError(f"Audio too short or silent for loudness measurement: {source.name}")
    effect = ("loudnorm=I=-20:TP=-2:LRA=11:linear=true:"
              f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
              f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
              f"offset={stats['target_offset']}")
    output = subprocess.run(command + ["-af", effect, "-ar", "24000", "-ac", "1",
                                      "-c:a", "pcm_s16le", "-f", "wav", "-"],
                            capture_output=True, check=True).stdout
    # ffmpeg's pipe WAV uses unknown-length headers. Rewrite a canonical header.
    with wave.open(io.BytesIO(output)) as audio:
        frames = audio.readframes(audio.getnframes())
    canonical = io.BytesIO()
    with wave.open(canonical, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(frames)
    validate_wav(canonical.getvalue())
    atomic_write(target, canonical.getvalue())


def build_page(output, region, ffmpeg):
    trials = []
    for index, (category, text, note) in enumerate(PROMPTS):
        clips = []
        for voice in VOICES:
            fingerprint = hashlib.sha256(json.dumps(
                [region, voice, text, FORMAT, "plain-zh-CN-v1"], ensure_ascii=False).encode()).hexdigest()
            clips.append({"voice": voice, "raw": f"raw/{fingerprint}.wav",
                          "normalized": f"normalized/{fingerprint}-lufs20-v1.wav"})
        trials.append({"id": str(index + 1), "category": category, "text": text, "note": note, "clips": clips})
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = {t["id"]: t for t in trials}
        for trial in manifest["trials"]:
            fresh = expected.pop(trial["id"])
            if (trial["text"], sorted(c["raw"] for c in trial["clips"])) != (
                    fresh["text"], sorted(c["raw"] for c in fresh["clips"])):
                raise ValueError("Configuration changed; choose a new --output directory")
        if expected:
            raise ValueError("Corpus changed; choose a new --output directory")
    else:
        rng = random.SystemRandom()
        # Counterbalance label positions within each category (four prompts).
        for start in range(0, len(trials), len(VOICES)):
            order = rng.sample(range(len(VOICES)), len(VOICES))
            rotations = rng.sample(range(len(VOICES)), len(VOICES))
            for trial, shift in zip(trials[start:start + len(VOICES)], rotations):
                trial["clips"] = [trial["clips"][order[(i + shift) % len(VOICES)]] for i in range(len(VOICES))]
        rng.shuffle(trials)
        manifest = {"id": os.urandom(12).hex(), "created": datetime.now(timezone.utc).isoformat(),
                    "region": region, "format": FORMAT, "normalization": "EBU R128, -20 LUFS, -2 dBTP",
                    "input": "Plain Chinese text, default delivery; no reading overrides", "trials": trials}
        atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2).encode())

    load_dotenv(ROOT / ".env")
    key = os.getenv("SPEECH_KEY")
    if not key:
        raise ValueError("Set SPEECH_KEY in the environment or project .env")
    with requests.Session() as session:
        for i, trial in enumerate(manifest["trials"], 1):
            for clip in trial["clips"]:
                raw, normalized = output / clip["raw"], output / clip["normalized"]
                if raw.exists():
                    validate_wav(raw.read_bytes())
                else:
                    atomic_write(raw, synthesize(session, trial["text"], clip["voice"], key, region))
                if not normalized.exists():
                    normalize(ffmpeg, raw, normalized)
                validate_wav(normalized.read_bytes())
            print(f"Prepared {i}/{len(manifest['trials'])} examples", flush=True)

    # The self-contained page can be opened from disk without a server.
    public = json.loads(json.dumps(manifest))
    for trial in public["trials"]:
        for clip in trial["clips"]:
            clip["audio"] = "data:audio/wav;base64," + base64.b64encode((output / clip["normalized"]).read_bytes()).decode()
            del clip["raw"], clip["normalized"]
    encoded = json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("tts_compare.html").read_text()
    atomic_write(output / "index.html", template.replace("__COMPARISON_DATA__", encoded).encode())
    print(f"Open {output.resolve() / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tts-comparison")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    region = os.getenv("SPEECH_REGION") or "eastus"
    if not re.fullmatch(r"[a-z0-9-]+", region):
        parser.error("Invalid SPEECH_REGION")
    if not args.ffmpeg:
        parser.error("ffmpeg is required: put it on PATH or pass --ffmpeg /path/to/ffmpeg")
    try:
        build_page(args.output, region, args.ffmpeg)
    except (ValueError, RuntimeError, requests.RequestException, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Comparison generation stopped: {error}\nCompleted clips are saved; rerun to resume.\n")


if __name__ == "__main__":
    main()
