import hashlib
import os
import re
import tempfile
from xml.sax.saxutils import escape, quoteattr

import requests
from pypinyin.contrib.tone_convert import to_tone3

from db import DATA_DIR
from loach_word_order import word_order

AUDIO_DIR = os.path.join(DATA_DIR, "static", "audio", "v2")
os.makedirs(AUDIO_DIR, exist_ok=True)

VOICE = "zh-CN-Xiaochen:DragonHDLatestNeural"
OUTPUT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"


def sapi_pinyin(reading: str) -> str:
    """Convert our numbered/marked pinyin to Azure's Mandarin phone set.

    Azure spells 女 nv 3, 略 lue 4, and erhua 花儿 hua r 1.
    Dictionary tones are retained; this selects a reading, not a new tone
    sandhi engine. See Microsoft's speech-ssml-phonetic-sets#zh-cn.
    """
    phones = []
    for syllable in reading.lower().replace("u:", "v").split():
        numbered = to_tone3(syllable, neutral_tone_with_five=True)
        match = re.fullmatch(r"([a-z]+)([1-5])", numbered)
        if not match:
            raise ValueError("Invalid pinyin reading")
        base, tone = match.groups()
        base = base.replace("ve", "ue")
        if base == "r" and phones:
            # CC-CEDICT can spell the erhua suffix as a separate r5.
            previous, previous_tone = phones.pop().rsplit(" ", 1)
            phones.append(f"{previous} r {previous_tone}")
        elif base.endswith("r") and base not in ("er", "r"):
            phones.append(f"{base[:-1]} r {tone}")
        else:
            phones.append(f"{'er' if base == 'r' else base} {tone}")
    if not phones:
        raise ValueError("Empty pinyin reading")
    return " ".join(phones)


def speech_ssml(text: str, reading: str = None, word: str = None) -> str:
    body = escape(text)
    if reading is not None:
        target = text if word is None else word
        if not target or not text.endswith(target):
            raise ValueError("The pronunciation target must end the spoken phrase")
        prefix = escape(text[:-len(target)])
        body = (f'{prefix}<phoneme alphabet="sapi" ph={quoteattr(sapi_pinyin(reading))}>'
                f'{escape(target)}</phoneme>')
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="zh-CN"><voice name={quoteattr(VOICE)}>{body}</voice></speak>')


def audio_path(text: str, reading: str = None, word: str = None) -> str:
    # Hash the actual synthesis input: voice, pronunciation, target, text,
    # and output format. Legacy text-only recordings never match this cache.
    request = OUTPUT_FORMAT + "\n" + speech_ssml(text, reading, word)
    digest = hashlib.sha256(request.encode("utf-8")).hexdigest()
    return os.path.join(AUDIO_DIR, digest + ".mp3")


def pronounce(text: str, key: str, region: str, reading: str = None, word: str = None):
    """Return mp3 audio for our text, from the cache when we have it,
    else fetched on the user's own Azure Speech key.

    Single words and short counted phrases (一只狗, which drills the
    measure word for free) get cached on first fetch; full sentences
    are streamed through without caching."""
    ssml = speech_ssml(text, reading, word)
    path = audio_path(text, reading, word)
    cache = False
    # A phrase is only worth caching if it's short and unpunctuated --
    # a fixed little chunk we'll replay often, not a one-off sentence.
    short_phrase = len(text) <= 8 and not any(
        c in text for c in "。，、；：！？"
    )
    if text in word_order or short_phrase:
        try:
            with open(path, "rb") as f:
                data = f.read()
            # Treat an empty file as a miss: past versions left 0-byte
            # artifacts behind, and serving one would stick forever.
            if data:
                return data
        except FileNotFoundError:
            pass
        cache = True

    def cache_as_stream(response):
        tmp = None
        try:
            content = response.iter_content(chunk_size=4096)
            if not cache:
                yield from content
                return

            # Publish only after the entire response has been consumed.
            fd, tmp = tempfile.mkstemp(dir=AUDIO_DIR, suffix=".tmp")
            size = 0
            with os.fdopen(fd, "wb") as f:
                for item in content:
                    f.write(item)
                    size += len(item)
                    yield item
            if size:
                os.replace(tmp, path)
        finally:
            response.close()
            if tmp is not None:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass

    host = "{}.tts.speech.microsoft.com".format(region or "eastus")
    url = "https://{}/cognitiveservices/v1".format(host)

    headers = {
        "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
        "Content-Type": "application/ssml+xml",
        "Host": host,
        "Ocp-Apim-Subscription-Key": key,
        "User-Agent": "hanzi-tts",
    }

    r = requests.post(url, headers=headers, data=ssml.encode("utf-8"),
                      stream=True, timeout=(10, 60))
    # A failed call must never reach the cache (or the client): before
    # this check, an Azure error body would be saved as the word's mp3
    # and served from the cache forever after.
    try:
        r.raise_for_status()
    except requests.RequestException:
        r.close()
        raise

    return cache_as_stream(r)
