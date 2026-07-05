import os
import tempfile

import requests

from db import DATA_DIR
from loach_word_order import word_order

AUDIO_DIR = os.path.join(DATA_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

SSML = """
    <speak version='1.0' xml:lang='zh-CN'><voice xml:lang='en-US' xml:gender='Female'
        name='zh-CN-XiaoxiaoNeural'>
            {}
    </voice></speak>
    """


def audio_path(text: str) -> str:
    return os.path.join(AUDIO_DIR, "{}.mp3".format(text))


def pronounce(text: str, key: str, region: str):
    """Return mp3 audio for our text, from the cache when we have it,
    else fetched on the user's own Azure Speech key.

    Single words get cached on first fetch; full sentences are streamed
    through without caching."""
    cache = False
    if text in word_order:
        try:
            with open(audio_path(text), "rb") as f:
                data = f.read()
            # Treat an empty file as a miss: past versions left 0-byte
            # artifacts behind, and serving one would stick forever.
            if data:
                return data
        except FileNotFoundError:
            pass
        cache = not text.strip().endswith("。")

    def cache_as_stream(content, do_cache):
        if not do_cache:
            yield from content
            return

        # Write through a temp file and rename only once the stream
        # completes, so an aborted download can't cache a truncated mp3.
        fd, tmp = tempfile.mkstemp(dir=AUDIO_DIR, suffix=".tmp")
        done = False
        try:
            with os.fdopen(fd, "wb") as f:
                for item in content:
                    f.write(item)
                    yield item
            done = True
        finally:
            if done:
                os.replace(tmp, audio_path(text))
            else:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass

    host = "{}.tts.speech.microsoft.com".format(region or "eastus")
    url = "https://{}/cognitiveservices/v1".format(host)

    headers = {
        "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        "Content-Type": "application/ssml+xml",
        "Host": host,
        "Ocp-Apim-Subscription-Key": key,
        "User-Agent": "hanzi-tts",
    }

    r = requests.post(url, headers=headers, data=SSML.format(text))
    # A failed call must never reach the cache (or the client): before
    # this check, an Azure error body would be saved as the word's mp3
    # and served from the cache forever after.
    r.raise_for_status()

    return cache_as_stream(r.iter_content(chunk_size=128), cache)
