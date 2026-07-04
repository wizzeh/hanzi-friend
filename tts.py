import os

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
                return f.read()
        except FileNotFoundError:
            cache = not text.strip().endswith("。")

    def cache_as_stream(content, do_cache):
        if not do_cache:
            yield from content
            return

        with open(audio_path(text), "wb") as f:
            for item in content:
                f.write(item)
                yield item

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

    return cache_as_stream(r.iter_content(chunk_size=128), cache)
