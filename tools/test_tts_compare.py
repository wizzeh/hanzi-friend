import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave
import xml.etree.ElementTree as ET

from tools import tts_compare as compare


def audio_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x10\x00\xf0" * 12000)
    return buffer.getvalue()


class ComparisonTests(unittest.TestCase):
    def test_ssml_preserves_chinese_and_escapes_markup(self):
        root = ET.fromstring(compare.ssml("女 & <水>", compare.VOICES[0]))
        self.assertEqual(root[0].text, "女 & <水>")
        self.assertEqual(root.attrib['{http://www.w3.org/XML/1998/namespace}lang'], 'zh-CN')

    def test_rejects_truncated_and_silent_audio(self):
        compare.validate_wav(audio_bytes())
        with self.assertRaises(ValueError):
            compare.validate_wav(audio_bytes()[:-100])
        with self.assertRaises(ValueError):
            compare.validate_wav(audio_bytes()[:44] + bytes(48000))

    def test_provider_failure_never_becomes_audio(self):
        session = Mock()
        session.post.return_value = Mock(ok=False, status_code=401)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            compare.synthesize(session, "水", compare.VOICES[0], "secret", "eastus")
        self.assertEqual(session.post.call_count, 1)

    def test_transient_failure_has_bounded_retry(self):
        session = Mock()
        session.post.return_value = Mock(ok=False, status_code=429)
        with patch.object(compare.time, "sleep"), self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            compare.synthesize(session, "水", compare.VOICES[0], "secret", "eastus")
        self.assertEqual(session.post.call_count, 3)

    def test_resuming_preserves_blinding_and_avoids_rebilling(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            def normalize(_ffmpeg, source, target):
                compare.atomic_write(target, source.read_bytes())
            with patch.dict(os.environ, {"SPEECH_KEY": "test-only"}), \
                 patch.object(compare, "synthesize", return_value=audio_bytes()) as synthesize, \
                 patch.object(compare, "normalize", side_effect=normalize), \
                 patch("builtins.print"):
                compare.build_page(output, "eastus", "ffmpeg")
                manifest = (output / "manifest.json").read_bytes()
                self.assertEqual(synthesize.call_count, len(compare.PROMPTS) * len(compare.VOICES))
                synthesize.reset_mock()
                compare.build_page(output, "eastus", "ffmpeg")
                synthesize.assert_not_called()
                self.assertEqual(manifest, (output / "manifest.json").read_bytes())
                self.assertNotIn("test-only", (output / "index.html").read_text())
                trials = json.loads(manifest)["trials"]
                self.assertEqual(len({t['id'] for t in trials}), len(compare.PROMPTS))
                for category in {t['category'] for t in trials}:
                    group = [t for t in trials if t['category'] == category]
                    for index in range(len(compare.VOICES)):
                        self.assertEqual({t['clips'][index]['voice'] for t in group}, set(compare.VOICES))
                # A region change must not silently reuse the original trial.
                with self.assertRaisesRegex(ValueError, "Configuration changed"):
                    compare.build_page(output, "westeurope", "ffmpeg")
                synthesize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
