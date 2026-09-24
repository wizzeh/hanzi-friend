import html
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET

import requests

import app as application
from hanzi import Decomposition, HanziInfo
from quiz import QuizPick
import tts


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = patch.object(tts, "AUDIO_DIR", self.directory.name)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def response(self, chunks=(b"mp3-start", b"mp3-end")):
        response = Mock()
        response.iter_content.return_value = iter(chunks)
        return response

    def test_reading_applies_only_to_word_in_counted_phrase(self):
        root = ET.fromstring(tts.speech_ssml("一只狗", "gou3", "狗"))
        voice = root[0]
        self.assertEqual(voice.attrib["name"], "zh-CN-Xiaochen:DragonHDLatestNeural")
        self.assertEqual(voice.text, "一只")
        self.assertEqual(voice[0].text, "狗")
        self.assertEqual(voice[0].attrib, {"alphabet": "sapi", "ph": "gou 3"})

    def test_pinyin_forms_neutral_tones_and_erhua(self):
        cases = {"xiang4": "xiang 4", "xiàng": "xiang 4", " NU:3 ": "nv 3",
                 "nǚ": "nv 3", "lve4": "lue 4", "hai2 zi": "hai 2 zi 5",
                 "huar1": "hua r 1", "hua1 r5": "hua r 1"}
        for reading, expected in cases.items():
            with self.subTest(reading=reading):
                self.assertEqual(tts.sapi_pinyin(reading), expected)

    def test_plain_sentences_are_escaped_without_phonemes(self):
        text = '你好，"水" & <火>。'
        voice = ET.fromstring(tts.speech_ssml(text))[0]
        self.assertEqual(voice.text, text)
        self.assertEqual(len(voice), 0)

    def test_invalid_reading_or_target_fails_before_api_call(self):
        for reading, word in [("", "相"), ('xiang4\"/><audio>', "相"), ("xiang4", "狗")]:
            with self.subTest(reading=reading), patch.object(tts.requests, "post") as post:
                with self.assertRaises(ValueError):
                    tts.pronounce("相", "key", "eastus", reading, word)
                post.assert_not_called()

    def test_cache_separates_readings_voice_and_format(self):
        first = tts.audio_path("相", "xiang1")
        self.assertNotEqual(first, tts.audio_path("相", "xiang4"))
        self.assertNotEqual(first, tts.audio_path("相"))
        self.assertEqual(first, tts.audio_path("相", "xiāng", "相"))
        self.assertEqual(tts.audio_path("女", "nv3"), tts.audio_path("女", "nǚ"))
        with patch.object(tts, "VOICE", "zh-CN-XiaoxiaoNeural"):
            self.assertNotEqual(first, tts.audio_path("相", "xiang1"))
        with patch.object(tts, "OUTPUT_FORMAT", "different"):
            self.assertNotEqual(first, tts.audio_path("相", "xiang1"))
        self.assertEqual(Path(tts.audio_path("../水")).parent, Path(self.directory.name))

    def test_readings_generate_distinct_requests_and_cache_independently(self):
        Path(self.directory.name, "相.mp3").write_bytes(b"legacy")
        responses = [self.response([b"first-tone"]), self.response([b"fourth-tone"])]
        with patch.object(tts.requests, "post", side_effect=responses) as post:
            self.assertEqual(b"".join(tts.pronounce("相", "key", "eastus", "xiang1")), b"first-tone")
            self.assertEqual(b"".join(tts.pronounce("相", "key", "eastus", "xiang4")), b"fourth-tone")
            self.assertEqual(tts.pronounce("相", "key", "eastus", "xiang1"), b"first-tone")
            self.assertEqual(tts.pronounce("相", "key", "eastus", "xiang4"), b"fourth-tone")
            self.assertEqual(post.call_count, 2)
            for call, tone in zip(post.call_args_list, [1, 4]):
                voice = ET.fromstring(call.kwargs["data"].decode("utf-8"))[0]
                self.assertEqual(voice[0].attrib["ph"], f"xiang {tone}")
                self.assertTrue(call.kwargs["stream"])
        for response in responses:
            response.close.assert_called_once()

    def test_abort_does_not_publish_partial_audio(self):
        response = self.response()
        with patch.object(tts.requests, "post", return_value=response):
            audio = tts.pronounce("相", "key", "eastus", "xiang4")
            self.assertEqual(next(audio), b"mp3-start")
            audio.close()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        response.close.assert_called_once()

    def test_upstream_error_does_not_pollute_cache(self):
        response = self.response()
        response.raise_for_status.side_effect = requests.HTTPError("failed")
        with patch.object(tts.requests, "post", return_value=response), self.assertRaises(requests.HTTPError):
            tts.pronounce("相", "key", "eastus", "xiang4")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        response.close.assert_called_once()

    def test_empty_response_is_not_cached(self):
        with patch.object(tts.requests, "post", return_value=self.response([])):
            self.assertEqual(b"".join(tts.pronounce("相", "key", "eastus")), b"")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_sentences_do_not_fill_word_cache(self):
        response = self.response()
        with patch.object(tts.requests, "post", return_value=response):
            self.assertEqual(b"".join(tts.pronounce("你好，今天怎么样？", "key", "eastus")), b"mp3-startmp3-end")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        response.close.assert_called_once()


class SpeechRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = application.app.test_client()
        with self.client.session_transaction() as session:
            session["user_id"] = 1
        keys = patch.object(application.store, "user_keys", return_value={
            "openai_api_key": "test", "speech_key": "test", "speech_region": "eastus"})
        keys.start()
        self.addCleanup(keys.stop)

    def test_route_passes_target_and_reading_and_serves_mp3(self):
        with patch.object(tts, "pronounce", return_value=iter([b"audio"])) as pronounce:
            response = self.client.get("/pronounce/一只狗", query_string={"word": "狗", "reading": "gou3"})
            self.assertEqual(response.data, b"audio")
            self.assertEqual(response.mimetype, "audio/mpeg")
            pronounce.assert_called_once_with("一只狗", "test", "eastus", reading="gou3", word="狗")

    def test_plain_sentence_has_no_override(self):
        with patch.object(tts, "pronounce", return_value=b"audio") as pronounce:
            response = self.client.get("/pronounce/你好，今天怎么样？")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(pronounce.call_args.kwargs, {"reading": None, "word": None})

    def test_bad_input_and_provider_failures_have_distinct_status(self):
        response = self.client.get("/pronounce/相", query_string={"reading": "xiang4", "word": "狗"})
        self.assertEqual(response.status_code, 400)
        with patch.object(tts, "pronounce", side_effect=requests.Timeout("private detail")):
            response = self.client.get("/pronounce/相")
            self.assertEqual(response.status_code, 502)
            self.assertNotIn(b"private detail", response.data)

    def test_quiz_html_carries_reading_through_counted_phrase(self):
        info = HanziInfo("狗", "gǒu", ["gou3"], ["dog"], ["dog"], Decomposition([], 0), "", "")
        with application.app.test_request_context("/"), \
             patch.object(application, "current_user", return_value=1), \
             patch.object(application, "hanzi_info", return_value=info), \
             patch.object(application, "generate_component_test", return_value=[]), \
             patch.object(application, "lookalikes_for", return_value=[]), \
             patch.object(application.store, "characters_seen", return_value=1), \
             patch.object(application.store, "user_words", return_value=[]), \
             patch.object(application.classifiers, "speak_phrase", return_value="一只狗"):
            for kind in ("intro", "pronunciation", "meaning", "classifier", "contrast"):
                with self.subTest(kind=kind):
                    rendered = application.render_quiz(QuizPick("狗", kind, "gou3", None, 1))
                    url = html.unescape(re.search(r'<audio src="([^"]+)" id="pronounce"', rendered)[1])
                    self.assertEqual(parse_qs(urlparse(url).query), {"reading": ["gou3"], "word": ["狗"]})
                    with patch.object(tts, "pronounce", return_value=b"audio") as pronounce:
                        self.assertEqual(self.client.get(url).status_code, 200)
                        self.assertEqual(pronounce.call_args.args[0], "一只狗")


if __name__ == "__main__":
    unittest.main()
