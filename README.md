# Hanzi Friend

A multi-user spaced repetition app I built to help me learn 汉字. Words are
taught as character-pronunciation pairs (相-xiāng and 相-xiàng are separate
cards) using senses curated by an AI pass over CC-CEDICT, scheduled with
FSRS, with AI-generated translation sentences and Azure TTS audio. Words
you meet in the wild can be queued on the Teach page; they get introduced
ahead of the usual frequency order.

Confusable characters (component-based similarity: 未/末, 拢/龙) are
actively drilled rather than avoided: new words that resemble something
you know get a contrast quiz, intros and definitions warn about learned
lookalikes and homophones, and decomposition quizzes pick distractors
that resemble the real components.

## Configuration

The app is bring-your-own-key: each user supplies their own OpenAI and
Azure Speech keys on the Settings page, and can't study until they do.
Login and registration are gated by a browser proof-of-work challenge.

Speech uses `zh-CN-Xiaochen:DragonHDLatestNeural`. Word cards send their
selected pinyin reading as an Azure SSML pronunciation override, including
when the word follows a classifier (一只狗). Full sentences use contextual
pronunciation. Audio is cached under `static/audio/v2/` in the data directory,
keyed by voice, text, pronunciation, and output format. Older recordings
remain on disk but are not reused.

Environment variables (a `.env` file works for development):

- `FLASK_SECRET` -- session and proof-of-work signing key
- `INVITE_CODE` -- enables registration; leave unset to keep it closed
- `POW_DIFFICULTY` -- leading zero bits for the login proof-of-work (default 15)
- `HANZI_DATA_DIR` -- where `hanzi.db` and the audio cache live (default `.`)
- `OPENAI_API_KEY` -- only for offline scripts like the lexicon backfill;
  the app itself always uses the logged-in user's key

## Running

Development: `nix develop`, then `python app.py`.

Production: `nix run`, or on NixOS import `nixosModules.default` and set:

```nix
services.hanzi-friend = {
    enable = true;
    port = 8089;
    environmentFile = "/run/secrets/hanzi-friend.env";
};
```

The service listens on localhost; front it with your reverse proxy.

## Blind Mandarin voice comparison

Run `python tools/tts_compare.py` inside `nix develop`, with `ffmpeg` on
PATH (or pass `--ffmpeg /path/to/ffmpeg`). It uses `SPEECH_KEY` and
`SPEECH_REGION` from the environment or project `.env`. This makes paid
Azure synthesis calls: the first complete run generates 80 short clips.
Reruns reuse completed clips and preserve the randomized assignment.

Open `tts-comparison/index.html` in a browser. The generated page contains
all audio and works offline. Rate naturalness and pronunciation separately,
then choose a preferred version for each example. Ratings stay in browser
storage; use **Download ratings** for a portable backup. **Finish and reveal
voices** locks ratings and shows overall and per-category results. The
download includes the voice identities, so inspect it after listening.

This round compares Xiaoxiao Neural (the current voice), Xiaoxiao Dragon HD
Flash, Xiaochen Dragon HD, and Yunxi Dragon HD Flash. MiniMax is not included.
The 20 prompts cover characters, counted phrases, contextual readings, tone
changes, and sentences. Voice positions are balanced within each category
and prompt order is shuffled. All voices receive the same plain Chinese
text at default delivery, without pinyin or pronunciation overrides. The
ambiguous isolated 相 accepts either reading; selecting a particular card's
reading needs a separate test.

Audio is generated as 24 kHz mono PCM and matched to -20 LUFS / -2 dBTP
using two-pass FFmpeg loudness normalization. This tests synthesis quality
without the app's MP3 compression. Original WAVs and the assignment manifest
remain in the ignored `tts-comparison/` directory. Identities are hidden by
the UI, not encrypted; avoid inspecting the page source or manifest until
you finish. Use a different `--output` directory for a fresh comparison.
This experiment does not change the app's voice, database, or audio cache.

Voice availability and controls: [Azure HD voices](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/high-definition-voices)
and [Speech REST API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech).

## Migrating from the TinyDB era

```
python -m migrate.to_sqlite <username> [db.json]
```

creates your user and imports cards, review history, and the lexicon into
`HANZI_DATA_DIR/hanzi.db`.

## Todo

### Character Order
This app uses the optimal character learning order computed by Loach and Wang in [doi:10.1371/journal.pone.0163623](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5051716/). This order is based on usage frequency subject to the constraint of learning character components before they appear in other characters. A problem with this learning order is that it can create (by chance) clusters of characters with similar pronunciations or shapes which can make them hard to distinguish when learning. Confusables are now detected (similarity.py) and drilled via contrast quizzes; recomputing the order itself with a dissimilarity constraint remains open. `data/confusion` holds the earlier DINOv2 embedding experiment, which the component-overlap approach outperformed ~4x on recall.
