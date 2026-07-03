# Hanzi Friend

A multi-user spaced repetition app I built to help me learn 汉字. Words are
taught as character-pronunciation pairs (相-xiāng and 相-xiàng are separate
cards) using senses curated by an AI pass over CC-CEDICT, scheduled with
FSRS, with AI-generated translation sentences and Azure TTS audio. Words
you meet in the wild can be queued on the Teach page; they get introduced
ahead of the usual frequency order.

## Configuration

Environment variables (a `.env` file works for development):

- `OPENAI_API_KEY` -- lexicon enrichment and translation sentences
- `SPEECH_KEY` -- Azure TTS key
- `FLASK_SECRET` -- session signing key
- `INVITE_CODE` -- enables registration; leave unset to keep it closed
- `HANZI_DATA_DIR` -- where `hanzi.db` and the audio cache live (default `.`)

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

## Migrating from the TinyDB era

```
python -m migrate.to_sqlite <username> [db.json]
```

creates your user and imports cards, review history, and the lexicon into
`HANZI_DATA_DIR/hanzi.db`.

## Todo

### Character Order
This app uses the optimal character learning order computed by Loach and Wang in [doi:10.1371/journal.pone.0163623](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5051716/). This order is based on usage frequency subject to the constraint of learning character components before they appear in other characters. A problem with this learning order is that it can create (by chance) clusters of characters with similar pronunciations or shapes which can make them hard to distinguish when learning. A future improvement would involve recomputing the character order subject to a constraint that nearby characters not be too similar (see `data/confusion` for an embedding-based similarity experiment).
