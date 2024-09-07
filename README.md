# Hanzi Friend

This is a spaced repetition app I learned to help me learn 汉字. To get it working, you will need a .env file with the following keys:

- `OPENAI_API_KEY`
- `SPEECH_KEY` -- Azure TTS key
- `SPEECH_REGION` -- Azure TTS region

## Todo
### Character Order
This app uses the optimal character learning order computed by Loach and Wang in [doi:10.1371/journal.pone.0163623](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5051716/). This order is based on usage frequency subject to the constraint of learning character components before they appear in other characters. A problem with this learning order is that it can create (by chance) clusters of characters with similar pronunciations which can make them hard to distinguish when learning. A future improvement would involve recomputing the character order subject to a constraint that nearby characters not be too similar in pronunciation.

### Distinction by Meaning
The app is currently designed around the idea of teaching 汉字. However there are many characters which have multiple meanings with their own pronunciation. For example, 相 can mean "one another" when pronounced xiang1 but "appearance" when pronounced xiang4. An improvement would be to teach character-pronunciation pairs instead of just characters.

### Database Migration/Session Support
Currently the app assumes there is only one user and is designed to be run locally. Adding session support and switching to a more robust database could allow Hanzi Friend to be hosted, which would also have the benefit of letting you practice on any device.

### Handling of Radicals, Antiquated Words, etc.
Because of the character order approach the app will sometimes attempt to teach radicals that aren't used by themselves, antiquated words, etc. A better approach would filter out antiquated meanings and only quiz the meaning of radicals.


### Minor Bugs
- Always marks pronunciation answers wrong on multi-character words or characters with multiple pronunciations.
- AI sometimes generates phrases which do not contain the target word.
- AI sometimes generates malformatted output.
- Code cleanup
