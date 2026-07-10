import os

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect

load_dotenv()

import db as store
import quiz
import stats
import tts
from auth import bp as auth_bp, login_required, keys_required, current_user
from hanzi import (
    hanzi_info,
    generate_component_test,
    fixed_tone_convert,
    recognition_only,
    is_known_word,
)
from translation import Translation
import similarity

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or os.urandom(24)
app.register_blueprint(auth_bp)


def describe_char(char: str):
    entry = store.get_lexicon(char)
    if not entry:
        return {"char": char, "reading": "", "gloss": ""}
    senses = [s for s in entry["senses"] if s["learn"]] or entry["senses"]
    return {
        "char": char,
        "reading": fixed_tone_convert(senses[0]["pinyin"]),
        "gloss": senses[0]["gloss"],
    }


def lookalikes_for(user: int, word: str):
    """Learned characters worth a 'don't confuse' warning: strong visual
    matches plus same-sounding characters."""
    if len(word) != 1:
        return []
    learned = [w for w in store.user_words(user) if w != word]

    visual = [
        dict(describe_char(c), kind="looks like")
        for c, score in similarity.top_visual(word, learned, 3)
        if score >= similarity.CONTRAST_THRESHOLD
    ]
    shown = {v["char"] for v in visual}
    phonetic = [
        dict(describe_char(c), kind="sounds like")
        for c in learned
        if c not in shown
        and len(c) == 1
        and similarity.phonetic_score(word, c) >= 1.0
    ][:3]
    return visual + phonetic


def render_quiz(pick: quiz.QuizPick):
    user = current_user()
    info = hanzi_info(pick.word, pick.reading, user_id=user)

    common = dict(
        hanzi=info.hanzi,
        pinyin=info.pinyin,
        meaning=info.meaning,
        glosses=info.glosses,
        decomposition=generate_component_test(info.decomposition, pick.word),
        pinyin_numbers=info.pinyin_variants,
        quiz_type=pick.quiz_type,
        user_definition=info.user_definition,
        story=info.story,
        remaining_cards=pick.remaining,
        card_id=pick.card_id,
        intervals=quiz.preview_intervals(pick.card_id) if pick.card_id else None,
        reading=pick.reading,
        reading_display=fixed_tone_convert(pick.reading) if pick.reading else "",
        is_component=recognition_only(pick.word),
        lookalikes=lookalikes_for(user, pick.word)
        if pick.quiz_type in ("intro", "meaning")
        else [],
    )

    if pick.quiz_type == "contrast":
        distractors = [
            c
            for c, _ in similarity.top_visual(
                pick.word, [w for w in store.user_words(user) if w != pick.word], 3
            )
        ]
        choices = distractors + [pick.word]
        from random import shuffle as _shuffle

        _shuffle(choices)
        return render_template(
            "quizzes/contrast.html",
            choices=choices,
            words_known=store.characters_seen(user),
            **common,
        )

    if pick.quiz_type.startswith("translation"):
        translation = Translation.for_hanzi(user, pick.word, pick.reading, info.glosses)
        return render_template(
            "quizzes/translation.html",
            to_translate=translation.english
            if pick.quiz_type.endswith("english")
            else translation.chinese,
            translation=translation.chinese
            if pick.quiz_type.endswith("english")
            else translation.english,
            is_english=pick.quiz_type.endswith("english"),
            **common,
        )
    else:
        return render_template(
            "quizzes/" + pick.quiz_type + ".html",
            words_known=store.characters_seen(user),
            **common,
        )


def render_next():
    return render_quiz(quiz.next_quiz(current_user()))


@app.route("/")
@keys_required
def index():
    return render_template(
        "hanzi.html",
    )


@app.route("/learn", methods=["POST"])
@keys_required
def start():
    return render_next()


@app.route("/learn/<hanzi>", methods=["POST"])
@keys_required
def learn(hanzi):
    reading = request.args.get("reading")
    # Pre-learned words don't advance our position in the frequency
    # order; the order catches up to them on its own.
    was_prelearned = store.remove_prelearn(current_user(), hanzi)
    quiz.learn_word(current_user(), hanzi, reading)
    if reading is None and not was_prelearned:
        store.increment_characters_seen(current_user())
    return render_next()


def render_teach(error=None):
    return render_template(
        "teach.html", error=error, queue=store.prelearn_queue(current_user())
    )


@app.route("/teach", methods=["GET"])
@keys_required
def teach():
    return render_teach()


@app.route("/teach", methods=["POST"])
@keys_required
def teach_post():
    word = request.form.get("word", "").strip()
    if not word or not is_known_word(word):
        return render_teach(error="Not in the dictionary: {}".format(word))
    if store.has_cards_for_word(current_user(), word):
        return render_teach(error="Already learning {}".format(word))

    store.queue_prelearn(current_user(), word)
    return redirect("/teach")


@app.route("/teach/<word>/remove", methods=["POST"])
@login_required
def teach_remove(word):
    store.remove_prelearn(current_user(), word)
    return redirect("/teach")


@app.route("/review/<int:card_id>/<difficulty>", methods=["POST"])
@keys_required
def rate(card_id, difficulty):
    quiz.review(current_user(), card_id, difficulty)
    return render_next()


@app.route("/settings", methods=["GET"])
@login_required
def settings():
    return render_template(
        "settings.html",
        keys=store.user_keys(current_user()),
        saved=request.args.get("saved"),
    )


@app.route("/settings", methods=["POST"])
@login_required
def settings_post():
    store.set_user_keys(
        current_user(),
        request.form.get("openai_api_key", "").strip(),
        request.form.get("speech_key", "").strip(),
        request.form.get("speech_region", "").strip(),
    )
    return redirect("/settings?saved=1")


@app.route("/table")
@login_required
def table():
    return render_template("chart.html")


@app.route("/stats")
@login_required
def stats_page():
    return render_template("stats.html", **stats.page_data(current_user(), describe_char))


@app.route("/difficult")
@login_required
def difficult():
    return quiz.most_difficult_words(current_user())


@app.route("/<hanzi>/update-definition", methods=["POST"])
@login_required
def set_definition(hanzi):
    store.set_user_definition(current_user(), hanzi, request.form["definition"])

    return ("", 200)


@app.route("/pronounce/<text>", methods=["GET"])
@keys_required
def pronounce(text):
    keys = store.user_keys(current_user())
    try:
        return tts.pronounce(text, keys["speech_key"], keys["speech_region"])
    except requests.HTTPError as e:
        # Bad key, rate limit, etc. The audio element ignores failures,
        # so a quiet 502 is the right shape for the client.
        return "speech synthesis failed: {}".format(e), 502


if __name__ == "__main__":
    app.run(debug=True, port=8089)
