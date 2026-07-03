import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect

load_dotenv()

import db as store
import quiz
import tts
from auth import bp as auth_bp, login_required, current_user
from hanzi import (
    hanzi_info,
    generate_component_test,
    fixed_tone_convert,
    recognition_only,
    is_known_word,
)
from translation import Translation

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or os.urandom(24)
app.register_blueprint(auth_bp)


def render_quiz(pick: quiz.QuizPick):
    user = current_user()
    info = hanzi_info(pick.word, pick.reading, user_id=user)

    common = dict(
        hanzi=info.hanzi,
        pinyin=info.pinyin,
        meaning=info.meaning,
        glosses=info.glosses,
        decomposition=generate_component_test(info.decomposition),
        pinyin_numbers=info.pinyin_variants,
        quiz_type=pick.quiz_type,
        user_definition=info.user_definition,
        story=info.story,
        remaining_cards=pick.remaining,
        card_id=pick.card_id,
        reading=pick.reading,
        reading_display=fixed_tone_convert(pick.reading) if pick.reading else "",
        is_component=recognition_only(pick.word),
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
@login_required
def index():
    return render_template(
        "hanzi.html",
    )


@app.route("/learn", methods=["POST"])
@login_required
def start():
    return render_next()


@app.route("/learn/<hanzi>", methods=["POST"])
@login_required
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
@login_required
def teach():
    return render_teach()


@app.route("/teach", methods=["POST"])
@login_required
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
@login_required
def rate(card_id, difficulty):
    quiz.review(current_user(), card_id, difficulty)
    return render_next()


@app.route("/table")
@login_required
def table():
    return render_template("chart.html")


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
@login_required
def pronounce(text):
    return tts.pronounce(text)


if __name__ == "__main__":
    app.run(debug=True, port=8089)
