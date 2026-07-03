import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect

load_dotenv()

import db as store
import quiz
import tts
from hanzi import (
    hanzi_info,
    generate_component_test,
    fixed_tone_convert,
    recognition_only,
)
from translation import Translation

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET")


def render_quiz(pick: quiz.QuizPick):
    info = hanzi_info(pick.word, pick.reading)

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
        translation = Translation.for_hanzi(pick.word, pick.reading, info.glosses)
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
            words_known=store.characters_seen(),
            **common,
        )


def render_next():
    return render_quiz(quiz.next_quiz())


@app.route("/")
def index():
    return render_template(
        "hanzi.html",
    )


@app.route("/learn", methods=["POST"])
def start():
    return render_next()


@app.route("/learn/<hanzi>", methods=["POST"])
def learn(hanzi):
    reading = request.args.get("reading")
    quiz.learn_word(hanzi, reading)
    if reading is None:
        store.increment_characters_seen()
    return render_next()


@app.route("/teach", methods=["GET"])
def teach():
    return render_template("teach.html")


@app.route("/teach", methods=["POST"])
def teach_post():
    hanzi = request.form.get("word")
    if hanzi:
        quiz.learn_word(hanzi)

    return redirect("/")


@app.route("/review/<int:card_id>/<difficulty>", methods=["POST"])
def rate(card_id, difficulty):
    quiz.review(card_id, difficulty)
    return render_next()


@app.route("/table")
def table():
    return render_template("chart.html")


@app.route("/difficult")
def difficult():
    return quiz.most_difficult_words()


@app.route("/<hanzi>/update-definition", methods=["POST"])
def set_definition(hanzi):
    store.set_user_definition(hanzi, request.form["definition"])

    return ("", 200)


@app.route("/pronounce/<text>", methods=["GET"])
def pronounce(text):
    return tts.pronounce(text)


if __name__ == "__main__":
    app.run(debug=True, port=8089)
