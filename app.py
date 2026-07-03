import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect

load_dotenv()

import db as store
import quiz
import tts
from hanzi import hanzi_info, generate_component_test
from translation import Translation

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET")


def render_quiz(info, quiz_type, remaining_cards):
    if quiz_type.startswith("translation"):
        translation = Translation.for_hanzi(info.hanzi)
        return render_template(
            "quizzes/translation.html",
            hanzi=info.hanzi,
            pinyin=info.pinyin,
            meaning=info.meaning,
            decomposition=generate_component_test(info.decomposition),
            pinyin_numbers=info.pinyin_variants,
            quiz_type=quiz_type,
            to_translate=translation.english
            if quiz_type.endswith("english")
            else translation.chinese,
            translation=translation.chinese
            if quiz_type.endswith("english")
            else translation.english,
            is_english=quiz_type.endswith("english"),
            user_definition=info.user_definition,
            remaining_cards=remaining_cards,
        )
    else:
        return render_template(
            "quizzes/" + quiz_type + ".html",
            hanzi=info.hanzi,
            pinyin=info.pinyin,
            meaning=info.meaning,
            decomposition=generate_component_test(info.decomposition),
            pinyin_numbers=info.pinyin_variants,
            quiz_type=quiz_type,
            words_known=store.characters_seen(),
            user_definition=info.user_definition,
            remaining_cards=remaining_cards,
        )


def render_next():
    word, quiz_type, remaining_cards = quiz.next_quiz()
    return render_quiz(hanzi_info(word), quiz_type, remaining_cards)


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
    quiz.learn_word(hanzi)
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


@app.route("/<hanzi>/<quiz_type>/<difficulty>", methods=["POST"])
def rate(hanzi, quiz_type, difficulty):
    quiz.review(hanzi, quiz_type, difficulty)
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
