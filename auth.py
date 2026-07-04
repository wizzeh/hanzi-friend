import os
from functools import wraps

from flask import Blueprint, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash

import db as store
import pow as pow_gate

bp = Blueprint("auth", __name__)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return view(*args, **kwargs)

    return wrapped


def keys_required(view):
    """This instance is bring-your-own-key: no studying until the user
    has entered their API keys."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        keys = store.user_keys(session["user_id"])
        if not (keys["openai_api_key"] and keys["speech_key"]):
            return redirect("/settings")
        return view(*args, **kwargs)

    return wrapped


def current_user() -> int:
    return session["user_id"]


def render_login(error=None):
    return render_template("login.html", error=error, pow=pow_gate.challenge())


def render_register(error=None):
    return render_template("register.html", error=error, pow=pow_gate.challenge())


@bp.route("/login", methods=["GET"])
def login():
    return render_login()


@bp.route("/login", methods=["POST"])
def login_post():
    if not pow_gate.verify(request.form):
        return render_login(error="Anti-bot check failed; try again.")

    user = store.get_user(request.form.get("username", ""))
    if user is None or not check_password_hash(
        user["password_hash"], request.form.get("password", "")
    ):
        return render_login(error="Wrong username or password.")

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return redirect("/")


@bp.route("/register", methods=["GET"])
def register():
    return render_register()


@bp.route("/register", methods=["POST"])
def register_post():
    if not pow_gate.verify(request.form):
        return render_register(error="Anti-bot check failed; try again.")

    # Registration stays closed unless an invite code is configured.
    invite_code = os.environ.get("INVITE_CODE")
    if not invite_code:
        return render_register(error="Registration is closed.")
    if request.form.get("invite", "") != invite_code:
        return render_register(error="Wrong invite code.")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not username or not password:
        return render_register(error="Username and password are required.")
    if store.get_user(username) is not None:
        return render_register(error="Username is taken.")

    user_id = store.create_user(username, generate_password_hash(password))
    session["user_id"] = user_id
    session["username"] = username
    return redirect("/")


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/login")
