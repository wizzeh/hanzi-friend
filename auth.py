import os
from functools import wraps

from flask import Blueprint, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash

import db as store

bp = Blueprint("auth", __name__)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return view(*args, **kwargs)

    return wrapped


def current_user() -> int:
    return session["user_id"]


@bp.route("/login", methods=["GET"])
def login():
    return render_template("login.html", error=None)


@bp.route("/login", methods=["POST"])
def login_post():
    user = store.get_user(request.form.get("username", ""))
    if user is None or not check_password_hash(
        user["password_hash"], request.form.get("password", "")
    ):
        return render_template("login.html", error="Wrong username or password.")

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return redirect("/")


@bp.route("/register", methods=["GET"])
def register():
    return render_template("register.html", error=None)


@bp.route("/register", methods=["POST"])
def register_post():
    # Registration stays closed unless an invite code is configured.
    invite_code = os.environ.get("INVITE_CODE")
    if not invite_code:
        return render_template("register.html", error="Registration is closed.")
    if request.form.get("invite", "") != invite_code:
        return render_template("register.html", error="Wrong invite code.")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not username or not password:
        return render_template(
            "register.html", error="Username and password are required."
        )
    if store.get_user(username) is not None:
        return render_template("register.html", error="Username is taken.")

    user_id = store.create_user(username, generate_password_hash(password))
    session["user_id"] = user_id
    session["username"] = username
    return redirect("/")


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/login")
