"""Proof-of-work gate for the public login and register forms.

The form carries an HMAC-signed challenge (nonce, expiry, difficulty);
the browser brute-forces a solution such that
sha256("nonce:solution") has at least `difficulty` leading zero bits.
Nonces are one-shot, so a solved challenge can't be replayed. This
doesn't stop a determined attacker, it makes hammering the form cost
real CPU per attempt.
"""

import hashlib
import hmac
import os
import secrets
import time

from flask import current_app

DIFFICULTY = int(os.environ.get("POW_DIFFICULTY", "15"))
TTL = 120

_used = {}


def _secret() -> bytes:
    key = current_app.secret_key
    return key if isinstance(key, bytes) else key.encode()


def _sign(nonce: str, expiry: int, difficulty: int) -> str:
    payload = "{}.{}.{}".format(nonce, expiry, difficulty).encode()
    return hmac.new(_secret(), payload, hashlib.sha256).hexdigest()


def challenge():
    nonce = secrets.token_hex(16)
    expiry = int(time.time()) + TTL
    return {
        "nonce": nonce,
        "expiry": expiry,
        "difficulty": DIFFICULTY,
        "sig": _sign(nonce, expiry, DIFFICULTY),
    }


def _leading_zero_bits(digest: bytes) -> int:
    return 256 - int.from_bytes(digest, "big").bit_length()


def verify(form) -> bool:
    try:
        nonce = form["pow_nonce"]
        expiry = int(form["pow_expiry"])
        difficulty = int(form["pow_difficulty"])
        sig = form["pow_sig"]
        solution = form["pow_solution"]
    except (KeyError, ValueError):
        return False

    # The difficulty is signed, but pin it anyway so a stale challenge
    # from before a config change can't ride on its old easier setting.
    if difficulty != DIFFICULTY:
        return False
    if not hmac.compare_digest(sig, _sign(nonce, expiry, difficulty)):
        return False

    now = time.time()
    if expiry < now:
        return False

    for used_nonce, used_expiry in list(_used.items()):
        if used_expiry < now:
            del _used[used_nonce]
    if nonce in _used:
        return False

    digest = hashlib.sha256("{}:{}".format(nonce, solution).encode()).digest()
    if _leading_zero_bits(digest) < difficulty:
        return False

    _used[nonce] = expiry
    return True
