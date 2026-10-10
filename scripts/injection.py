"""Prompt-injection heuristics for text that came from outside this machine.

Warn-only by design: a pattern list cannot tell an attack from a ticket that
discusses one, so a hit becomes a note for Claude, never a block.
"""
import re

INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")
TAG_CHARS = re.compile("[\U000e0000-\U000e007f]")

PATTERNS = [(kind, re.compile(rx)) for kind, rx in [
    ("ignore-instructions", r"(?i)\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|system|your)\b[^.\n]{0,20}\b(?:instructions?|prompts?|rules|guidelines|messages)\b"),
    ("new-instructions", r"(?i)\b(?:new|updated|real|actual)\s+(?:system\s+)?instructions?\s*:"),
    ("role-change", r"(?i)\byou are now\b[^.\n]{0,30}\b(?:assistant|ai|agent|dan|developer mode|jailbroken|unrestricted)\b"),
    ("fake-role-tag", r"(?i)</?(?:system(?:-reminder)?|assistant|human|tool_result|function_results|instructions?)>"),
    ("fake-role-line", r"(?im)^\s*(?:system|assistant|human)\s*:\s"),
    ("hide-from-user", r"(?i)\b(?:do not|don't|never)\s+(?:tell|inform|mention|reveal|show)\b[^.\n]{0,20}\b(?:the )?(?:user|human|operator)\b"),
    ("exfiltration", r"(?i)\b(?:send|post|upload|forward|exfiltrate)\b[^\n]{0,40}(?:\b(?:credentials?|secrets?|tokens?|api keys?|passwords?|ssh keys?)\b|\.env\b)[^\n]{0,40}\b(?:to|at)\s+(?:https?://|\S+@\S+)"),
]]


def strip_invisible(text):
    """Remove zero-width, bidi-override and Unicode tag characters (hidden 'ASCII smuggling' text)."""
    return INVISIBLE.sub("", text)


def hits(text):
    """Kinds of instruction-like content in text; [] when it reads as plain data."""
    found = ["hidden-tag-characters"] if TAG_CHARS.search(text) else []
    stripped = strip_invisible(text)
    return found + [kind for kind, rx in PATTERNS if rx.search(stripped)]
