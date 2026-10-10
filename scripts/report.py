"""Findings log and user notifications for tool-guard.

record() appends one JSON line per finding for later analysis: what was found,
where it came from and the scores, never the text or a secret value.
notify() pops a desktop notification on the machine running Claude Code, in the
background; it never blocks the hook and never raises.
"""
import base64
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

JIRA_KEY = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
# Windows only shows toasts from a registered app; PowerShell's own id always is.
WINDOWS_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
TOAST = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $x.GetElementsByTagName('text')
$t.Item(0).AppendChild($x.CreateTextNode('{title}')) | Out-Null
$t.Item(1).AppendChild($x.CreateTextNode('{body}')) | Out-Null
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{app}').Show([Windows.UI.Notifications.ToastNotification]::new($x))
"""


def record(path, entry):
    """Append one finding as a JSON line, stamped with local time and its UTC offset."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stamped = {"ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), **entry}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(stamped, ensure_ascii=False) + "\n")


def fingerprint(text):
    """Enough to group repeats of the same output without keeping it."""
    return {"chars": len(text), "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]}


def notify(title, body):
    """Best effort desktop notification; it never fails the hook.

    It runs in the foreground: on Windows a notifier started in the background dies
    with the hook process before the toast shows. Only confirmed findings notify, so
    the half second this costs is rare.
    """
    try:
        if os.name == "nt":
            script = TOAST.format(title=title.replace("'", "''"), body=body.replace("'", "''"), app=WINDOWS_APP_ID)
            encoded = base64.b64encode(script.encode("utf-16-le")).decode()
            argv = ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
        elif sys.platform == "darwin":
            argv = ["osascript", "-e", f"display notification {json.dumps(body)} with title {json.dumps(title)}"]
        else:
            argv = ["notify-send", title, body]
        subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except Exception:
        pass
