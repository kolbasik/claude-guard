"""The scoring server and guard.py's client for it, with a fake classifier: no ONNX needed."""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import guard  # noqa: E402
import server  # noqa: E402


class FakeModel:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def score(self, text):
        self.calls += 1
        return 0.95 if "MARKER_HIGH" in text else 0.1


class FakeJudge:
    """An OpenAI-compatible chat endpoint that always answers with a fixed verdict."""

    def __init__(self, verdict):
        self.calls = 0
        judge = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_POST(self):
                judge.calls += 1
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.dumps({"choices": [{"message": {"content": json.dumps({"injection": verdict})}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class Running:
    """A scoring server on a free local port for the length of a with-block."""

    def __init__(self, token="", judge_url=None, judge_above=0.0, confirmers=(), confirm_above=0.5,
                 claude_cmd=None, claude_token_file=None, screens=(), screen_above=0.5, claude_above=0.9):
        self.model = FakeModel()
        self.cwd = tempfile.mkdtemp()
        args = SimpleNamespace(host="127.0.0.1", port=0, judge_url=judge_url, judge_model="fake-judge" if judge_url else None,
                               judge_timeout=5, judge_above=judge_above, confirm_above=confirm_above,
                               claude_model="haiku" if claude_cmd else None, claude_cmd=claude_cmd, claude_above=claude_above,
                               screen_above=screen_above,
                               claude_timeout=30, claude_token_file=claude_token_file,
                               claude_cwd=os.path.join(self.cwd, "judge-cwd"), claude_config=os.path.join(self.cwd, "judge-config"))
        with mock.patch.dict(os.environ, {"TOOL_GUARD_TOKEN": token}):
            self.server = server.build_server(args, [self.model], confirmers, screens)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def post(url, text, token=None):
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    req = urllib.request.Request(url + "/score", json.dumps({"text": text}).encode(), headers)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def guard_rules(url, token=""):
    cfg = guard.default_config()
    cfg["classifier_url"] = url
    with mock.patch.dict(os.environ, {"TOOL_GUARD_TOKEN": token}):
        return guard.build_rules(cfg, [])


def web_output(text):
    return {"tool_name": "WebFetch", "tool_input": {"url": "https://example.com"}, "tool_response": text}


class ServerTests(unittest.TestCase):
    def test_score_returns_each_model(self):
        # Arrange
        with Running() as s:
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual(reply["scores"], {"fake": 0.95})

    def test_token_is_required_when_set(self):
        # Arrange
        with Running(token="t0ken-for-test") as s:
            # Act
            with self.assertRaises(urllib.error.HTTPError) as err:
                post(s.url, "text")
        # Assert
        self.assertEqual(err.exception.code, 401)

    def test_right_token_is_accepted(self):
        # Arrange
        with Running(token="t0ken-for-test") as s:
            # Act
            reply = post(s.url, "text", token="t0ken-for-test")
        # Assert
        self.assertIn("fake", reply["scores"])


class FakeConfirmer:
    name = "second"

    def __init__(self, value=0.97, fail=False):
        self.calls, self.value, self.fail = 0, value, fail

    def score(self, text):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return self.value


class ConfirmTests(unittest.TestCase):
    def test_suspect_text_gets_second_opinion(self):
        # Arrange
        second = FakeConfirmer()
        with Running(confirmers=[second]) as s:
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual(reply["scores"], {"fake": 0.95, "second": 0.97})

    def test_plain_text_skips_second_opinion(self):
        # Arrange
        second = FakeConfirmer()
        with Running(confirmers=[second]) as s:
            # Act
            post(s.url, "an ordinary page")
        # Assert
        self.assertEqual(second.calls, 0)

    def test_second_opinion_down_is_reported_not_fatal(self):
        # Arrange
        second = FakeConfirmer(fail=True)
        with Running(confirmers=[second]) as s:
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual(reply["scores"], {"fake": 0.95, "second_error": "ConnectionError"})


FAKE_CLAUDE = r'''
import json, os, sys
text = sys.stdin.read()
def reply(**out):
    print(json.dumps(out)); sys.exit(0)
expected = os.environ.get("FAKE_CLAUDE_EXPECT_TOKEN")
if expected and os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") != expected:
    reply(is_error=True, result="Not logged in - Please run /login", terminal_reason="api_error")
if os.listdir("."):
    reply(is_error=True, result="working folder is not empty", terminal_reason="test")
if not os.environ.get("CLAUDE_CONFIG_DIR", "").endswith("judge-config") or os.environ.get("DISABLE_TELEMETRY") != "1":
    reply(is_error=True, result="judge not isolated from the user config", terminal_reason="test")
injection = ("MARKER_HIGH" in text or "MARKER_CLAUDE_YES" in text) and "MARKER_ORDINARY" not in text
reply(is_error=False, result=json.dumps({"injection": injection}))
'''


def fake_claude():
    path = os.path.join(tempfile.mkdtemp(), "fake_claude.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(FAKE_CLAUDE)
    return [sys.executable, path]


class ClaudeJudgeTests(unittest.TestCase):
    def test_command_is_locked_down(self):
        # Arrange
        cli = ["claude"]
        # Act
        cmd = server.claude_command(cli, "haiku")
        # Assert
        joined = " ".join(f"[{a}]" for a in cmd)
        self.assertTrue(all(flag in joined for flag in ("[--thinking] [disabled]", "[--tools] []", "[--strict-mcp-config]",
                                                        "[--setting-sources] []", "[--no-session-persistence]")))

    def test_flagged_text_gets_claude_verdict(self):
        # Arrange
        with Running(claude_cmd=fake_claude()) as s:
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual(reply["scores"]["claude"], 1.0)

    def test_plain_text_never_reaches_claude(self):
        # Arrange
        with Running(claude_cmd=fake_claude()) as s:
            # Act
            reply = post(s.url, "an ordinary page")
        # Assert
        self.assertEqual(set(reply["scores"]), {"fake"})

    def test_not_logged_in_is_reported(self):
        # Arrange
        with Running(claude_cmd=fake_claude()) as s, mock.patch.dict(os.environ, {"FAKE_CLAUDE_EXPECT_TOKEN": "tok-for-test"}):
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertIn("Not logged in", reply["scores"]["claude_error"])

    def test_token_file_logs_claude_in(self):
        # Arrange
        token_file = os.path.join(tempfile.mkdtemp(), "token")
        with open(token_file, "w") as f:
            f.write("tok-for-test\n")
        with Running(claude_cmd=fake_claude(), claude_token_file=token_file) as s, \
                mock.patch.dict(os.environ, {"FAKE_CLAUDE_EXPECT_TOKEN": "tok-for-test"}):
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual(reply["scores"]["claude"], 1.0)

    def test_disagreeing_claude_annotates_but_keeps_warning(self):
        # Arrange
        with Running(claude_cmd=fake_claude()) as s:
            event = web_output("A long enough page body with MARKER_HIGH and MARKER_ORDINARY in it.")
            # Act
            result = guard.on_output(event, guard_rules(s.url))
        # Assert
        note = result["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(("model:fake" in note, "claude-judge:ordinary-data" in note), (True, True))

    def test_top_ignores_error_entries(self):
        # Arrange
        scores = {"a": 0.3, "b_error": "ConnectionError"}
        # Act
        result = server.top(scores)
        # Assert
        self.assertEqual(result, 0.3)


class FakeScreen:
    name = "screen-model"

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def score(self, text):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return 0.95 if "MARKER_SCREEN" in text else 0.1


class ScreenTests(unittest.TestCase):
    def test_screen_scores_every_text(self):
        # Arrange
        with Running(claude_cmd=fake_claude(), screens=[FakeScreen()]) as s:
            # Act
            reply = post(s.url, "an ordinary page")
        # Assert
        self.assertEqual(reply["scores"], {"fake": 0.1, "screen:screen-model": 0.1})

    def test_screen_hit_sends_text_to_claude(self):
        # Arrange
        with Running(claude_cmd=fake_claude(), screens=[FakeScreen()]) as s:
            # Act
            reply = post(s.url, "MARKER_SCREEN and MARKER_CLAUDE_YES in a page")
        # Assert
        self.assertEqual((reply["scores"]["fake"], reply["scores"]["claude"]), (0.1, 1.0))

    def test_screen_hit_alone_never_flags(self):
        # Arrange
        with Running(claude_cmd=fake_claude(), screens=[FakeScreen()]) as s:
            event = web_output("A long enough page body where only MARKER_SCREEN fires.")
            # Act
            result = guard.on_output(event, guard_rules(s.url))
        # Assert
        self.assertIsNone(result)

    def test_screen_hit_confirmed_by_claude_flags_as_claude(self):
        # Arrange
        with Running(claude_cmd=fake_claude(), screens=[FakeScreen()]) as s:
            event = web_output("A long enough page body with MARKER_SCREEN and MARKER_CLAUDE_YES in it.")
            # Act
            result = guard.on_output(event, guard_rules(s.url))
        # Assert
        note = result["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(("model:claude" in note, "screen" in note), (True, False))

    def test_classifier_band_goes_to_claude_below_old_threshold(self):
        # Arrange
        with Running(claude_cmd=fake_claude(), screens=[FakeScreen()], claude_above=0.5) as s:
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual(reply["scores"]["claude"], 1.0)

    def test_screen_down_is_reported_not_fatal(self):
        # Arrange
        with Running(claude_cmd=fake_claude(), screens=[FakeScreen(fail=True)]) as s:
            # Act
            reply = post(s.url, "an ordinary page")
        # Assert
        self.assertEqual(reply["scores"], {"fake": 0.1, "screen:screen-model_error": "ConnectionError"})

    def test_top_keeps_screens_apart(self):
        # Arrange
        scores = {"a": 0.3, "screen:b": 0.97, "c_error": "X"}
        # Act
        result = (server.top(scores), server.top(scores, screens=True))
        # Assert
        self.assertEqual(result, (0.3, 0.97))


class JudgeTests(unittest.TestCase):
    def test_judge_verdict_is_reported(self):
        # Arrange
        with FakeJudge(True) as j, Running(judge_url=j.url) as s:
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual(reply["scores"]["judge"], 1.0)

    def test_judge_skipped_below_threshold(self):
        # Arrange
        with FakeJudge(True) as j, Running(judge_url=j.url, judge_above=0.5) as s:
            # Act
            post(s.url, "an ordinary page")
        # Assert
        self.assertEqual(j.calls, 0)

    def test_judge_down_is_reported_not_fatal(self):
        # Arrange
        with Running(judge_url="http://127.0.0.1:9/v1") as s:
            # Act
            reply = post(s.url, "MARKER_HIGH in a page")
        # Assert
        self.assertEqual((reply["scores"]["fake"], "judge_error" in reply["scores"]), (0.95, True))


class GuardClientTests(unittest.TestCase):
    def test_high_score_from_outside_is_flagged(self):
        # Arrange
        with Running() as s:
            event = web_output("A long enough page body that mentions MARKER_HIGH somewhere in it.")
            # Act
            result = guard.on_output(event, guard_rules(s.url))
        # Assert
        self.assertIn("model:fake", result["hookSpecificOutput"]["additionalContext"])

    def test_low_score_adds_nothing(self):
        # Arrange
        with Running() as s:
            event = web_output("A long enough page body with nothing unusual in it at all.")
            # Act
            result = guard.on_output(event, guard_rules(s.url))
        # Assert
        self.assertIsNone(result)

    def test_local_command_output_is_not_sent(self):
        # Arrange
        with Running() as s:
            event = {"tool_name": "Bash", "tool_input": {"command": "cat notes.md"},
                     "tool_response": {"stdout": "Local notes long enough to be worth scoring, MARKER_HIGH."}}
            # Act
            guard.on_output(event, guard_rules(s.url))
        # Assert
        self.assertEqual(s.model.calls, 0)

    def test_server_down_falls_back_to_patterns(self):
        # Arrange
        event = web_output("A long enough page body that mentions MARKER_HIGH somewhere in it.")
        # Act
        result = guard.on_output(event, guard_rules("http://127.0.0.1:9"))
        # Assert
        self.assertIsNone(result)

    def test_client_sends_the_token(self):
        # Arrange
        with Running(token="t0ken-for-test") as s:
            event = web_output("A long enough page body that mentions MARKER_HIGH somewhere in it.")
            # Act
            result = guard.on_output(event, guard_rules(s.url, token="t0ken-for-test"))
        # Assert
        self.assertIn("model:fake", result["hookSpecificOutput"]["additionalContext"])

    def test_scalar_config_replaces_default(self):
        # Arrange
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".claude", "tool-guard"))
            with open(os.path.join(tmp, ".claude", "tool-guard", "settings.json"), "w") as f:
                json.dump({"classifier_url": "http://10.0.0.1:8765", "classifier_threshold": 0.8}, f)
            with mock.patch.dict(os.environ, {"CLAUDE_PROJECT_DIR": tmp}):
                # Act
                cfg = guard.load_config()
        # Assert
        self.assertEqual((cfg["classifier_url"], cfg["classifier_threshold"]), ("http://10.0.0.1:8765", 0.8))


if __name__ == "__main__":
    unittest.main()
