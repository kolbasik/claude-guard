"""bench/score.py: combining saved scores, and the Ollaya adapter against a fake server."""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import judge  # noqa: E402
import score  # noqa: E402


def write_scores(folder, name, rows):
    with open(os.path.join(folder, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"label": l, "source": s, "score": p}) + "\n" for p, l, s in rows)


def combined(rows_by_name, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows in rows_by_name.items():
            write_scores(tmp, name, rows)
        out = io.StringIO()
        with mock.patch.object(score, "SCORES", tmp), contextlib.redirect_stdout(out):
            score.combine([], kw.get("th", 0.9), kw.get("screen", 0.5))
    return out.getvalue()


def line(output, title):
    return next(l for l in output.splitlines() if l.strip().startswith(title))


class CombineTests(unittest.TestCase):
    # Two injected and two benign samples; a is sure of one attack and unsure of the other.
    A = [(0.95, 1, "set"), (0.6, 1, "set"), (0.1, 0, "set"), (0.7, 0, "set")]
    B = [(0.2, 1, "set"), (0.92, 1, "set"), (0.1, 0, "set"), (0.1, 0, "set")]

    def test_any_model_catches_what_either_catches(self):
        # Arrange
        runs = {"a": self.A, "b": self.B}
        # Act
        output = combined(runs)
        # Assert
        self.assertIn("set 100.0%", line(output, "any model").split("|")[0])

    def test_cascade_confirms_unsure_screen(self):
        # Arrange
        runs = {"a": self.A, "b": self.B}
        # Act
        output = combined(runs)
        # Assert
        self.assertEqual(line(output, "cascade a -> b").split("caught: ")[1], "set 100.0% | false alarms: set   0.0%")

    def test_partial_second_model_serves_cascade(self):
        # Arrange: b was scored only on the sample a was unsure of (a = 0.6)
        partial = [(None, 1, "set"), (0.92, 1, "set"), (None, 0, "set"), (0.1, 0, "set")]
        runs = {"a": self.A, "b": partial}
        # Act
        output = combined(runs)
        # Assert
        self.assertEqual(line(output, "cascade a -> b").split("caught: ")[1], "set 100.0% | false alarms: set   0.0%")

    def test_partial_model_too_thin_for_cascade_is_skipped(self):
        # Arrange: b skipped the sample a was unsure of
        partial = [(None, 1, "set"), (None, 1, "set"), (None, 0, "set"), (0.1, 0, "set")]
        runs = {"a": self.A, "b": partial}
        # Act
        output = combined(runs)
        # Assert
        self.assertIn("skipped", line(output, "cascade a -> b"))

    def test_different_sample_sets_are_refused(self):
        # Arrange
        runs = {"a": self.A, "b": self.B[:3]}
        # Act
        with self.assertRaises(SystemExit) as err:
            combined(runs)
        # Assert
        self.assertIn("different sample sets", str(err.exception))


class FakeOllaya:
    def __init__(self, answers, max_chars=None):
        self.requests, self.paths = [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                fake.paths.append(self.path)
                fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                state = fake.requests[-1]["state"]
                prompt = state["prompt"] if isinstance(state, dict) else state   # qwen3guard gets the bare text
                too_long = max_chars is not None and len(prompt) > max_chars
                body = json.dumps({"error": "prompt 0 has 4982 tokens; expected 1-2050"} if too_long else
                                  {"answers": answers(prompt) if callable(answers) else answers}).encode()
                self.send_response(400 if too_long else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class OllayaTests(unittest.TestCase):
    def test_decision_model_scores_likelier_question(self):
        # Arrange
        answers = {"prompt_injection": {"noul": 0.2}, "jailbreak": {"noul": 0.7}}
        with FakeOllaya(answers) as fake:
            model = score.Ollaya("decider:4b", fake.url)
            # Act
            result = model.score("a ticket")
        # Assert
        self.assertEqual((result, fake.requests[-1]["questions"].keys()), (0.7, score.GUARD_QUESTIONS.keys()))

    def test_qwen3guard_counts_only_jailbreak_category(self):
        # Arrange
        answers = {"unsafe": {"noul": 0.8}, "category": {"choice": "violent"}}
        with FakeOllaya(answers) as fake:
            model = score.Ollaya("qwen3guard", fake.url)
            # Act
            result = model.score("a ticket")
        # Assert
        self.assertEqual((result, "questions" in fake.requests[-1]), (0.0, False))

    def test_ollama_model_asks_systemone_endpoint(self):
        # Arrange
        answers = {"prompt_injection": {"noul": 0.9}, "jailbreak": {"noul": 0.1}}
        with FakeOllaya(answers) as fake:
            model = score.Ollama("tev1:4b", fake.url)
            # Act
            result = model.score("a ticket")
        # Assert
        self.assertEqual((result, fake.paths[-1], model.name), (0.9, "/v1/systemone", "ollama-tev1-4b"))

    def test_ollama_text_over_token_limit_scores_its_likelier_half(self):
        # Arrange
        def answers(prompt):
            return {"prompt_injection": {"noul": 0.8 if "attack" in prompt else 0.1}, "jailbreak": {"noul": 0.0}}
        with FakeOllaya(answers, max_chars=20) as fake:
            model = score.Ollama("tev1:4b", fake.url)
            # Act
            result = model.score("plain text here " + "attack")
        # Assert
        self.assertEqual(result, 0.8)


class JudgeVariantTests(unittest.TestCase):
    def test_every_variant_pins_a_full_model_id(self):
        # Arrange
        variants = judge.VARIANTS
        # Act
        aliases = [name for name, (model, _, _) in variants.items() if not model.startswith("claude-")]
        # Assert
        self.assertEqual(aliases, [])


if __name__ == "__main__":
    unittest.main()
