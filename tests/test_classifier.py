"""classifier.py's mapped model copy, with a fake ONNX Runtime session: no real model needed."""
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
DEPS = all(importlib.util.find_spec(m) for m in ("onnxruntime", "tokenizers", "numpy"))
if DEPS:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    import classifier  # noqa: E402


def fake_runtime(fail_save=False):
    """An InferenceSession stand-in that records each load and writes the model it is asked to save."""
    loads = []

    def session(path, so=None, providers=None):
        out = so.optimized_model_filepath if so else ""
        loads.append((path, so))
        if out and fail_save:
            raise RuntimeError("no space left on device")
        for p in (out, out + ".data") if out else ():
            open(p, "w").close()

    return session, loads


def model_dir(root):
    d = os.path.join(root, "model")
    os.makedirs(d)
    open(os.path.join(d, "model_quantized.onnx"), "w").close()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"id2label": {"0": "benign", "1": "injection"}, "max_position_embeddings": 512}, f)
    Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(os.path.join(d, "tokenizer.json"))
    return d


@unittest.skipUnless(DEPS, "needs onnxruntime, tokenizers and numpy")
class MappedCopyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = model_dir(tmp.name)
        self.src = os.path.join(self.dir, "model_quantized.onnx")

    def patch(self, fail_save=False):
        session, loads = fake_runtime(fail_save)
        patcher = mock.patch.object(classifier.ort, "InferenceSession", session)
        patcher.start()
        self.addCleanup(patcher.stop)
        return loads

    def test_mapped_copy_is_built_once(self):
        # Arrange
        loads = self.patch()
        first = classifier.mapped(self.src)
        # Act
        second = classifier.mapped(self.src)
        # Assert
        self.assertEqual((second, len(loads)), (first, 1))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "mapped", "model.onnx.data")))

    def test_mapped_copy_is_rebuilt_when_the_model_is_newer(self):
        # Arrange
        loads = self.patch()
        classifier.mapped(self.src)
        later = time.time() + 60
        os.utime(self.src, (later, later))
        # Act
        classifier.mapped(self.src)
        # Assert
        self.assertEqual(len(loads), 2)

    def test_failed_save_leaves_no_partial_copy(self):
        # Arrange
        self.patch(fail_save=True)
        # Act
        with self.assertRaises(RuntimeError):
            classifier.mapped(self.src)
        # Assert
        self.assertEqual(sorted(os.listdir(self.dir)), ["config.json", "model_quantized.onnx", "tokenizer.json"])

    def test_classifier_serves_the_mapped_copy_without_prepacking(self):
        # Arrange
        loads = self.patch()
        # Act
        classifier.Classifier(self.dir)
        # Assert
        path, so = loads[-1]
        self.assertEqual(path, os.path.join(self.dir, "mapped", "model.onnx"))
        self.assertEqual(so.get_session_config_entry("session.disable_prepacking"), "1")

    def test_classifier_serves_the_original_when_the_copy_fails(self):
        # Arrange
        loads = self.patch(fail_save=True)
        # Act
        with mock.patch("sys.stderr"):
            classifier.Classifier(self.dir)
        # Assert
        self.assertEqual(loads[-1], (self.src, None))


if __name__ == "__main__":
    unittest.main()
