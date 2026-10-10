"""Local ONNX prompt-injection classifier: text in, probability of the attack label out.

A model directory holds one *.onnx file, tokenizer.json and config.json (id2label).
Long text is scored in overlapping windows and the highest window wins, so an
injection buried deep in a document is not averaged away.

Needs onnxruntime, tokenizers and numpy; guard.py itself does not.
"""
import glob
import json
import os
import re

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

POSITIVE = re.compile(r"inject|jailbreak|malicious|unsafe|attack", re.I)


class Classifier:
    def __init__(self, model_dir, window=None, max_windows=32):
        onnx = glob.glob(os.path.join(model_dir, "*.onnx"))
        if len(onnx) != 1:
            raise ValueError(f"{model_dir} needs exactly one .onnx file, found {len(onnx)}")
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        labels = {int(k): v for k, v in cfg["id2label"].items()}
        self.name = os.path.basename(os.path.normpath(model_dir))
        self.positive = next((i for i, v in labels.items() if POSITIVE.search(v)), max(labels))
        self.window = window or min(cfg.get("max_position_embeddings", 512), 1024)
        # ponytail: long text is cut at max_windows windows (~24k tokens at 1024); raise it if tickets get longer.
        self.max_windows = max_windows
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(self.window, stride=self.window // 4)
        self.session = ort.InferenceSession(onnx[0], providers=["CPUExecutionProvider"])

    def score(self, text):
        """Highest attack probability over the text's windows, 0..1."""
        first = self.tokenizer.encode(text)
        best = 0.0
        for enc in ([first] + list(first.overflowing))[: self.max_windows]:
            feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                    "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
            logits = self.session.run(None, feed)[0][0]
            exp = np.exp(logits - logits.max())
            best = max(best, float(exp[self.positive] / exp.sum()))
        return best
