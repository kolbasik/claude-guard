"""Download chosen files of a Hugging Face model at a pinned revision, verifying LFS hashes.

  python fetch_model.py <repo> <dest-dir> <file> [<file> ...] [--revision <sha>]

Files land flat in dest-dir; REVISION records repo@sha. Stdlib only, so it runs
before any dependency is installed.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--revision", help="commit sha; default is the current main")
    args = ap.parse_args()

    api = f"https://huggingface.co/api/models/{args.repo}" + (f"/revision/{args.revision}" if args.revision else "")
    with urllib.request.urlopen(api + "?blobs=true", timeout=30) as r:
        info = json.load(r)
    if info.get("gated"):
        sys.exit(f"{args.repo} is gated; fetch_model.py only downloads open models")
    rev = info["sha"]
    siblings = {s["rfilename"]: s for s in info["siblings"]}
    os.makedirs(args.dest, exist_ok=True)
    for name in args.files:
        if name not in siblings:
            sys.exit(f"{name} is not in {args.repo}@{rev[:12]}")
        out = os.path.join(args.dest, os.path.basename(name))
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"https://huggingface.co/{args.repo}/resolve/{rev}/{name}", timeout=60) as r, \
                open(out, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
        expected = (siblings[name].get("lfs") or {}).get("sha256")
        if expected and expected != digest.hexdigest():
            os.remove(out)
            sys.exit(f"{name}: sha256 mismatch, file removed")
        print(f"{name}: {os.path.getsize(out) / 1e6:.1f} MB{' sha256 ok' if expected else ''}")
    with open(os.path.join(args.dest, "REVISION"), "w") as f:
        f.write(f"{args.repo}@{rev}\n")
    print(f"{args.repo}@{rev}")


if __name__ == "__main__":
    main()
