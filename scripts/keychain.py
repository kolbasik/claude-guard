"""Read macOS keychain generic-password values in-process, for matching only.

Goes through /usr/bin/security, the tool that wrote the items, so they trust it and no
access prompt appears. The values stay in this process's memory: never print, log or
write them.
"""
import fnmatch
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

SECURITY = "/usr/bin/security"
SERVICE = re.compile(r'^\s+"svce"<blob>="(.*)"$', re.M)


def services(dump, target_filter):
    """Generic-password service names in a `security dump-keychain` listing matching target_filter."""
    names = []
    for item in re.split(r"(?m)^keychain: ", dump)[1:]:
        m = SERVICE.search(item)
        if '\nclass: "genp"\n' in item and m and fnmatch.fnmatchcase(m.group(1), target_filter):
            names.append(m.group(1))
    return names


def _security(*args):
    try:
        r = subprocess.run([SECURITY, *args], capture_output=True, encoding="utf-8", timeout=3)
    except subprocess.SubprocessError as e:   # a hung access prompt
        raise OSError(f"security {args[0]}: {type(e).__name__}") from None
    if r.returncode:   # 36: locked, as in an ssh session
        raise OSError(f"security {args[0]} exited {r.returncode}")
    return r.stdout


def secret_values(target_filter, keychain=None):
    """Generic-password values whose service name matches target_filter (e.g. 'myapp/*')."""
    kc = [keychain] if keychain else []
    names = services(_security("dump-keychain", *kc), target_filter)
    # ponytail: one `security` per value (~35 ms); in parallel a dozen cost about one.
    with ThreadPoolExecutor(max_workers=16) as pool:
        return [v[:-1] if v.endswith("\n") else v
                for v in pool.map(lambda n: _security("find-generic-password", "-s", n, "-w", *kc), names)]
