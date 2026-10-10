"""Read Windows Credential Manager values in-process, for matching only.

The values stay in this process's memory: never print, log or write them.
"""
import ctypes
from ctypes import wintypes

CRED_TYPE_GENERIC = 1
ERROR_NOT_FOUND = 1168


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


class _CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", _FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def secret_values(target_filter):
    """Generic-credential values whose target name matches target_filter (e.g. 'myapp/*')."""
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi.CredEnumerateW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(_CREDENTIAL))),
    ]
    advapi.CredEnumerateW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = [ctypes.c_void_p]

    count = wintypes.DWORD()
    creds = ctypes.POINTER(ctypes.POINTER(_CREDENTIAL))()
    if not advapi.CredEnumerateW(target_filter, 0, ctypes.byref(count), ctypes.byref(creds)):
        if ctypes.get_last_error() == ERROR_NOT_FOUND:
            return []
        raise OSError(f"CredEnumerate failed (win32 {ctypes.get_last_error()})")
    try:
        values = []
        for i in range(count.value):
            cred = creds[i].contents
            if cred.Type == CRED_TYPE_GENERIC and cred.CredentialBlobSize:
                blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
                values.append(blob.decode("utf-16-le"))
        return values
    finally:
        advapi.CredFree(creds)
