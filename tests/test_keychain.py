"""Fake values only; the macOS tests write to a throwaway keychain, never the login one."""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import guard  # noqa: E402
import keychain  # noqa: E402

KNOWN = "s3cret-Known-Value"
DUMP = '''keychain: "/k.keychain-db"
version: 512
class: "genp"
attributes:
    "acct"<blob>="u"
    "svce"<blob>="secrets/fi/jenkins-token"
keychain: "/k.keychain-db"
version: 512
class: "genp"
attributes:
    "svce"<blob>="git:https://example.com"
keychain: "/k.keychain-db"
version: 512
class: "inet"
attributes:
    "svce"<blob>="secrets/fi/not-generic"
'''


class ServicesTests(unittest.TestCase):
    def test_only_matching_generic_passwords_are_listed(self):
        # Arrange
        target_filter = "secrets/*"
        # Act
        names = keychain.services(DUMP, target_filter)
        # Assert
        self.assertEqual(names, ["secrets/fi/jenkins-token"])


@unittest.skipUnless(sys.platform == "darwin", "macOS keychain")
class SecretValuesTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.gettempdir(), f"guard-test-{os.getpid()}.keychain-db")
        subprocess.run([keychain.SECURITY, "create-keychain", "-p", "t", self.path], check=True)
        subprocess.run([keychain.SECURITY, "unlock-keychain", "-p", "t", self.path], check=True)

    def tearDown(self):
        subprocess.run([keychain.SECURITY, "delete-keychain", self.path], check=True)

    def add(self, service, value):
        command = f'add-generic-password -s {service} -a u -X {value.encode().hex()} "{self.path}"\n'
        subprocess.run([keychain.SECURITY, "-i"], input=command, encoding="utf-8", check=True)

    def test_values_matching_the_filter_are_read(self):
        # Arrange
        self.add("secrets/test/one", KNOWN)
        self.add("secrets/test/two", 'with "quotes" and \\ too')
        self.add("git:https://example.com", "not-a-workspace-secret")
        # Act
        values = keychain.secret_values("secrets/*", self.path)
        # Assert
        self.assertEqual(sorted(values), sorted([KNOWN, 'with "quotes" and \\ too']))

    @unittest.skipUnless(os.environ.get("SSH_CONNECTION"), "a GUI session asks to unlock instead of failing")
    def test_locked_keychain_raises_oserror(self):
        # Arrange
        self.add("secrets/test/one", KNOWN)
        subprocess.run([keychain.SECURITY, "lock-keychain", self.path], check=True)
        # Act / Assert
        with self.assertRaises(OSError):
            keychain.secret_values("secrets/*", self.path)

    def test_guard_redacts_keychain_values(self):
        # Arrange
        cfg = guard.default_config()
        cfg["credential_manager_filters"] = ["secrets/*"]
        with mock.patch.object(keychain, "secret_values", return_value=[KNOWN]):
            # Act
            known = guard.known_secrets(cfg)
        # Assert
        self.assertIn(KNOWN, known)


if __name__ == "__main__":
    unittest.main()
