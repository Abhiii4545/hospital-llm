"""Redaction for real patient documents.

Section 4.2 of the brief: **redaction happens BEFORE a document is stored, not
after.** A file that lands in `data/real/` unredacted and is cleaned up later has
already existed on disk in the clear, and probably in a backup.

Two properties matter and they pull against each other:

* **Geometry must survive.** Surrogates keep the original's length and character
  class, so line wrapping, column widths and where a field overflows are
  unchanged. Those are signal - a model trained on redacted pages whose text
  reflows differently is trained on a different layout distribution than the one
  it will see.
* **The mapping must be one-way.** Surrogates come from an HMAC keyed by a secret
  held OUTSIDE the repository. Without the key you cannot invert them; with the
  same key the same name always maps to the same surrogate, so a patient
  appearing on three pages stays one person.

The key is refused if it lives inside the repository, because a one-way mapping
whose key sits next to the data is not one-way.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from reckon.schema import RawDocument

__all__ = [
    "RedactionKey",
    "surrogate",
    "redact_value",
    "redact_document",
    "redact_text",
    "RedactionManifest",
    "PII_FIELDS",
    "DEFAULT_KEY_PATH",
]

#: Outside the repository, deliberately. See the module docstring.
DEFAULT_KEY_PATH = Path.home() / ".reckon" / "redaction.key"

#: Schema fields treated as identifying. `hospital.*` is NOT here: the hospital
#: is not the data subject, and its name is needed to slice results by provider.
PII_FIELDS: tuple[str, ...] = (
    "patient.name",
    "patient.uhid",
    "patient.ip_number",
    "insurance.policy_number",
    "insurance.claim_number",
    "insurance.employee_id",
)

#: Free-text patterns redacted anywhere they appear.
_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aadhaar", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("pan", re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])")),
    ("phone", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
)


class RedactionKey:
    """A secret held outside the repository."""

    def __init__(self, material: bytes, source: Path | None = None) -> None:
        if len(material) < 32:
            raise ValueError("redaction key must be at least 32 bytes")
        self._material = material
        self.source = source

    @staticmethod
    def load_or_create(path: Path | str = DEFAULT_KEY_PATH) -> "RedactionKey":
        path = Path(path).expanduser().resolve()
        repo = Path(__file__).resolve().parents[2]
        if repo in path.parents or path == repo:
            raise ValueError(
                f"refusing to use a redaction key inside the repository ({path}). "
                "A one-way mapping whose key sits beside the data is not one-way."
            )
        if path.exists():
            return RedactionKey(path.read_bytes(), path)

        path.parent.mkdir(parents=True, exist_ok=True)
        material = secrets.token_bytes(64)
        path.write_bytes(material)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass                                  # best effort on Windows
        return RedactionKey(material, path)

    def digest(self, kind: str, value: str) -> bytes:
        return hmac.new(
            self._material, f"{kind}:{value}".encode("utf-8"), hashlib.sha256
        ).digest()


def surrogate(value: str, key: RedactionKey, kind: str = "generic") -> str:
    """A replacement with the SAME length and character class as *value*.

    Digits map to digits, uppercase to uppercase, lowercase to lowercase, and
    everything else - spaces, slashes, punctuation - is left alone. The result is
    the same width on the page, so nothing reflows.

    Deterministic for a given key, so one patient stays one patient across pages,
    and one-way without it.
    """
    if not value:
        return value

    stream = key.digest(kind, value)
    # Extend the keystream if the value is longer than one digest.
    while len(stream) < len(value):
        stream += hmac.new(stream, b"extend", hashlib.sha256).digest()

    out: list[str] = []
    for character, byte in zip(value, stream):
        if character.isdigit():
            out.append(string.digits[byte % 10])
        elif character.isupper():
            out.append(string.ascii_uppercase[byte % 26])
        elif character.islower():
            out.append(string.ascii_lowercase[byte % 26])
        else:
            out.append(character)          # punctuation and spacing preserved
    return "".join(out)


def redact_value(value: str | None, key: RedactionKey, kind: str) -> str | None:
    return None if value is None else surrogate(value, key, kind)


def redact_text(text: str, key: RedactionKey) -> tuple[str, dict[str, int]]:
    """Redact free-text PII patterns. Returns the text and a per-pattern count."""
    counts: dict[str, int] = {}
    for name, pattern in _TEXT_PATTERNS:
        found = pattern.findall(text)
        if found:
            counts[name] = len(found)
            text = pattern.sub(lambda m: surrogate(m.group(0), key, name), text)
    return text, counts


@dataclass
class RedactionManifest:
    """What was redacted, without recording what it was redacted FROM.

    Deliberately stores only field paths and counts. A manifest listing the
    original values would recreate the problem it exists to document.
    """

    document_id: str
    redacted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    fields_redacted: list[str] = field(default_factory=list)
    text_patterns: dict[str, int] = field(default_factory=dict)
    key_fingerprint: str = ""
    consent_reference: str | None = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "redacted_at": self.redacted_at,
            "fields_redacted": self.fields_redacted,
            "text_patterns": self.text_patterns,
            "key_fingerprint": self.key_fingerprint,
            "consent_reference": self.consent_reference,
        }

    def append_to(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
        return path


def redact_document(
    document: RawDocument,
    key: RedactionKey,
    document_id: str,
    consent_reference: str | None = None,
    extra_fields: Iterable[str] = (),
) -> tuple[RawDocument, RedactionManifest]:
    """Return a redacted copy plus a manifest of what was replaced.

    The original is not mutated, so a caller cannot accidentally keep working
    with a half-redacted object.
    """
    redacted = document.model_copy(deep=True)
    manifest = RedactionManifest(
        document_id=document_id,
        consent_reference=consent_reference,
        key_fingerprint=hashlib.sha256(key.digest("fingerprint", "")).hexdigest()[:16],
    )

    for path in (*PII_FIELDS, *extra_fields):
        block, name = path.split(".", 1)
        target = getattr(redacted, block, None)
        if target is None:
            continue
        value = getattr(target, name, None)
        if value:
            setattr(target, name, surrogate(value, key, path))
            manifest.fields_redacted.append(path)

    # Line-item descriptions can carry a patient name ("Room Rent - Mr Kumar").
    for item in redacted.line_items:
        if item.description:
            cleaned, counts = redact_text(item.description, key)
            if counts:
                item.description = cleaned
                for pattern, n in counts.items():
                    manifest.text_patterns[pattern] = (
                        manifest.text_patterns.get(pattern, 0) + n
                    )

    return redacted, manifest
