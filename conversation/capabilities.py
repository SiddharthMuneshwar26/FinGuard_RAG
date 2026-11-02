"""Configuration-driven capability flags per client channel (see config/channels.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Capabilities:
    channel: str
    trusted: bool = False
    pii_policy: str = "block"  # "block" | "mask"
    max_input_chars: int = 1000
    streaming: bool = True
    generation: bool = True
    history_turns: int = 2
    retrieval_cache: bool = True

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class CapabilityRegistry:
    """Resolves the X-Channel header to a capability profile.

    Unknown or missing channels get the default (least-privileged) profile, so a
    client cannot unlock behaviour by inventing a channel name.
    """

    def __init__(self, profiles: dict[str, Capabilities], default_channel: str) -> None:
        if default_channel not in profiles:
            raise ValueError(f"default_channel {default_channel!r} is not defined")
        self._profiles = profiles
        self.default_channel = default_channel

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityRegistry":
        profiles: dict[str, Capabilities] = {}
        for name, flags in (data.get("channels") or {}).items():
            flags = dict(flags or {})
            policy = str(flags.get("pii_policy", "block"))
            if policy not in {"block", "mask"}:
                raise ValueError(f"channel {name!r}: pii_policy must be 'block' or 'mask'")
            profiles[name] = Capabilities(channel=name, **flags)
        return cls(profiles, str(data.get("default_channel", "web")))

    @classmethod
    def from_file(cls, path: Path) -> "CapabilityRegistry":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def resolve(self, channel: str | None) -> Capabilities:
        key = (channel or "").strip().lower()
        return self._profiles.get(key) or self._profiles[self.default_channel]

    @property
    def channels(self) -> list[str]:
        return list(self._profiles)
