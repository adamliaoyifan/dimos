# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLN configuration loader.

Reads a YAML file and returns typed dataclasses that can be passed
directly to blueprint factories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG = Path(__file__).parent / "vln_config.yaml"


@dataclass
class VLMConfig:
    backend: str = "qwen3_local"
    base_url: str = "http://192.168.2.109:8000"
    model_name: str = "Qwen3-VL-8B-Instruct"


@dataclass
class AgentConfig:
    model: str = "ollama:qwen3:14b"
    ollama_base_url: str = "http://192.168.2.109:11434"


@dataclass
class SearchConfig:
    vlm_check_interval: float = 1.0
    search_timeout: float = 120.0
    approach_timeout: float = 30.0
    similarity_threshold: float = 0.23


@dataclass
class NavDPConfig:
    enabled: bool = False
    navdp_server_url: str = "http://192.168.2.109:8880"
    vlm_server_url: str = "http://192.168.2.109:8000"


@dataclass
class EscapeConfig:
    enabled: bool = True
    stuck_time_window: float = 6.0
    stuck_distance_threshold: float = 0.05
    escape_backup_distance: float = 0.3
    escape_rotate_degrees: float = 90.0
    max_escape_attempts: int = 4


@dataclass
class BlueprintConfig:
    enable_tts: bool = False
    n_workers: int = 8


@dataclass
class VLNTestConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    navdp: NavDPConfig = field(default_factory=NavDPConfig)
    escape: EscapeConfig = field(default_factory=EscapeConfig)
    blueprint: BlueprintConfig = field(default_factory=BlueprintConfig)


def load_vln_config(path: str | Path | None = None) -> VLNTestConfig:
    """Load VLN configuration from a YAML file.

    Args:
        path: Path to the YAML config file. If None, uses the default
              config shipped with DimOS.

    Returns:
        Parsed VLNTestConfig dataclass.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"VLN config not found: {config_path}")

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return VLNTestConfig(
        vlm=VLMConfig(**raw.get("vlm", {})),
        agent=AgentConfig(**raw.get("agent", {})),
        search=SearchConfig(**raw.get("search", {})),
        navdp=NavDPConfig(**raw.get("navdp", {})),
        escape=EscapeConfig(**raw.get("escape", {})),
        blueprint=BlueprintConfig(**raw.get("blueprint", {})),
    )


__all__ = [
    "VLNTestConfig",
    "VLMConfig",
    "AgentConfig",
    "SearchConfig",
    "NavDPConfig",
    "EscapeConfig",
    "BlueprintConfig",
    "load_vln_config",
]
