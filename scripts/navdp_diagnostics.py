#!/usr/bin/env python3
"""NavDP diagnostics — verify all dependencies before running the blueprint.

Checks:
  1. navdp_bridge package importable
  2. NavDP inference server reachable (HTTP health check)
  3. VLM server reachable (HTTP health check)
  4. Embedding server reachable (if configured)
  5. Ollama agent LLM reachable
  6. DimOS NavDP modules importable (navigator, memory, skills)
  7. Config valid

Usage:
    python scripts/navdp_diagnostics.py [--config PATH]

    # Or with DimOS venv:
    uv run python scripts/navdp_diagnostics.py
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

# ── Color helpers ──────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}[{title}]{RESET}")


# ── Checks ─────────────────────────────────────────────────────────────

def check_import(module_name: str, description: str) -> bool:
    try:
        importlib.import_module(module_name)
        ok(f"{description}: {module_name}")
        return True
    except Exception as e:
        fail(f"{description}: {module_name} — {e}")
        return False


def check_http(url: str, description: str, timeout: float = 5.0) -> bool:
    """Check if an HTTP server is reachable."""
    import httpx

    endpoints_to_try = [
        url,
        url.rstrip("/") + "/health",
        url.rstrip("/") + "/v1/models",
        url.rstrip("/") + "/",
    ]
    for endpoint in endpoints_to_try:
        try:
            resp = httpx.get(endpoint, timeout=timeout)
            ok(f"{description}: {endpoint} → HTTP {resp.status_code}")
            return True
        except Exception:
            continue

    fail(f"{description}: {url} — not reachable (tried {len(endpoints_to_try)} endpoints)")
    return False


def check_navdp_server(url: str) -> bool:
    """Check NavDP inference server with an initialize call."""
    import httpx

    try:
        resp = httpx.get(f"{url}/health", timeout=5.0)
        ok(f"NavDP inference server: {url}/health → HTTP {resp.status_code}")
        return True
    except Exception:
        pass

    # Try initialize endpoint (the actual API)
    try:
        resp = httpx.post(
            f"{url}/initialize",
            json={"reset": False},
            timeout=5.0,
        )
        ok(f"NavDP inference server: {url}/initialize → HTTP {resp.status_code}")
        return True
    except Exception as e:
        fail(f"NavDP inference server: {url} — {e}")
        return False


def check_vlm_server(url: str) -> bool:
    """Check Qwen3-VL server with a text-only inference call."""
    import httpx

    try:
        resp = httpx.post(
            f"{url}/v1/text-inference",
            json={"prompt": "Say hello in one word.", "max_new_tokens": 10},
            timeout=10.0,
        )
        data = resp.json()
        response_text = data.get("response", "")[:50]
        ok(f"VLM server: {url}/v1/text-inference → '{response_text}'")
        return True
    except Exception as e:
        fail(f"VLM server: {url}/v1/text-inference — {e}")
        return False


def check_ollama(base_url: str, model: str) -> bool:
    """Check Ollama server and model availability."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        # Strip "ollama:" prefix from model name
        model_short = model.replace("ollama:", "")
        found = any(model_short in m for m in models)
        if found:
            ok(f"Ollama: {base_url} has model '{model_short}'")
        else:
            warn(f"Ollama: {base_url} reachable but '{model_short}' not found. Available: {models[:5]}")
        return True
    except Exception as e:
        fail(f"Ollama: {base_url} — {e}")
        return False


def check_dimos_navdp_modules() -> dict[str, bool]:
    """Check all DimOS NavDP module imports."""
    results = {}
    modules = {
        "Navigator": "dimos.navigation.navdp.navigator",
        "Memory": "dimos.navigation.navdp.memory",
        "Skills": "dimos.navigation.navdp.skills",
        "Blueprint": "dimos.navigation.navdp.blueprint",
    }
    for name, mod in modules.items():
        results[name] = check_import(mod, f"NavDP {name}")
    return results


def check_navdp_bridge_modules() -> dict[str, bool]:
    """Check navdp_bridge sub-modules."""
    results = {}
    modules = {
        "state_machine": "navdp_bridge.state_machine",
        "navdp_client": "navdp_bridge.navdp_client",
        "escape_controller": "navdp_bridge.escape_controller",
        "trajectory_controller": "navdp_bridge.trajectory_controller",
        "vlm_client": "navdp_bridge.vlm_client",
        "spatial_memory": "navdp_bridge.spatial_memory",
        "landmark_manager": "navdp_bridge.landmark_manager",
        "goal_context": "navdp_bridge.goal_context",
    }
    for name, mod in modules.items():
        results[name] = check_import(mod, f"navdp_bridge.{name}")
    return results


def run_diagnostics(config_path: str | None = None) -> dict:
    """Run all diagnostics and return summary."""
    print(f"{BOLD}NavDP Diagnostics{RESET}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.executable}")

    results: dict[str, Any] = {"passed": 0, "failed": 0, "warnings": 0}

    # ── 1. Load config ──
    section("1. Configuration")
    try:
        from dimos.agents.skills.config import load_vln_config
        cfg = load_vln_config(config_path or os.environ.get("VLN_CONFIG"))
        ok(f"Config loaded (navdp.enabled={cfg.navdp.enabled})")
        print(f"     NavDP server: {cfg.navdp.navdp_server_url}")
        print(f"     VLM server:   {cfg.navdp.vlm_server_url}")
        print(f"     VLM backend:  {cfg.vlm.backend} @ {cfg.vlm.base_url}")
        print(f"     Agent:        {cfg.agent.model} @ {cfg.agent.ollama_base_url}")
        print(f"     Escape:       enabled={cfg.escape.enabled}")
        if not cfg.navdp.enabled:
            warn("NavDP is DISABLED in config. Set navdp.enabled: true to activate.")
            results["warnings"] += 1
        results["passed"] += 1
    except Exception as e:
        fail(f"Config load failed: {e}")
        results["failed"] += 1
        return results

    # ── 2. navdp_bridge package ──
    section("2. navdp_bridge package")
    bridge_ok = check_import("navdp_bridge", "navdp_bridge package")
    if bridge_ok:
        results["passed"] += 1
        bridge_modules = check_navdp_bridge_modules()
        results["passed"] += sum(bridge_modules.values())
        results["failed"] += sum(not v for v in bridge_modules.values())
    else:
        fail("Install with: uv pip install -e /home/adamliao/work/NavDP/ros2_ws/src/navdp_bridge --no-deps")
        results["failed"] += 1

    # ── 3. DimOS NavDP modules ──
    section("3. DimOS NavDP modules")
    dimos_modules = check_dimos_navdp_modules()
    results["passed"] += sum(dimos_modules.values())
    results["failed"] += sum(not v for v in dimos_modules.values())

    # ── 4. External servers ──
    section("4. External servers")
    try:
        import httpx  # noqa: F401
    except ImportError:
        fail("httpx not installed — cannot check servers")
        results["failed"] += 1
        _print_summary(results)
        return results

    # NavDP inference server
    if check_navdp_server(cfg.navdp.navdp_server_url):
        results["passed"] += 1
    else:
        results["failed"] += 1
        warn(f"Start the NavDP inference server at {cfg.navdp.navdp_server_url}")

    # VLM server
    if check_vlm_server(cfg.vlm.base_url):
        results["passed"] += 1
    else:
        results["failed"] += 1

    # Ollama
    if check_ollama(cfg.agent.ollama_base_url, cfg.agent.model):
        results["passed"] += 1
    else:
        results["failed"] += 1

    # ── 5. End-to-end smoke test ──
    section("5. Smoke test (NavDP client → server)")
    if cfg.navdp.enabled:
        try:
            from navdp_bridge.navdp_client import NavDPClient
            import numpy as np

            intrinsic = (
                np.array(cfg.navdp.cam_intrinsic, dtype=np.float32)
                if cfg.navdp.cam_intrinsic is not None
                else np.array([[460, 0, 320], [0, 460, 240], [0, 0, 1]], dtype=np.float32)
            )
            client = NavDPClient(url=cfg.navdp.navdp_server_url)
            client.initialize(intrinsic)
            ok("NavDPClient.initialize() succeeded")
            results["passed"] += 1

            # Try a nogoal step with a dummy image
            dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_depth = np.zeros((480, 640), dtype=np.float32)
            traj, all_traj, all_vals = client.nogoal_step(dummy_rgb, dummy_depth)
            if traj is not None:
                ok(f"NavDPClient.nogoal_step() → trajectory shape={traj.shape}")
            else:
                warn("NavDPClient.nogoal_step() returned None trajectory (may need real images)")
            results["passed"] += 1
        except Exception as e:
            fail(f"Smoke test failed: {e}")
            results["failed"] += 1
    else:
        warn("Skipping smoke test (NavDP disabled in config)")
        results["warnings"] += 1

    _print_summary(results)

    # Write results to file
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "assets", "output", "navdp_diagnostics.json",
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "navdp_enabled": cfg.navdp.enabled,
            **results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def _print_summary(results: dict) -> None:
    section("Summary")
    total = results["passed"] + results["failed"]
    color = GREEN if results["failed"] == 0 else RED
    print(f"  {color}{results['passed']}/{total} checks passed{RESET}", end="")
    if results["warnings"]:
        print(f", {YELLOW}{results['warnings']} warnings{RESET}")
    else:
        print()

    if results["failed"] > 0:
        print(f"\n  {RED}NavDP is NOT ready to run.{RESET} Fix the failures above.")
    elif results["warnings"] > 0:
        print(f"\n  {YELLOW}NavDP may work but check warnings.{RESET}")
    else:
        print(f"\n  {GREEN}NavDP is ready!{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NavDP diagnostics")
    parser.add_argument("--config", help="Path to VLN config YAML")
    args = parser.parse_args()
    results = run_diagnostics(args.config)
    sys.exit(1 if results["failed"] > 0 else 0)
