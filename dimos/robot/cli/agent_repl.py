# Copyright 2025-2026 Dimensional Inc.
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

"""Interactive REPL for sending messages to the running agent via MCP."""

from __future__ import annotations

import requests
from rich.console import Console
from rich.panel import Panel

from dimos.agents.mcp.mcp_adapter import McpAdapter, McpError


def _print_help(console: Console) -> None:
    body = """[bold]Commands[/bold]
  /help     Show this help
  /exit     Exit the REPL
  /quit     Same as /exit

Any other line is sent to the agent as a single message (same as [cyan]dimos agent-send[/])."""
    console.print(Panel(body, title="dimos agent-repl", border_style="blue"))


def run_agent_repl() -> None:
    """Read lines and forward each non-command line as [agent_send](message=...)."""
    console = Console()
    adapter = McpAdapter.from_run_entry()

    console.print(
        "[bold]dimos agent-repl[/] — each line is sent via MCP [cyan]agent_send[/]. "
        "Commands: [cyan]/help[/], [cyan]/exit[/]."
    )

    while True:
        try:
            line = console.input("[bold]dimos>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exited.[/]")
            return

        if not line:
            continue
        lowered = line.lower()
        if lowered in ("/exit", "/quit"):
            return
        if lowered == "/help":
            _print_help(console)
            continue

        try:
            text = adapter.call_tool_text("agent_send", {"message": line})
        except requests.ConnectionError:
            console.print("[red]Error: no running MCP server (is DimOS running?)[/]")
            continue
        except McpError as e:
            console.print(f"[red]Error: {e}[/]")
            continue

        console.print(text)
