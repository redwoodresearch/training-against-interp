import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

from fastmcp import FastMCP

from src.docent_utils import AgentTranscript
from src.model_organisms import SystemPromptModel
from src.utils import load_prompt_file

from .claude_agent import ClaudeAgent
from .tools import create_auditing_mcp

PROMPT_DIR = "src/auditor/prompts/claude_agent"


def _bind_free_socket() -> socket.socket:
    """Bind a socket to a free port and return it (kept open to prevent races)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", 0))
    sock.listen(1)
    return sock


@asynccontextmanager
async def launch_mcp_servers(servers: List[FastMCP]):
    """Launch FastMCP servers as HTTP and yield mcp_servers config for ClaudeAgentOptions."""
    tasks = []
    configs = {}
    sockets = []
    for server in servers:
        sock = _bind_free_socket()
        sockets.append(sock)
        port = sock.getsockname()[1]
        task = asyncio.create_task(
            server.run_http_async(
                transport="http",
                host="127.0.0.1",
                port=port,
                log_level="error",
                show_banner=False,
                uvicorn_config={"fd": sock.fileno()},
            )
        )
        tasks.append(task)
        configs[server.name] = {"type": "http", "url": f"http://localhost:{port}/mcp"}

    await asyncio.sleep(1)
    try:
        yield configs
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()


@dataclass
class RunResult:
    findings: str
    transcript: AgentTranscript
    metadata: dict = field(default_factory=dict)


async def run_agent(
    target_model: SystemPromptModel,
    max_tokens: int = 200_000,
    agent_model: str = "claude-haiku-4-5-20251001",
    include_prefill_tools: bool = False,
    verbose: bool = False,
) -> RunResult:
    """Run an investigator agent against a target model and return findings."""
    system_prompt = load_prompt_file(f"{PROMPT_DIR}/system_prompt_message.jinja2")
    mcp = create_auditing_mcp(target_model, include_prefill_tools=include_prefill_tools)

    async with launch_mcp_servers([mcp]) as mcp_config:
        agent = ClaudeAgent(
            model=agent_model,
            system_prompt=system_prompt,
            mcp_servers=mcp_config,
            verbose=verbose,
        )

        async with agent:
            # Investigation loop (token-based)
            remaining = max_tokens
            stall_count = 0

            while remaining > 0 and stall_count < 3:
                user_prompt = load_prompt_file(
                    f"{PROMPT_DIR}/user_message.jinja2",
                    remaining_tokens=remaining,
                )
                tokens_before = agent.total_generated_tokens
                await agent.query(user_prompt, max_token_length=remaining)
                used = agent.total_generated_tokens - tokens_before
                remaining -= used
                stall_count = stall_count + 1 if used == 0 else 0

            # Articulation phase
            articulation = load_prompt_file(f"{PROMPT_DIR}/articulation_prompt.jinja2")
            await agent.query(articulation)

        transcript = agent.get_transcript()
        findings = _extract_last_assistant_text(transcript)

    return RunResult(
        findings=findings,
        transcript=transcript,
        metadata={
            "agent_model": agent_model,
            "max_tokens": max_tokens,
            "total_generated_tokens": agent.total_generated_tokens,
            "total_runtime_seconds": agent.total_run_time,
        },
    )


def _extract_last_assistant_text(transcript: AgentTranscript) -> str:
    for msg in reversed(transcript.messages):
        if msg.role == "assistant" and msg.content:
            return msg.content
    return ""
