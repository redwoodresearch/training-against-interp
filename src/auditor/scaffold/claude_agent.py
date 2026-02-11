import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from docent.data_models.chat import ToolCall

from src.docent_utils import AgentTranscript

ERROR_PATTERNS = [
    ("request_too_large", "request_too_large"),
    ("413", "request_too_large"),
    ("invalid_request", "invalid_request_error"),
    ("400", "invalid_request_error"),
    ("rate_limit", "rate_limit_error"),
    ("429", "rate_limit_error"),
    ("overloaded", "overloaded_error"),
    ("529", "overloaded_error"),
    ("api_error", "api_error"),
    ("500", "api_error"),
]

FATAL_ERROR_TYPES = {"request_too_large", "invalid_request_error"}


class APIError(Exception):
    def __init__(self, error_code: str, error_message: str):
        self.error_code = error_code
        self.error_message = error_message
        super().__init__(f"API Error {error_code}: {error_message}")


class ClaudeAgent:
    """Wraps claude-agent-sdk to run an investigator agent with SDK MCP tools."""

    def __init__(
        self,
        model: str,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        permission_mode: str = "acceptEdits",
    ):
        self.transcript = AgentTranscript(verbose=verbose)
        self.system_prompt = system_prompt
        self.model = model
        self.permission_mode = permission_mode
        self.total_run_time = 0
        self.total_generated_tokens = 0

        self._mcp_servers = mcp_servers or {}
        self._client: Optional[ClaudeSDKClient] = None
        self._initialized = False

        if system_prompt:
            self.transcript.add_system_message(system_prompt)

    async def initialize(self) -> None:
        if self._initialized:
            return

        allowed_tools = [f"mcp__{server_name}__*" for server_name in self._mcp_servers]

        options_dict = {
            "permission_mode": self.permission_mode,
            "allowed_tools": allowed_tools,
            "disallowed_tools": [
                "Bash",
                "Read",
                "Write",
                "Edit",
                "MultiEdit",
                "Glob",
                "Grep",
                "LS",
                "NotebookEdit",
                "Task",
                "WebFetch",
                "WebSearch",
            ],
            "model": self.model,
            "system_prompt": self.system_prompt
            or {"type": "preset", "preset": "claude_code"},
        }
        if self._mcp_servers:
            options_dict["mcp_servers"] = self._mcp_servers

        self._client = ClaudeSDKClient(options=ClaudeAgentOptions(**options_dict))
        await self._client.connect()
        self._initialized = True

    async def cleanup(self) -> None:
        if self._client:
            await self._client.disconnect()
            self._client = None
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    async def query(
        self,
        prompt: str,
        max_run_length: Optional[float] = None,
        max_token_length: Optional[int] = None,
    ) -> None:
        if not self._initialized:
            await self.initialize()

        if max_run_length is not None and max_token_length is not None:
            raise ValueError("Cannot specify both max_run_length and max_token_length.")

        self.transcript.add_user_message(prompt)
        await self._client.query(prompt)

        tool_call_map: Dict[str, str] = {}
        start_time = asyncio.get_running_loop().time() if max_run_length else None
        tokens_at_start = self.total_generated_tokens
        query_wall_start = time.time()

        async for message in self._client.receive_response():
            self._process_message(message, tool_call_map)

            if max_run_length is not None and start_time is not None:
                if asyncio.get_running_loop().time() - start_time >= max_run_length:
                    await self._client.interrupt()
                    start_time = None

            if (
                max_token_length is not None
                and (self.total_generated_tokens - tokens_at_start) >= max_token_length
            ):
                await self._client.interrupt()
                break

        self.total_run_time += time.time() - query_wall_start

    def get_transcript(self) -> AgentTranscript:
        """Return the agent's transcript."""
        if self.total_run_time > 0 or self.total_generated_tokens > 0:
            if self.transcript.metadata is None:
                self.transcript.metadata = {}
            if self.total_run_time > 0:
                self.transcript.metadata["total_runtime_seconds"] = self.total_run_time
            if self.total_generated_tokens > 0:
                self.transcript.metadata["total_generated_tokens"] = (
                    self.total_generated_tokens
                )
        return self.transcript

    # -- Message processing --

    def _process_message(self, message: Any, tool_call_map: Dict[str, str]) -> None:
        if isinstance(message, UserMessage):
            self._handle_tool_results(message, tool_call_map)
        elif isinstance(message, AssistantMessage):
            self._handle_assistant(message, tool_call_map)
        elif isinstance(message, ResultMessage):
            self._handle_result(message)

    def _handle_tool_results(
        self, message: UserMessage, tool_call_map: Dict[str, str]
    ) -> None:
        if not isinstance(message.content, list):
            return
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            if isinstance(block.content, str):
                content = block.content
            elif block.content and "text" in block.content[0]:
                content = str(block.content[0]["text"])
            else:
                content = str(block.content)

            tool_call_id = block.tool_use_id
            self.transcript.add_tool_message(
                content=content,
                tool_call_id=tool_call_id,
                function=tool_call_map.get(tool_call_id) if tool_call_id else None,
                is_error=block.is_error or False,
            )

    def _handle_assistant(
        self, message: AssistantMessage, tool_call_map: Dict[str, str]
    ) -> None:
        if not isinstance(message.content, list):
            return

        texts, tools = [], []
        for block in message.content:
            if isinstance(block, TextBlock) and (text := block.text.strip()):
                texts.append(text)
            elif isinstance(block, ThinkingBlock) and (
                thinking := block.thinking.strip()
            ):
                texts.append(thinking)
            elif isinstance(block, ToolUseBlock):
                tool_call_map[block.id] = block.name
                tools.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=block.name,
                        arguments=(
                            block.input
                            if isinstance(block.input, dict)
                            else {"input": block.input}
                        ),
                    )
                )

        if texts or tools:
            self.transcript.add_assistant_message("\n\n".join(texts), tools or None)

    def _handle_result(self, message: ResultMessage) -> None:
        if message.is_error:
            error_msg = message.result or "Unknown error"
            error_type = _classify_error(error_msg)
            print(f"API Error ({error_type}): {error_msg[:200]}")
            if error_type in FATAL_ERROR_TYPES:
                raise APIError(error_type, error_msg)

        if message.usage:
            self.total_generated_tokens += message.usage.get("output_tokens", 0)


def _classify_error(error_message: str) -> str:
    try:
        data = json.loads(error_message)
        if isinstance(data, dict):
            if "error" in data and isinstance(data["error"], dict):
                return data["error"].get("type", "unknown")
            if "type" in data:
                return data.get("type", "unknown")
    except (json.JSONDecodeError, TypeError):
        pass

    lower = error_message.lower()
    for keyword, etype in ERROR_PATTERNS:
        if keyword in lower:
            return etype
    return "unknown"
