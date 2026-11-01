"""Restrict Fara's native computer-use output to Jev's writing and review roles."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import openai


class FaraWriterError(ValueError):
    """Fara returned an unsupported or malformed computer-use response."""


TOOL_SCHEMA = {
    "name": "computer_use",
    "description": "Return a field-text proposal or finish a screen review. Do not operate the computer.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["type", "terminate"]},
            "text": {"type": "string", "description": "Text proposed for the focused field; only for action=type."},
            "answer": {"type": "string", "description": "Review result; only for action=terminate."},
            "achieved": {"type": "boolean", "description": "Whether the screenshot proves the whole goal; for reviews."},
            "focus": {"type": "string", "description": "One next UI step if incomplete; for reviews."},
            "question": {"type": "string", "description": "One question only the user can answer; for reviews."},
        },
        "required": ["action"],
    },
}

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_REVIEW_FIELDS = {"achieved": bool, "answer": str, "focus": str, "question": str}


class FaraWriter:
    """OpenAI-compatible Ollama client, adapted to Jev's Anthropic-shaped writer interface.

    Fara's action output is treated as data only. `type` supplies field text and `terminate`
    supplies a review. All other action names are rejected before Jev receives a result.
    """

    def __init__(self, base_url: str, api_key: str):
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = self._client.base_url
        self.messages = SimpleNamespace(create=self._create)

    def _create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        output_config=None,
        thinking=None,
        reasoning=None,
    ):
        mode = _mode(output_config)
        policy = _policy(mode)
        task_system = system if mode == "field" else policy
        tool_schema = json.loads(json.dumps(TOOL_SCHEMA))
        if mode == "review":
            tool_schema["parameters"]["properties"]["action"]["enum"] = ["terminate"]
            tool_schema["parameters"]["required"] = ["action", "answer", "achieved", "focus", "question"]
            examples = (
                "For a screen review, return action=terminate with answer, achieved, focus, and question as separate arguments. "
                'Example: <tool_call>{"name":"computer_use","arguments":{"action":"terminate","answer":"The requested value is visible.","achieved":true,"focus":"","question":""}}</tool_call>.'
            )
        else:
            examples = (
                'For example: <tool_call>{"name":"computer_use","arguments":{"action":"type","text":"replace with the exact field text"}}</tool_call>. '
                'To decline, use: <tool_call>{"name":"computer_use","arguments":{"action":"terminate","answer":""}}</tool_call>.'
            )
        fara_system = (
            "You are Fara, a screenshot-reading computer-use model. In this integration you never control the UI. "
            "You may only return a computer_use action of type or terminate. Never click, press a key, scroll, "
            "navigate, search, submit, or perform any other action. Jev alone executes desktop actions.\n\n"
            f"{task_system}\n\n"
            "Use the Fara tool-call format exactly:\n"
            "<tools>\n"
            f"{json.dumps(tool_schema, separators=(',', ':'))}\n"
            "</tools>\n"
            "For one function call, return a JSON object inside <tool_call> tags, with name=computer_use and an arguments object. "
            f"{examples}"
        )
        chat = [{"role": "system", "content": fara_system}, *(_chat_message(m, mode) for m in messages)]
        try:
            reply = self._client.chat.completions.create(
                model=model,
                messages=chat,
                max_tokens=max(1024, max_tokens),
                temperature=0,
                extra_body={"think": False},
            )
        except openai.APIError:
            raise
        message = reply.choices[0].message if reply.choices else None
        text = getattr(message, "content", None) if message else None
        normalized = normalize_tool_reply(text or "", mode)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(normalized, ensure_ascii=False))],
            usage=_usage(getattr(reply, "usage", None)),
        )


def _mode(output_config) -> str:
    schema = (output_config or {}).get("format", {}).get("schema", {})
    properties = schema.get("properties", {})
    names = set(properties)
    if "fill" in names and "text" in names:
        return "field"
    if names == set(_REVIEW_FIELDS):
        return "review"
    raise FaraWriterError(f"Fara is not enabled for this writer task ({', '.join(sorted(names))}).")


def _policy(mode: str) -> str:
    if mode == "field":
        return (
            "The user packet describes one focused field and the requested goal. Return action=type with the exact "
            "text that should go into that field. Do not include a plan or perform the action. If the field should "
            "not be filled or the necessary value is missing, return action=terminate with an empty answer."
        )
    return (
        "Review the supplied screenshot and screen text against every part of the user's goal. Base the result only on "
        "the current screenshot, screen text, prior actions, and information the user supplied; do not guess. Say in at "
        "most four plain sentences what the screen shows. Set achieved=true only if the screenshot visibly proves the "
        "whole goal. If it is incomplete and one next UI step could help, provide one short imperative focus and leave "
        "question empty. If only the user can supply missing information, provide one short question and leave focus "
        "empty. Never set both. Otherwise leave both empty. Return action=terminate only, with answer, achieved, focus, "
        "and question as separate arguments."
    )


def normalize_tool_reply(text: str, mode: str) -> dict:
    """Parse exactly one allowed native tool call and translate it to Jev's structured fields."""
    calls = _TOOL_CALL.findall(text)
    if len(calls) != 1:
        raise FaraWriterError("Fara must return exactly one <tool_call> block.")
    try:
        call = json.loads(calls[0])
    except json.JSONDecodeError as exc:
        raise FaraWriterError("Fara returned malformed tool-call JSON.") from exc
    if not isinstance(call, dict) or set(call) != {"name", "arguments"} or call.get("name") != "computer_use":
        raise FaraWriterError("Fara returned an unsupported tool call.")
    args = call.get("arguments")
    if not isinstance(args, dict) or not isinstance(args.get("action"), str):
        raise FaraWriterError("Fara returned malformed tool arguments.")

    action = args["action"]
    if mode == "field":
        if action == "type" and set(args) == {"action", "text"} and isinstance(args["text"], str):
            return {"fill": bool(args["text"]), "text": args["text"], "reason": "Fara proposed field text.", "submit": False}
        if action == "terminate" and set(args) == {"action", "answer"} and isinstance(args["answer"], str):
            return {"fill": False, "text": "", "reason": args["answer"], "submit": False}
        raise FaraWriterError(f"Fara returned an unsupported field action: {action!r}.")

    if mode == "review":
        if action != "terminate":
            raise FaraWriterError(f"Fara returned an unsupported review action: {action!r}.")
        if set(args) != {"action", *_REVIEW_FIELDS}:
            raise FaraWriterError("Fara's final review has the wrong fields.")
        review = {name: args[name] for name in _REVIEW_FIELDS}
        _check_review(review)
        return review
    raise FaraWriterError(f"Unknown Fara adapter mode: {mode!r}.")


def _check_review(review: dict) -> None:
    if not isinstance(review, dict) or set(review) != set(_REVIEW_FIELDS):
        raise FaraWriterError("Fara's final review has the wrong fields.")
    if any(type(review[name]) is not kind for name, kind in _REVIEW_FIELDS.items()):
        raise FaraWriterError("Fara's final review contains a field with the wrong type.")


def _chat_message(message: dict, mode: str) -> dict:
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    blocks = list(content)
    if mode == "review":
        blocks = [_compact_review_context(block) if block["type"] == "text" else block for block in blocks]
        blocks.sort(key=lambda block: 0 if block["type"] == "text" else 1)
    parts = []
    for block in blocks:
        if block["type"] == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image":
            source = block["source"]
            parts.append({"type": "image_url", "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"}})
        else:
            raise ValueError(f"Fara cannot receive a {block['type']!r} block.")
    return {"role": message["role"], "content": parts}


def _compact_review_context(block: dict) -> dict:
    """Keep the review's core user/screen evidence in the compact packet Fara follows reliably."""
    try:
        packet = json.loads(block["text"])
    except (KeyError, json.JSONDecodeError):
        return block
    if not isinstance(packet, dict) or "goal" not in packet:
        return block

    compact = {name: packet.get(name, "") for name in ("goal", "why_the_run_stopped")}
    compact["screen_text_in_reading_order"] = []  # Fara reads the attached screenshot; OCR/menu text only distracts it.
    return {**block, "text": json.dumps(compact, ensure_ascii=False)}


def _usage(usage) -> SimpleNamespace | None:
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) or 0
    return SimpleNamespace(
        input_tokens=max(prompt - cached, 0),
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
        output_tokens=getattr(usage, "completion_tokens", None) or 0,
    )
