import json
from types import SimpleNamespace

import pytest

from typesafe_computer_use.fara_writer import FaraWriter, FaraWriterError, _compact_review_context, normalize_tool_reply
from typesafe_computer_use.writer import compose_text


def _call(action: str, **arguments) -> str:
    return f"<tool_call>{json.dumps({'name': 'computer_use', 'arguments': {'action': action, **arguments}})}</tool_call>"


def test_type_action_becomes_only_a_field_text_proposal():
    assert normalize_tool_reply(_call("type", text="hello"), "field") == {
        "fill": True,
        "text": "hello",
        "reason": "Fara proposed field text.",
        "submit": False,
    }


def test_terminate_can_decline_a_field():
    assert normalize_tool_reply(_call("terminate", answer=""), "field") == {
        "fill": False,
        "text": "",
        "reason": "",
        "submit": False,
    }


def test_terminate_review_maps_its_separate_fields():
    review = {"achieved": True, "answer": "The value is visible.", "focus": "", "question": ""}
    assert normalize_tool_reply(_call("terminate", **review), "review") == review


@pytest.mark.parametrize("action", ["left_click", "right_click", "scroll", "visit_url", "key", "web_search"])
def test_field_writer_rejects_every_computer_action_except_type_or_terminate(action):
    with pytest.raises(FaraWriterError, match="unsupported field action"):
        normalize_tool_reply(_call(action, text="hello"), "field")


def test_screen_reviewer_rejects_a_type_action():
    with pytest.raises(FaraWriterError, match="unsupported review action"):
        normalize_tool_reply(_call("type", text="hello"), "review")


@pytest.mark.parametrize(
    "reply",
    [
        "no tool call",
        '<tool_call>{"name":"computer_use","arguments":{"action":"type","text":"x"}</tool_call>',
        _call("type", text="x") + _call("type", text="y"),
        '<tool_call>{"name":"other","arguments":{"action":"type","text":"x"}}</tool_call>',
        '<tool_call>{"name":"computer_use","arguments":{"action":"type","text":7}}</tool_call>',
        '<tool_call>{"name":"computer_use","arguments":{"action":"type","text":"x","coordinate":[1,2]}}</tool_call>',
    ],
)
def test_malformed_or_unsupported_native_output_is_rejected(reply):
    with pytest.raises(FaraWriterError):
        normalize_tool_reply(reply, "field")


@pytest.mark.parametrize(
    "arguments",
    [
        {"answer": "ok"},
        {"answer": "ok", "achieved": "true", "focus": "", "question": ""},
        {"answer": "ok", "achieved": True, "focus": None, "question": ""},
        {"answer": "ok", "achieved": True, "focus": "", "question": "", "extra": 0},
    ],
)
def test_final_review_requires_the_exact_typed_shape(arguments):
    with pytest.raises(FaraWriterError, match="final review"):
        normalize_tool_reply(_call("terminate", **arguments), "review")


def test_fara_writer_sends_screen_images_and_requests_native_output(monkeypatch):
    captured = {}

    class Completion:
        @staticmethod
        def create(**request):
            captured.update(request)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=_call("type", text="FARA-CHECK-0123")))],
                usage=None,
            )

    fake_client = SimpleNamespace(base_url="http://127.0.0.1:11434/v1/", chat=SimpleNamespace(completions=Completion))
    monkeypatch.setattr("typesafe_computer_use.fara_writer.openai.OpenAI", lambda **_: fake_client)
    writer = FaraWriter("http://127.0.0.1:11434/v1", "local")
    reply = writer.messages.create(
        model="fara1p5-4b-q6k-16k",
        max_tokens=256,
        system="Fill this focused field.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"}},
                    {"type": "text", "text": "FARA-CHECK-0123"},
                ],
            }
        ],
        output_config={"format": {"schema": {"properties": {"fill": {}, "text": {}}}}},
    )

    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 1024
    assert captured["extra_body"] == {"think": False}
    assert "response_format" not in captured
    image = captured["messages"][1]["content"][0]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"] == "data:image/png;base64,cG5n"
    assert json.loads(reply.content[0].text)["text"] == "FARA-CHECK-0123"


def test_review_payload_keeps_only_goal_stop_reason_and_screen_text():
    block = {
        "type": "text",
        "text": json.dumps(
            {
                "goal": "check the field",
                "why_the_run_stopped": "the value is visible",
                "screen_text_in_reading_order": ["FARA-CHECK-0123"],
                "actions_taken": ["typed the value"],
                "user_can_be_asked": False,
                "frontmost_app": "Notepad",
            }
        ),
    }
    assert json.loads(_compact_review_context(block)["text"]) == {
        "goal": "check the field",
        "why_the_run_stopped": "the value is visible",
        "screen_text_in_reading_order": [],
    }


def test_desktop_field_writer_passes_its_capture_to_fara(clean_env, screen, monkeypatch):
    clean_env.setenv("CLICKER_WRITER_API", "fara")
    seen = {}

    def create(**request):
        seen.update(request)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"fill":true,"text":"visible text","reason":"","submit":false}')]
        )

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    compose_text(fake, "type visible text", screen, [], [])
    assert [block["type"] for block in seen["messages"][0]["content"]] == ["image", "text"]
    assert seen["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
