# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay requests recorded from the code before the renderer merge.

``render_golden/*.json`` holds, for each request and each place that renders a
prompt (chat route, completions route, multimodal-encoder route, router,
``count_tokens``, KV-cache governor, Responses), what the pre-merge code produced:
the prompt text, the token ids, or the error. They were recorded by running the
real pre-merge handlers on the base commit recorded in each file's ``meta``.

Every case is replayed here through the merged code and must match, except the
cases in ``INTENDED_CHANGES``, each of which names the deliberate behavior change
that explains the difference. A difference that is not in the table fails.
"""

from __future__ import annotations

import asyncio
import copy
import glob
import hashlib
import json
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "render_golden")

# --------------------------------------------------------------------------------------
# What changed on purpose. (tokenizer id, case id, path) -> why. Anything else must be equal.
# --------------------------------------------------------------------------------------

SERVER_TEMPLATE_NOW_APPLIES = (
    "the governor, Responses and multimodal-encoder paths now apply the server "
    "--chat_template like the chat route does (they ignored it before)"
)
NO_BOS_COUNT = (
    "count_tokens counted with the tokenizer's default special tokens (a BOS the chat route "
    "does not add); it now counts the prompt the chat route executes"
)
LIST_OF_CHARACTERS = (
    "for a tokenizer that renders the chat text itself, the governor and Responses paths "
    "returned the rendered text as a list of characters (their tokenize=True call is a no-op "
    "there); they now return real token ids"
)
ROUTER_HONORS_ADD_SPECIAL_TOKENS = (
    "the router ignored add_special_tokens; it now tokenizes like the chat route (BOS added)"
)
ROUTER_MATCHES_ROUTE = (
    "the router rendered the conversation its own way (content parts, empty tool-call "
    "content, truncate_prompt_tokens); it now equals the prompt the chat route executes"
)
ROUTER_NO_WRITE_BACK = (
    "ids are routed on but no longer written back into the request (a named tool_choice's "
    "forced prefix is added by the worker; a model whose own processor builds the prompt, or "
    "a request the core rejects, is routed on an estimate)"
)
ROUTER_ESTIMATES_UNRECOVERABLE_MESSAGES = (
    "the router crashed on messages the request model rejects (object-valued tool-call "
    "arguments, which the chat route accepts from the raw body); it now routes on an estimate "
    "and leaves the rendering to the worker"
)
ROUTER_APPLIES_EXTENSION = (
    "the router ignored the model extension's request-to-template mapping (thinking, "
    "response_format, tool_choice) and wrote back ids the chat route would not have produced; "
    "it now applies the extension like the chat route"
)
INTENDED_CHANGES: Dict[Any, str] = {}

NATIVE = ("bpe_native", "dsv32", "dsv4", "passthrough")
ALL_TOKENIZERS = (
    "bpe_jinja",
    "bpe_k3",
    "bpe_native",
    "dsv32",
    "dsv4",
    "passthrough",
    "tiktoken_like",
)


def _declare(tokenizers, case_ids, paths, reason):
    """Mark replay of ``case_ids`` (or every case, for ``None``) on ``paths`` as changed."""
    for tokenizer in tokenizers:
        document = _FIXTURE_CASES.get(tokenizer, {})
        for case in case_ids if case_ids is not None else list(document):
            for path in paths:
                INTENDED_CHANGES[(tokenizer, case, path)] = reason


def _fixture_cases() -> Dict[str, Dict[str, Any]]:
    out = {}
    for path in sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.json"))):
        with open(path) as handle:
            document = json.load(handle)
        out[document["meta"]["tokenizer_id"]] = {c["id"]: c for c in document["cases"]}
    return out


_FIXTURE_CASES = _fixture_cases()

_declare(
    ["bpe_jinja"],
    ["chat.server_template"],
    ["governor", "responses", "mm_encoder"],
    SERVER_TEMPLATE_NOW_APPLIES,
)
_declare(
    ["bpe_jinja"],
    ["anthropic.plain", "anthropic.system", "anthropic.tools", "anthropic.server_template"],
    ["count_tokens"],
    NO_BOS_COUNT,
)
_declare(
    ["bpe_native", "dsv32", "dsv4"],
    ["anthropic.plain"],
    ["count_tokens"],
    NO_BOS_COUNT,
)
_declare(
    NATIVE,
    [c for c in _FIXTURE_CASES["bpe_native"] if c.startswith("chat.")],
    ["governor", "responses"],
    LIST_OF_CHARACTERS,
)
_declare(ALL_TOKENIZERS, ["chat.add_special_tokens"], ["router"], ROUTER_HONORS_ADD_SPECIAL_TOKENS)
_declare(["bpe_jinja"], ["chat.content_parts", "chat.truncate_5"], ["router"], ROUTER_MATCHES_ROUTE)
_declare(["bpe_native", "passthrough"], ["chat.tool_loop"], ["router"], ROUTER_MATCHES_ROUTE)
_declare(
    ["bpe_jinja"],
    ["chat.tool_loop_obj_args"],
    ["router"],
    ROUTER_ESTIMATES_UNRECOVERABLE_MESSAGES,
)
_declare(
    ["bpe_jinja"],
    [
        "chat.named_tool_choice_with_parser",
        "chat.named_tool_choice_without_parser",
        "chat.tool_choice_required",
    ],
    ["router"],
    ROUTER_NO_WRITE_BACK,
)
_declare(
    ["passthrough"],
    [
        "chat.documents",
        "chat.reasoning_history",
        "chat.single",
        "chat.thinking_kwarg",
        "chat.three_turns",
        "chat.tools",
    ],
    ["router"],
    ROUTER_NO_WRITE_BACK,
)
_declare(
    ["bpe_k3"],
    [
        "chat.reasoning_effort",
        "chat.response_format_json_object",
        "chat.response_format_json_schema",
        "chat.thinking_effort",
        "chat.tools_required",
    ],
    ["router"],
    ROUTER_APPLIES_EXTENSION,
)

# --------------------------------------------------------------------------------------
# Tokenizers: rebuilt exactly as when the fixtures were recorded (the vocabulary hash in
# each fixture's meta proves it)
# --------------------------------------------------------------------------------------

JINJA_TEMPLATE = (
    "{%- if documents %}{% for d in documents %}[doc:{{ d.title }}|{{ d.text }}]{% endfor %}\n"
    "{% endif -%}"
    "{%- if tools %}[tools:{% for t in tools %}{{ t.function.name }}"
    "{{ ',' if not loop.last }}{% endfor %}]\n{% endif -%}"
    "{%- if thinking is defined and thinking %}[thinking]\n{% endif -%}"
    "{%- for m in messages %}"
    "{%- if m.role == 'tool' %}<tool>{{ m.content }}\n"
    "{%- else %}<{{ m.role }}>"
    "{%- if m.reasoning_content is defined and m.reasoning_content %}"
    "[reasoning:{{ m.reasoning_content }}]{% endif -%}"
    "{%- if m.content is string %}{{ m.content }}"
    "{%- elif m.content %}{% for part in m.content %}"
    "{{ part.text if part.text is defined else part }}{% endfor %}{% endif -%}"
    "{%- if m.tool_calls is defined and m.tool_calls %}{% for tc in m.tool_calls %}"
    "[call:{{ tc.function.name }}({{ tc.function.arguments | tojson }})]{% endfor %}{% endif %}\n"
    "{%- endif %}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt %}<assistant>{% endif -%}"
)
K3_PREFIX = (
    "{%- if thinking is defined %}[thinking={{ thinking }}]{% endif -%}"
    "{%- if thinking_effort is defined %}[effort={{ thinking_effort }}]{% endif -%}"
    "{%- if tool_choice is defined %}[tool_choice={{ tool_choice }}]{% endif -%}"
    "{%- if response_format is defined %}[rf={{ response_format }}]{% endif -%}"
    "{%- if response_schema is defined %}[schema]{% endif -%}"
)
K3_TEMPLATE = K3_PREFIX + JINJA_TEMPLATE


def _make_bpe(chat_template, cls=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=360,
        special_tokens=["<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = [
        "hello world this is a test of the chat template for rendering prompts",
        "user assistant system tool weather city paris get_weather search tools",
        "thinking reasoning call tool_choice response document title text",
    ] * 30
    backend.train_from_iterator(corpus, trainer)
    backend.post_processor = TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", backend.token_to_id("<s>"))]
    )
    return (cls or PreTrainedTokenizerFast)(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        chat_template=chat_template,
    )


def _native_class():
    from transformers import PreTrainedTokenizerFast

    class NativeRendererTokenizer(PreTrainedTokenizerFast):
        def apply_chat_template(self, conversation, tools=None, tokenize=False, **kwargs):
            parts = ["N"]
            if tools:
                parts.append("tools=" + ",".join(t["function"]["name"] for t in tools))
            if kwargs.get("documents"):
                parts.append(f"docs={len(kwargs['documents'])}")
            for message in conversation:
                parts.append(f"{message['role']}:{message.get('content')}")
            if kwargs.get("add_generation_prompt"):
                parts.append("assistant:")
            return "|".join(parts)

    return NativeRendererTokenizer


def _tiktoken_like_class():
    from transformers import PreTrainedTokenizerFast

    class TiktokenLikeTokenizer(PreTrainedTokenizerFast):
        def encode(self, text, *args, **kwargs):
            if "add_special_tokens" in kwargs:
                raise TypeError("encode() got an unexpected keyword argument 'add_special_tokens'")
            kwargs.pop("allowed_special", None)
            return super().encode(text, add_special_tokens=False)

    return TiktokenLikeTokenizer


def _wrap(hf):
    from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer

    return TransformersTokenizer(hf)


def build_tokenizers() -> Dict[str, Dict[str, Any]]:
    from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
    from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer

    out: Dict[str, Dict[str, Any]] = {}
    plain = _make_bpe(JINJA_TEMPLATE)
    out["bpe_jinja"] = dict(tokenizer=_wrap(plain), raw=plain, model_type="golden-generic")
    k3 = _make_bpe(K3_TEMPLATE)
    out["bpe_k3"] = dict(tokenizer=_wrap(k3), raw=k3, model_type="kimi_k3")
    native = _make_bpe(None, cls=_native_class())
    out["bpe_native"] = dict(tokenizer=_wrap(native), raw=native, model_type="golden-generic")
    ds32 = DeepseekV32Tokenizer(_make_bpe(None))
    out["dsv32"] = dict(tokenizer=ds32, raw=ds32, model_type="deepseek_v32")
    ds4 = DeepseekV4Tokenizer(_make_bpe(None))
    out["dsv4"] = dict(tokenizer=ds4, raw=ds4, model_type="deepseek_v4")
    import tensorrt_llm._torch.models.modeling_mistral  # noqa: F401  (registers the model type)

    out["passthrough"] = dict(tokenizer=_wrap(plain), raw=plain, model_type="mistral_common")
    tiktoken = _make_bpe(JINJA_TEMPLATE, cls=_tiktoken_like_class())
    out["tiktoken_like"] = dict(
        tokenizer=_wrap(tiktoken), raw=tiktoken, model_type="golden-generic"
    )
    return out


def _vocab_digest(entry) -> str:
    inner = getattr(entry["raw"], "tokenizer", entry["raw"])
    return hashlib.sha256(json.dumps(sorted(inner.get_vocab().items())).encode()).hexdigest()


# --------------------------------------------------------------------------------------
# Drivers: the same harness the fixtures were recorded with, pointed at the merged code
# --------------------------------------------------------------------------------------


def _cfg(entry, case) -> Dict[str, Any]:
    cfg = {
        "model_type": entry["model_type"],
        "chat_template": None,
        "allow_request_chat_template": False,
        "tool_parser": None,
        "guided_decoding_backend": None,
    }
    cfg.update(case["server"])
    return cfg


def _error(exc: BaseException) -> Dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}"[:600]}


def _default_ids(tokenizer, inputs, sampling_params) -> List[int]:
    from tensorrt_llm.inputs.registry import DefaultInputProcessor

    ids, _ = DefaultInputProcessor(None, None, tokenizer)(inputs, sampling_params)
    return list(ids)


def _stub_server(entry, cfg, captured):
    from tensorrt_llm.inputs.registry import DefaultInputProcessor
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        UsageInfo,
    )
    from tensorrt_llm.serve.openai_server import OpenAIServer

    model_config = type("ModelConfig", (), {"model_type": cfg["model_type"], "vocab_size": 1000})()

    def generate_async(*, inputs, **kwargs):
        captured.append({"inputs": inputs, "sampling_params": kwargs.get("sampling_params")})
        promise = SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            finished=True,
            request_id=1,
            disaggregated_params=SimpleNamespace(
                multimodal_embedding_handles=[{"tensor_size": [4, 8]}]
            ),
        )

        async def aresult():
            return promise

        promise.aresult = aresult
        return promise

    server = object.__new__(OpenAIServer)
    server.model = "m"
    server.allow_request_chat_template = cfg["allow_request_chat_template"]
    server.model_config = model_config
    server.processor = None
    server.tokenizer = entry["tokenizer"]
    server.chat_template = cfg["chat_template"]
    server.tool_parser = cfg["tool_parser"]
    server.tool_call_id_type = "random"
    server.multimodal_server_config = None
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            gather_generation_logits=False,
            reasoning_parser=None,
            backend="pytorch",
            guided_decoding_backend=cfg.get("guided_decoding_backend"),
            num_postprocess_workers=0,
        ),
        generate_async=generate_async,
        input_processor=DefaultInputProcessor(None, None, entry["tokenizer"]),
    )
    server.await_disconnected = AsyncMock()
    server._create_chat_response = AsyncMock(
        return_value=ChatCompletionResponse(
            id="x",
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
    )
    server._create_completion_response = AsyncMock()
    server._input_proc_executor = ThreadPoolExecutor(max_workers=1)
    return server


def _record_prompts(entry, captured) -> List[Dict[str, Any]]:
    out = []
    for item in captured:
        inputs = item["inputs"]
        if isinstance(inputs, dict) and "prompt" in inputs:
            ids = _default_ids(entry["tokenizer"], inputs, item["sampling_params"])
            out.append({"text": inputs["prompt"], "ids": ids})
        elif isinstance(inputs, dict):
            out.append({"text": None, "ids": list(inputs["prompt_token_ids"])})
        elif isinstance(inputs, str):
            ids = _default_ids(entry["tokenizer"], {"prompt": inputs}, item["sampling_params"])
            out.append({"text": inputs, "ids": ids})
        else:
            out.append({"text": None, "ids": list(inputs)})
    return out


def _post(route, handler, body):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.add_api_route(route, handler, methods=["POST"])
    response = TestClient(app).post(route, json=body)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return response.status_code, payload


def _message(payload):
    return (payload or {}).get("message") if isinstance(payload, dict) else str(payload)


def drive_chat(entry, case):
    captured: list = []
    server = _stub_server(entry, _cfg(entry, case), captured)
    status, payload = _post("/v1/chat/completions", server.openai_chat, case["body"])
    if status != 200:
        return {"status": status, "error": _message(payload)}
    return {"status": status, "prompts": _record_prompts(entry, captured)}


def drive_completion(entry, case):
    captured: list = []
    server = _stub_server(entry, _cfg(entry, case), captured)
    _, payload = _post("/v1/completions", server.openai_completion, case["body"])
    if not captured:
        return {"error": _message(payload)}
    return {"prompts": _record_prompts(entry, captured)}


def drive_mm_encoder(entry, case):
    captured: list = []
    server = _stub_server(entry, _cfg(entry, case), captured)
    status, payload = _post("/v1/chat/completions", server.openai_mm_encoder, case["body"])
    if status != 200:
        return {"status": status, "error": _message(payload)}
    prompts = []
    for item in captured:
        inputs = item["inputs"]
        is_dict = isinstance(inputs, dict)
        prompts.append(
            {
                "text": inputs.get("prompt") if is_dict else None,
                "ids": inputs.get("prompt_token_ids") if is_dict else None,
            }
        )
    return {"status": status, "prompts": prompts}


def drive_router(entry, case):
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest
    from tensorrt_llm.serve.router import KvCacheAwareRouter

    router = KvCacheAwareRouter(
        server_role=None,
        servers=["s1"],
        use_tokens=False,
        max_batch_size=32,
        tokens_per_block=32,
        use_harmony=False,
    )
    with mock.patch.object(router, "_get_tokenizer", return_value=entry["raw"]):
        # The router resolves the model type from a checkpoint path; give it one.
        router._model_path = "/golden/model"
        router._model_types["/golden/model"] = _cfg(entry, case)["model_type"]
        cls = CompletionRequest if case["kind"] == "completion" else ChatCompletionRequest
        try:
            request = cls(**copy.deepcopy(case["body"]))
        except Exception as exc:  # noqa: BLE001 - the request model's own rejection
            return {"request_error": str(exc)[:400]}
        if case["kind"] == "chat":
            # A worker that reports the same rendering configuration, so ids are forwarded.
            fingerprint = router._render_resources("m").fingerprint()
            router._server_info = {"s1": {"render_fingerprint": fingerprint}}
        try:
            token_lists = router._tokenize(request)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
    out: Dict[str, Any] = {"token_lists": [list(t) for t in token_lists]}
    if case["kind"] == "chat":
        out["wrote_back"] = request.prompt_token_ids
    else:
        out["prompt_after"] = request.prompt
    return out


def drive_count_tokens(entry, case):
    captured: list = []
    server = _stub_server(entry, _cfg(entry, case), captured)
    status, payload = _post(
        "/v1/messages/count_tokens", server.anthropic_count_tokens, case["body"]
    )
    if status == 200:
        return {"status": status, "input_tokens": payload.get("input_tokens")}
    return {"status": status, "error": json.dumps(payload)[:400]}


def drive_governor(entry, case):
    from tensorrt_llm.serve.openai_protocol import KVCacheTruncateRequest
    from tensorrt_llm.serve.resource_governor import ResourceGovernor

    cfg = _cfg(entry, case)
    body = case["body"]
    q: "queue.Queue" = queue.Queue()
    model_config = type("ModelConfig", (), {"model_type": cfg["model_type"]})()
    governor = ResourceGovernor(
        resource_governor_queue=q,
        tokenizer=entry["tokenizer"],
        model_config=model_config,
        processor=None,
        allow_request_chat_template=cfg["allow_request_chat_template"],
        chat_template=cfg["chat_template"],
    )
    fields = {"model": "m", "messages": body["messages"]}
    for key in (
        "tools",
        "add_generation_prompt",
        "documents",
        "chat_template",
        "chat_template_kwargs",
    ):
        if key in body:
            fields[key] = body[key]
    try:
        request = KVCacheTruncateRequest(**copy.deepcopy(fields))
    except Exception as exc:  # noqa: BLE001
        return {"request_error": str(exc)[:400]}
    try:
        response = asyncio.run(governor._truncate_kv_cache(request))
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    out: Dict[str, Any] = {"status": response.status_code}
    if response.status_code == 200:
        out["ids"] = list(q.get_nowait().messages)
    else:
        out["error"] = response.body.decode()[:400]
    return out


def drive_responses(entry, case):
    import tensorrt_llm.serve.responses_utils as responses_utils
    from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam

    cfg = _cfg(entry, case)
    body = case["body"]
    model_config = type("ModelConfig", (), {"model_type": cfg["model_type"]})()
    tools = [ChatCompletionToolsParam.model_validate(t) for t in body.get("tools", [])]

    async def fake_input_messages(request, prev_msgs):
        return copy.deepcopy(body["messages"])

    request = mock.Mock()
    request.tools = body.get("tools") or None
    request.store = False
    kwargs = body.get("chat_template_kwargs") or {}
    with (
        mock.patch.object(responses_utils, "_create_input_messages", fake_input_messages),
        mock.patch.object(responses_utils, "_get_chat_completion_function_tools", lambda t: tools),
        mock.patch.object(responses_utils, "reasoning_chat_template_kwargs", lambda r: kwargs),
        mock.patch.object(
            responses_utils, "reasoning_injected_chat_template_keys", lambda r: set()
        ),
    ):
        try:
            ids, _mm = asyncio.run(
                responses_utils._create_input_tokens(
                    request=request,
                    prev_response=None,
                    prev_msgs=None,
                    conversation_store=None,
                    enable_store=False,
                    tokenizer=entry["tokenizer"],
                    model_config=model_config,
                    processor=None,
                    chat_template=cfg["chat_template"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
    return {"ids": list(ids)}


DRIVERS = {
    "chat": drive_chat,
    "completion": drive_completion,
    "mm_encoder": drive_mm_encoder,
    "router": drive_router,
    "count_tokens": drive_count_tokens,
    "governor": drive_governor,
    "responses": drive_responses,
}

# --------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------


def _load_fixtures() -> Dict[str, Dict[str, Any]]:
    fixtures = {}
    for path in sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.json"))):
        with open(path) as handle:
            document = json.load(handle)
        fixtures[document["meta"]["tokenizer_id"]] = document
    return fixtures


FIXTURES = _load_fixtures()
PARAMS = [
    pytest.param(tok, case["id"], path, id=f"{tok}-{case['id']}-{path}")
    for tok, document in FIXTURES.items()
    for case in document["cases"]
    for path in case["results"]
]


@pytest.fixture(scope="module")
def tokenizers():
    return build_tokenizers()


@pytest.fixture(autouse=True)
def _no_legacy(monkeypatch):
    monkeypatch.delenv("TRTLLM_RENDER_LEGACY", raising=False)
    monkeypatch.delenv("TRTLLM_RENDER_IGNORED_FIELDS", raising=False)


def test_the_fixtures_exist() -> None:
    assert set(FIXTURES) == {
        "bpe_jinja",
        "bpe_k3",
        "bpe_native",
        "dsv32",
        "dsv4",
        "passthrough",
        "tiktoken_like",
    }
    assert len(PARAMS) > 300


@pytest.mark.parametrize("tokenizer_id", sorted(FIXTURES))
def test_the_rebuilt_tokenizer_is_the_one_the_fixtures_were_recorded_with(
    tokenizers, tokenizer_id
) -> None:
    meta = FIXTURES[tokenizer_id]["meta"]["tokenizer"]
    # The vocabulary hash only holds for the same tokenizers/transformers build, so a
    # mismatch is reported as that rather than as a rendering regression.
    assert _vocab_digest(tokenizers[tokenizer_id]) == meta["vocab_sha256"]


@pytest.mark.parametrize("tokenizer_id,case_id,path", PARAMS)
def test_replay(tokenizers, tokenizer_id, case_id, path) -> None:
    document = FIXTURES[tokenizer_id]
    case = next(c for c in document["cases"] if c["id"] == case_id)
    recorded = case["results"][path]
    assert "driver_error" not in recorded, recorded
    case = {**case, "body": case["request"]}

    replayed = json.loads(json.dumps(DRIVERS[path](tokenizers[tokenizer_id], case)))

    reason = INTENDED_CHANGES.get((tokenizer_id, case_id, path))
    if reason is None:
        assert replayed == recorded
    else:
        # A declared change must actually change something, or the entry is stale.
        assert replayed != recorded, f"stale INTENDED_CHANGES entry ({reason})"
