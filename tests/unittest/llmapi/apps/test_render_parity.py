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
"""The three ways to use the renderer give the same prompt as the chat route executes.

The usage options are the in-process Python API, the render routes embedded on a
serving worker, and the standalone CPU-only renderer's HTTP app. For every request
in the recorded corpus (``render_golden/``) each option must return exactly the
token ids the pre-merge chat or completions route handed to the engine, and must
refuse what that route refused.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.serve.render import RenderResources, UnsupportedRenderError
from tensorrt_llm.serve.render._http import build_render_app
from tensorrt_llm.serve.render.generate import mount_render_endpoints
from tensorrt_llm.serve.render.serving import RenderRequestError, ServingRender

from . import test_render_golden as golden

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

# Tokenizers whose prompts the renderer supports (a model whose own processor builds the
# prompt, and a tokenizer that cannot encode special tokens, are covered by other tests).
TOKENIZERS = ("bpe_jinja", "bpe_k3", "bpe_native", "dsv32", "dsv4", "harmony")
OPTIONS = ("python_api", "embedded_routes", "standalone_http")


def _is_named_tool_choice(body: Dict[str, Any]) -> bool:
    return isinstance(body.get("tool_choice"), dict)


def _cases():
    for tokenizer_id in TOKENIZERS:
        for case in golden.FIXTURES[tokenizer_id]["cases"]:
            if case["kind"] not in ("chat", "completion"):
                continue
            # A named tool_choice's forced prefix is added by the worker, so the renderer's
            # ids are deliberately untrusted there (tested in test_render_http).
            if _is_named_tool_choice(case["request"]):
                continue
            yield pytest.param(
                tokenizer_id,
                case["id"],
                id=f"{tokenizer_id}-{case['id']}",
                marks=[golden.needs_harmony_vocab] if tokenizer_id == "harmony" else [],
            )


CASES = list(_cases())


@pytest.fixture(scope="module")
def tokenizers():
    return golden.build_tokenizers()


@pytest.fixture(autouse=True)
def _no_legacy(monkeypatch):
    monkeypatch.delenv("TRTLLM_RENDER_LEGACY", raising=False)
    monkeypatch.delenv("TRTLLM_RENDER_IGNORED_FIELDS", raising=False)


def _case(tokenizer_id: str, case_id: str) -> Dict[str, Any]:
    case = next(c for c in golden.FIXTURES[tokenizer_id]["cases"] if c["id"] == case_id)
    return {**case, "body": case["request"]}


def _expected(case: Dict[str, Any]) -> Any:
    """What the pre-merge route executed: ``List[List[int]]``, or the refusal."""
    recorded = case["results"]["chat" if case["kind"] == "chat" else "completion"]
    if "error" in recorded:
        return None
    return [prompt["ids"] for prompt in recorded["prompts"]]


# -- the three usage options; each returns the token ids per prompt, or None if refused ---


def _python_api(resources: RenderResources, case: Dict[str, Any]):
    serving = ServingRender(lambda: resources)
    body = copy.deepcopy(case["body"])
    try:
        if case["kind"] == "chat":
            prepared = [asyncio.run(serving.render_chat(body))]
        else:
            prepared = asyncio.run(serving.render_completion(body))
    except (RenderRequestError, UnsupportedRenderError):
        return None
    return [item.token_ids for item in prepared]


def _post(app: FastAPI, case: Dict[str, Any]):
    route = "/v1/chat/completions/render" if case["kind"] == "chat" else "/v1/completions/render"
    response = TestClient(app).post(route, json=copy.deepcopy(case["body"]))
    if response.status_code != 200:
        assert 400 <= response.status_code < 500, response.text
        return None
    payload = response.json()
    items: List[Dict[str, Any]] = payload if isinstance(payload, list) else [payload]
    return [item["token_ids"] for item in items]


def _embedded_routes(resources, case, server):
    app = FastAPI()
    assert mount_render_endpoints(app, server, enabled=True)
    return _post(app, case)


def _standalone_http(resources, case, server):
    return _post(build_render_app(resources), case)


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("tokenizer_id,case_id", CASES)
def test_every_option_gives_the_prompt_the_chat_route_executed(
    tokenizers, tokenizer_id, case_id, option
) -> None:
    entry = tokenizers[tokenizer_id]
    case = _case(tokenizer_id, case_id)
    server = golden._stub_server(entry, golden._cfg(entry, case), [])
    resources = RenderResources.from_server(server)

    if option == "python_api":
        got = _python_api(resources, case)
    elif option == "embedded_routes":
        got = _embedded_routes(resources, case, server)
    else:
        got = _standalone_http(resources, case, server)

    assert got == _expected(case)
