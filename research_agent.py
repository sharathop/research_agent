"""
research_agent.py

Scaled, fixed version of the single-app proof-of-concept.

What changed vs. the original test_config.py:

1. PARAMETERIZED for all 100 apps (apps_registry.py) instead of a
   hardcoded "Notion" system prompt. One function does the research
   for any (app_name, category, hint) tuple.

2. FIXED the Composio call inconsistency. The original called
   COMPOSIO_SEARCH_TOOLS / COMPOSIO_GET_TOOL_SCHEMAS via
   `session.execute(name, arguments=...)` but called
   COMPOSIO_MULTI_EXECUTE_TOOL via
   `session.tools().COMPOSIO_MULTI_EXECUTE_TOOL(**arguments)`,
   which is not a documented Composio SDK call pattern and would
   raise at runtime. Per Composio's own docs (session.execute is the
   single entry point for all session meta-tools, including
   COMPOSIO_MULTI_EXECUTE_TOOL), all three now go through
   `session.execute(name, arguments=arguments)`.

3. ONE Composio session + ONE session.tools() fetch, reused across
   all 100 apps, instead of re-fetching per app.

4. CONCURRENCY with a bounded worker pool (default 4) so 100 apps
   finish inside the time budget, plus retry/backoff on rate-limit /
   transient errors so one flaky call doesn't kill the whole run.

5. PER-APP FAULT ISOLATION: a crash researching one app is caught,
   logged, and the app is marked needs_human_review instead of
   stopping the batch.

6. A REAL VERIFICATION PASS: for the two claims that drive the
   buildability verdict the most (credential_access and api.mcp),
   the agent re-checks with an independent web search + a second,
   separately-prompted "verifier" LLM call that only sees the claim
   and the fresh evidence (not the original reasoning), and can
   confirm / contradict / mark inconclusive. Contradictions downgrade
   confidence and flag the app for human review, so the accuracy
   improvement from this loop is visible and auditable afterward
   (see verification.first_pass vs verification.after_verification
   in each record, and needs_human_review.json for the batch).

7. Incremental writes: every app's result is written to
   results/<app>.json as soon as it finishes, plus a combined
   all_results.json and needs_human_review.json at the end, so a
   crash near app #90 doesn't lose the first 89.

8. TOKEN BUDGET CONTROL (added after the first pass, since 100 apps
   worth of Groq calls can exceed a real rate limit fast):
   - The tool-calling research loop now runs on a small/fast model
     (LOOP_MODEL, default llama-3.1-8b-instant) instead of the big
     model - it's the step called most often per app (up to
     MAX_STEPS times), and it only needs to pick tools/arguments
     correctly, not write good prose. The big model (SYNTH_MODEL,
     default openai/gpt-oss-120b) is reserved for the two calls per
     app where output quality matters: the final JSON report and the
     verification verdicts.
   - The two verification calls (credential_access, api.mcp) were
     merged into one combined call per app, halving that cost.
   - max_tokens caps on every call type prevent a verbose response
     from blowing past expectations on its own.
   - Every call's usage is recorded (TOKEN_USAGE); pass
     --token-budget N to stop starting NEW apps once N total tokens
     have been used (already-running apps still finish). See
     run_summary.json for the final tally.
   - MAX_WORKERS defaults lower (3) and MAX_STEPS lower (4), since
     concurrency and step count both multiply directly against
     tokens-per-minute limits.

This script intentionally does NOT build the aggregation/pattern
report or the HTML case study - that's a separate step on top of
these per-app JSON records.
"""

import os
import re
import json
import time
import random
import argparse
import threading
import traceback
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from groq import Groq
from composio import Composio
from tavily import TavilyClient

from apps_registry import APPS


# ============================================================
# 1. ENVIRONMENT / CLIENTS (created once, shared across apps)
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

for _name, _val in [
    ("GROQ_API_KEY", GROQ_API_KEY),
    ("COMPOSIO_API_KEY", COMPOSIO_API_KEY),
    ("TAVILY_API_KEY", TAVILY_API_KEY),
]:
    if not _val:
        raise ValueError(f"{_name} is missing from .env")

LOOP_MODEL = os.getenv("GROQ_LOOP_MODEL", "openai/gpt-oss-20b")
SYNTH_MODEL = os.getenv("GROQ_SYNTH_MODEL", "openai/gpt-oss-120b")

MAX_TOKENS_LOOP = 500
MAX_TOKENS_SYNTH = 1600
MAX_TOKENS_VERIFY = 250

groq_client = Groq(api_key=GROQ_API_KEY)
composio = Composio(api_key=COMPOSIO_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

try:
    _available_model_ids = {m.id for m in groq_client.models.list().data}
    _all_configured_models = {("GROQ_LOOP_MODEL", LOOP_MODEL), ("GROQ_SYNTH_MODEL", SYNTH_MODEL)}
    _fallback_models = [
        "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
    ]
    for _fm in _fallback_models:
        _all_configured_models.add(("fallback chain", _fm))
    for _label, _model in _all_configured_models:
        if _model not in _available_model_ids:
            raise ValueError(
                f"{_label}={_model!r} is not available on this Groq account. "
                f"Available models: {sorted(_available_model_ids)}. "
                f"Set {_label} in your .env to one of these, or check "
                f"https://console.groq.com/docs/models."
            )
except ValueError:
    raise
except Exception as e:  # noqa: BLE001
    print(f"Warning: could not verify model availability ahead of time ({e}). Proceeding anyway.")

session = composio.sessions.create(user_id="composio-research-agent", mcp=True)

ALL_TOOLS = session.tools()
print(f"Composio session ready. {len(ALL_TOOLS)} tools available.")

ALLOWED_TOOL_NAMES = {
    "COMPOSIO_SEARCH_TOOLS",
    "COMPOSIO_GET_TOOL_SCHEMAS",
    "COMPOSIO_MULTI_EXECUTE_TOOL",
}

GROQ_TOOLS = []
for _tool in ALL_TOOLS:
    _tool_name = _tool["function"]["name"]
    if _tool_name not in ALLOWED_TOOL_NAMES:
        continue
    _function = _tool["function"].copy()
    if _function.get("strict") is None:
        _function.pop("strict", None)
    GROQ_TOOLS.append({"type": "function", "function": _function})

BROWSER_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "browser.search",
        "description": (
            "Search the public web for missing research information "
            "about an application. Prefer official documentation, "
            "official pricing pages, and official developer pages. "
            "This is the ONLY browser operation available. Do not "
            "call browser.open or any other browser tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query."},
                "top_n": {
                    "type": "integer",
                    "description": "Maximum number of results.",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

GROQ_TOOLS.append(BROWSER_SEARCH_TOOL)

MAX_STEPS = 4
MAX_WORKERS = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_usage_lock = threading.Lock()
TOKEN_USAGE = {"total": 0, "by_model": {}}
TOKEN_BUDGET = None
BUDGET_EXCEEDED = threading.Event()


def record_usage(response, model):
    usage = getattr(response, "usage", None)
    total = getattr(usage, "total_tokens", None) if usage else None
    if total is None:
        return
    with _usage_lock:
        TOKEN_USAGE["total"] += total
        TOKEN_USAGE["by_model"][model] = TOKEN_USAGE["by_model"].get(model, 0) + total
        if TOKEN_BUDGET is not None and TOKEN_USAGE["total"] >= TOKEN_BUDGET:
            BUDGET_EXCEEDED.set()


TPM_LIMITS = {
    "openai/gpt-oss-20b": 8000,
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-safeguard-20b": 8000,  # unconfirmed exact number for
        # this account - assumed same as the other gpt-oss models until
        # checked against console.groq.com/settings/limits. If it's
        # actually different, groq_chat()'s record_actual() self-corrects
        # over the run, so a wrong starting guess here just paces
        # slightly too conservatively or loosely at first, not fatally.
    "groq/compound": 70000,
    "groq/compound-mini": 70000,
    "qwen/qwen3.8-27b": 8000,
}


class TokenRateLimiter:
    def __init__(self, tpm_limit):
        self.tpm_limit = tpm_limit
        self.window = collections.deque()
        self.lock = threading.Lock()

    def _used_in_window(self, now):
        cutoff = now - 60
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()
        return sum(tok for _, tok in self.window)

    def acquire(self, estimated_tokens):
        while True:
            with self.lock:
                now = time.time()
                used = self._used_in_window(now)
                if used + estimated_tokens <= self.tpm_limit or not self.window:
                    self.window.append((now, estimated_tokens))
                    return
                wait = 60 - (now - self.window[0][0]) + 0.2
            time.sleep(max(wait, 0.5))

    def record_actual(self, estimated_tokens, actual_tokens):
        with self.lock:
            for i in range(len(self.window) - 1, -1, -1):
                ts, tok = self.window[i]
                if tok == estimated_tokens:
                    self.window[i] = (ts, actual_tokens)
                    return


_rate_limiters = {model: TokenRateLimiter(limit) for model, limit in TPM_LIMITS.items()}


def _estimate_tokens(messages, max_tokens, tools=None):
    chars = sum(len(m.get("content") or "") for m in messages)
    if tools:
        chars += len(json.dumps(tools, default=str))
    return (chars // 4) + max_tokens


# Groq enforces a HARD per-request input-token cap (ITPM) separate from
# the rolling 60s TPM window - confirmed via a live 413: "Limit 7000,
# Requested 7083" on qwen/qwen3.8-27b. TokenRateLimiter above can't
# prevent this: a single oversized request can exceed it even with a
# completely empty minute, especially with PARALLEL tool calls in one
# step (multiple browser.search calls in the same turn each add their
# own tool-result message before the next call is even made). And
# retrying via call_with_retry doesn't help - the request is the same
# size on every attempt, so it just fails 4 times identically.
# This shrinks the actual conversation in place (mutating `messages`,
# so the reduction persists for the rest of this app's loop) whenever
# a request would cross the cap, by further truncating the OLDEST
# tool-role message contents first. It never removes a message -
# doing so would break the tool_call_id pairing the API requires.
HARD_REQUEST_TOKEN_CAP = 6300  # safety margin under the smallest
                                # confirmed per-request limit (7000)


def _shrink_messages_in_place(messages, tools, max_tokens, cap=HARD_REQUEST_TOKEN_CAP):
    def est():
        return _estimate_tokens(messages, max_tokens, tools=tools)
    if est() <= cap:
        return
    for msg in messages:
        if est() <= cap:
            return
        if msg.get("role") == "tool" and len(msg.get("content") or "") > 120:
            msg["content"] = (msg["content"][:120]
                               + '..."(shrunk further - hit per-request size limit)"')
    # If it's still over cap here, every tool message is already at the
    # 120-char floor - nothing more can be safely trimmed without
    # breaking message structure. The call proceeds and may still 413;
    # at minimum this makes that far less likely than before.


DEFAULT_TPM_FALLBACK = 6000

MODEL_FALLBACKS = {
    "loop": [
        os.getenv("GROQ_LOOP_MODEL", "openai/gpt-oss-20b"),
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-safeguard-20b",
    ],
    "synth": [
        os.getenv("GROQ_SYNTH_MODEL", "openai/gpt-oss-120b"),
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
    ],
}
for _role in MODEL_FALLBACKS:
    MODEL_FALLBACKS[_role] = list(dict.fromkeys(MODEL_FALLBACKS[_role]))

_model_role_lock = threading.Lock()
_current_model_index = {role: 0 for role in MODEL_FALLBACKS}


def _current_model(role):
    with _model_role_lock:
        idx = _current_model_index[role]
        return MODEL_FALLBACKS[role][idx]


def _rotate_model(role, exhausted_model, log=None):
    with _model_role_lock:
        idx = _current_model_index[role]
        chain = MODEL_FALLBACKS[role]
        if chain[idx] != exhausted_model:
            return chain[idx]
        if idx + 1 >= len(chain):
            return None
        _current_model_index[role] = idx + 1
        new_model = chain[idx + 1]
    if log:
        log(f"[model-rotation] {role}: {exhausted_model} daily quota exhausted -> switching to {new_model}")
    return new_model


_rate_limiters_lock = threading.Lock()


def groq_chat(role, messages, log=None, **kwargs):
    for _ in range(len(MODEL_FALLBACKS[role])):
        model = _current_model(role)

        if model not in _rate_limiters:
            with _rate_limiters_lock:
                if model not in _rate_limiters:
                    _rate_limiters[model] = TokenRateLimiter(
                        TPM_LIMITS.get(model, DEFAULT_TPM_FALLBACK)
                    )
        limiter = _rate_limiters[model]
        _shrink_messages_in_place(messages, kwargs.get("tools"), kwargs.get("max_tokens", 0))
        estimated = _estimate_tokens(messages, kwargs.get("max_tokens", 0), tools=kwargs.get("tools"))
        limiter.acquire(estimated)

        try:
            response = groq_client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as e:  # noqa: BLE001
            msg_lower = str(e).lower()
            is_daily_limit = "rate_limit_exceeded" in msg_lower and (
                "tokens per day" in msg_lower or "(tpd)" in msg_lower
            )
            if is_daily_limit:
                new_model = _rotate_model(role, model, log=log)
                if new_model is not None:
                    continue
            raise

        actual = getattr(getattr(response, "usage", None), "total_tokens", None)
        if actual:
            limiter.record_actual(estimated, actual)
        return response

    raise RuntimeError(f"All fallback models for role '{role}' exhausted their daily quota: "
                        f"{MODEL_FALLBACKS[role]}")


def call_with_retry(fn, *args, retries=4, rate_limit_retries=6, base_delay=1.5, **kwargs):
    last_err = None
    attempt = 0
    rate_limit_attempts = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg_lower = str(e).lower()

            retry_after_match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg_lower)
            is_rate_limit = "rate_limit_exceeded" in msg_lower or "429" in msg_lower
            transient = is_rate_limit or any(
                s in msg_lower for s in ["timeout", "timed out", "503", "502"]
            )

            if not transient:
                raise

            if is_rate_limit and retry_after_match:
                rate_limit_attempts += 1
                if rate_limit_attempts > rate_limit_retries:
                    raise
                minutes = float(retry_after_match.group(1) or 0)
                seconds = float(retry_after_match.group(2))
                delay = minutes * 60 + seconds + 2.0
            else:
                attempt += 1
                if attempt > retries:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)

            time.sleep(delay)


def compress_composio_result(result):
    data = result.data
    compact = {"toolkits": [], "primary_tools": [], "related_tools": []}
    results = data.get("results", [])
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            compact["toolkits"].extend(item.get("toolkits", []))
            compact["primary_tools"].extend(item.get("primary_tool_slugs", []))
            compact["related_tools"].extend(item.get("related_tool_slugs", []))
    for key in compact:
        compact[key] = list(dict.fromkeys(compact[key]))
    return compact


def compress_tool_schema_result(result):
    data = result.data
    compact = {"tool_schemas": {}}
    schemas = data.get("tool_schemas", {})
    if not isinstance(schemas, dict):
        return compact
    for slug, schema in schemas.items():
        if not isinstance(schema, dict):
            continue
        input_schema = schema.get("input_schema", {})
        properties, required = {}, []
        if isinstance(input_schema, dict):
            raw_properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            if isinstance(raw_properties, dict):
                for name, prop in raw_properties.items():
                    if not isinstance(prop, dict):
                        continue
                    properties[name] = {"type": prop.get("type", "unknown")}
                    if "enum" in prop:
                        enum_value = prop["enum"]
                        properties[name]["enum"] = (
                            enum_value[:10] if isinstance(enum_value, list) else enum_value
                        )
        compact["tool_schemas"][slug] = {
            "toolkit": schema.get("toolkit", ""),
            "tool_slug": schema.get("tool_slug", slug),
            "description": schema.get("description", "")[:300],
            "required": required,
            "parameters": properties,
        }
    return compact


def compress_web_result(search_result):
    compact = []
    for item in search_result.get("results", [])[:3]:
        compact.append({
            "title": item.get("title", "")[:150],
            "url": item.get("url", ""),
            "snippet": item.get("content", "")[:250],
        })
    return {"results": compact}


def build_system_prompt(app_name, category, hint):
    return f"""
You are an AI research agent researching ONE application.

The application is:

{app_name}

Category:

{category}

Starting hint (not ground truth, just a pointer):

{hint}

Your task is to determine:

- description
- authentication methods
- developer credential access
- pricing/access restrictions
- REST/GraphQL API
- API breadth
- MCP availability
- buildability
- main blocker
- evidence URLs


IMPORTANT RESEARCH WORKFLOW:

1. First use COMPOSIO_SEARCH_TOOLS to discover relevant
   tools for the application.

2. After discovering tools, use
   COMPOSIO_GET_TOOL_SCHEMAS to understand the useful
   tool parameters.

3. Use COMPOSIO_MULTI_EXECUTE_TOOL when an actual
   application tool can provide useful information.

4. If Composio cannot answer a research requirement,
   use browser.search.

5. Prefer official documentation, official developer
   documentation, official pricing pages, and official
   API documentation.

6. Do not connect personal accounts.

7. Do not use COMPOSIO_MANAGE_CONNECTIONS.

8. Do not invent facts.

9. If evidence is insufficient, use "unknown" or
   "unclear".

10. Do not repeat the same search unnecessarily.

11. The final answer must contain evidence URLs for
    important claims.

Buildability definitions:

ready:
Public API + obtainable credentials + sufficient API
surface for an integration.

constrained:
Public API exists but there are meaningful paid,
admin, permission, or access restrictions.

blocked:
No viable public API/integration path, or access is
partner-gated/unavailable.

unclear:
There is not enough evidence to determine this.


credential_access must be one of:

self_serve
trial
paid
admin
partner
contact_sales
unknown


api.breadth must be one of:

broad
moderate
limited
unknown


api.mcp must be one of:

yes
no
unknown


buildability must be one of:

ready
constrained
blocked
unclear


confidence must be one of:

high
medium
low


Do not connect a user's private application account.

During the research loop, use tools whenever useful.

When you believe enough information has been collected,
stop researching.
"""


FINAL_SYSTEM_PROMPT = """
You are the final report generator.

You are given research collected by another AI research
agent.

Convert the research into ONLY the following JSON object.

Do not call tools.

Do not write explanations.

Do not use markdown.

Do not use ```.

Return valid JSON only.


Required format:

{
  "app": "",
  "category": "",
  "description": "",
  "auth_methods": [],
  "credential_access": "",
  "api": {
    "types": [],
    "breadth": "",
    "mcp": ""
  },
  "buildability": "",
  "main_blocker": "",
  "evidence": [
    {
      "claim": "",
      "url": "",
      "source_type": ""
    }
  ],
  "confidence": ""
}


credential_access must be one of:

self_serve
trial
paid
admin
partner
contact_sales
unknown


api.breadth must be one of:

broad
moderate
limited
unknown


api.mcp must be one of:

yes
no
unknown


buildability must be one of:

ready
constrained
blocked
unclear


confidence must be one of:

high
medium
low


Do not invent evidence.

Use the URLs present in the research.

If something is not sufficiently supported,
use unknown or unclear.

Limit the evidence array to at most 5 of the most
important, most decision-relevant items. Do not include
every URL found - pick the ones that best support
credential_access, api.mcp, and buildability specifically.
"""

VERIFIER_SYSTEM_PROMPT = """
You are an independent fact-checker. You did NOT do the original
research. You are given two claims about an application, each with
its own small set of freshly retrieved web snippets. For EACH claim,
decide, using ONLY that claim's own snippets (not outside knowledge,
not the other claim's snippets), whether it is:

- "confirmed": the snippets clearly support the claim
- "contradicted": the snippets clearly contradict the claim
- "inconclusive": the snippets don't say enough either way

Return ONLY this JSON object, no markdown, no explanation outside it:

{
  "credential_access": {"verdict": "confirmed" | "contradicted" | "inconclusive", "reason": "one short sentence"},
  "api_mcp": {"verdict": "confirmed" | "contradicted" | "inconclusive", "reason": "one short sentence"}
}
"""


def create_assistant_message(message):
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ],
    }


def run_research_loop(app_name, category, hint, log):
    research_state = {
        "app": app_name,
        "category": category,
        "discovered_tools": {},
        "schemas": {},
        "execution_results": [],
        "web_evidence": [],
        "notes": [],
    }

    messages = [
        {"role": "system", "content": build_system_prompt(app_name, category, hint)},
        {
            "role": "user",
            "content": (
                f"Start researching {app_name}. Follow the required "
                "workflow and gather evidence for all required fields."
            ),
        },
    ]

    for step in range(MAX_STEPS):
        response = call_with_retry(
            groq_chat,
            role="loop",
            messages=messages,
            tools=GROQ_TOOLS,
            tool_choice="auto",
            max_tokens=MAX_TOKENS_LOOP,
            log=log,
        )
        record_usage(response, getattr(response, "model", "loop"))
        message = response.choices[0].message

        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            break

        messages.append(create_assistant_message(message))

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            log(f"[{app_name}] step {step + 1}: {tool_name} {json.dumps(arguments)[:200]}")

            tool_result = {}

            if tool_name in ("COMPOSIO_SEARCH_TOOLS", "COMPOSIO_GET_TOOL_SCHEMAS", "COMPOSIO_MULTI_EXECUTE_TOOL"):
                try:
                    result = call_with_retry(session.execute, tool_name, arguments=arguments)

                    if tool_name == "COMPOSIO_SEARCH_TOOLS":
                        compressed = compress_composio_result(result)
                        research_state["discovered_tools"] = compressed
                        tool_result = compressed

                    elif tool_name == "COMPOSIO_GET_TOOL_SCHEMAS":
                        compressed = compress_tool_schema_result(result)
                        research_state["schemas"] = compressed
                        tool_result = compressed

                    else:
                        result_data = result.data
                        if isinstance(result_data, dict):
                            compact_execution = {
                                k: (v[:1500] if isinstance(v, str) else v)
                                for k, v in result_data.items()
                            }
                        else:
                            compact_execution = str(result_data)[:2000]
                        research_state["execution_results"].append(compact_execution)
                        tool_result = compact_execution

                except Exception as e:  # noqa: BLE001
                    error_message = str(e)
                    research_state["notes"].append(f"{tool_name} error: {error_message}")
                    tool_result = {"error": error_message}

            elif tool_name == "browser.search":
                query = arguments.get("query", "")
                top_n = min(int(arguments.get("top_n", 3)), 3)
                try:
                    search_result = call_with_retry(
                        tavily_client.search, query=query, max_results=top_n
                    )
                    compressed_result = compress_web_result(search_result)
                    research_state["web_evidence"].extend(compressed_result["results"])
                    tool_result = compressed_result
                except Exception as e:  # noqa: BLE001
                    error_message = str(e)
                    research_state["notes"].append(f"Tavily error: {error_message}")
                    tool_result = {"error": error_message}

            else:
                tool_result = {"error": f"Unknown tool requested: {tool_name}"}

            # research_state keeps the FULL tool_result for the final
            # report. What gets echoed back into `messages` here is
            # capped hard, because THIS is what gets resent in full on
            # every subsequent step of the SAME app's loop - an
            # uncapped schema/execution dump is the main reason a
            # 4-step loop was observed using ~17.5K tokens/app. The
            # model only needs enough to decide its next tool call,
            # not the full data - cutting this roughly doubles how
            # many apps fit in the daily budget.
            MAX_TOOL_ECHO_CHARS = 500  # was 900 - lowered after a live 413
                                       # showed parallel tool calls in a
                                       # single step (e.g. 4 browser.search
                                       # calls at once) can stack several
                                       # of these before the next request,
                                       # crossing the per-request ITPM cap
                                       # even with each one individually
                                       # "capped". _shrink_messages_in_place
                                       # is the hard backstop; this just
                                       # reduces how often it needs to fire.
            echo_json = json.dumps(tool_result, default=str)
            if len(echo_json) > MAX_TOOL_ECHO_CHARS:
                echo_json = echo_json[:MAX_TOOL_ECHO_CHARS] + '..."(truncated for context - full data retained for the final report)"'

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": echo_json,
            })

    return research_state


def _strip_code_fences(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def generate_final_report(research_state):
    payload = json.dumps(research_state, indent=2, default=str)
    max_tokens = MAX_TOKENS_SYNTH

    for attempt in range(2):
        response = call_with_retry(
            groq_chat,
            role="synth",
            messages=[
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            max_tokens=max_tokens,
        )
        record_usage(response, getattr(response, "model", "synth"))
        choice = response.choices[0]
        final_text = choice.message.content
        truncated = getattr(choice, "finish_reason", None) == "length"
        cleaned = _strip_code_fences(final_text)
        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError as e:
            if attempt == 0:
                if truncated:
                    max_tokens = int(max_tokens * 1.6)
                else:
                    payload = (
                        payload
                        + "\n\nYour previous response was not valid JSON. "
                        + "Return ONLY the JSON object, nothing else."
                    )
                continue
            reason = "response was truncated (hit max_tokens)" if truncated else f"JSON parse error: {e}"
            return None, f"{reason} after retry. Raw text: {final_text[:500]}"
    return None, "unreachable"


def _search_for_claim(query):
    try:
        search_result = call_with_retry(tavily_client.search, query=query, max_results=3)
        return compress_web_result(search_result)["results"]
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}


def verify_report(app_name, report):
    if not report:
        return {"credential_access": None, "api_mcp": None}

    cred_claim = report.get("credential_access")
    mcp_claim = (report.get("api") or {}).get("mcp")

    fields = {}
    if cred_claim not in (None, "", "unknown"):
        fields["credential_access"] = {
            "claim": cred_claim,
            "evidence": _search_for_claim(f"{app_name} developer API credentials sign up free trial pricing"),
        }
    if mcp_claim not in (None, "", "unknown"):
        fields["api_mcp"] = {
            "claim": mcp_claim,
            "evidence": _search_for_claim(f"{app_name} MCP server model context protocol"),
        }

    result = {
        "credential_access": {"verdict": "inconclusive", "reason": "original claim was already unknown"},
        "api_mcp": {"verdict": "inconclusive", "reason": "original claim was already unknown"},
    }

    if not fields:
        return result

    for key, val in list(fields.items()):
        if isinstance(val["evidence"], dict) and "_error" in val["evidence"]:
            result[key] = {"verdict": "inconclusive", "reason": f"verification search failed: {val['evidence']['_error']}"}
            del fields[key]
        elif not val["evidence"]:
            result[key] = {"verdict": "inconclusive", "reason": "no verification search results"}
            del fields[key]

    if not fields:
        return result

    user_content = json.dumps({"app": app_name, "claims": fields}, indent=2)

    try:
        response = call_with_retry(
            groq_chat,
            role="synth",
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=MAX_TOKENS_VERIFY,
        )
        record_usage(response, getattr(response, "model", "synth"))
        raw = _strip_code_fences(response.choices[0].message.content)
        parsed = json.loads(raw)
        for key in fields:
            if key in parsed:
                parsed[key]["_evidence"] = fields[key]["evidence"]
                result[key] = parsed[key]
    except Exception as e:  # noqa: BLE001
        for key in fields:
            result[key] = {"verdict": "inconclusive", "reason": f"verifier call failed: {e}"}

    return result


def apply_verification(report, verification):
    if not report:
        return report, True

    verdicts = [
        verification.get("credential_access", {}).get("verdict"),
        verification.get("api_mcp", {}).get("verdict"),
    ]

    needs_human = False
    original_confidence = report.get("confidence", "unknown")
    new_confidence = original_confidence

    if "contradicted" in verdicts:
        new_confidence = "low"
        needs_human = True
    elif "inconclusive" in verdicts and original_confidence == "high":
        new_confidence = "medium"

    report["confidence"] = new_confidence
    report["verification"] = {
        "first_pass_confidence": original_confidence,
        "after_verification_confidence": new_confidence,
        "checks": verification,
    }
    return report, needs_human


def process_app(app_entry, log):
    app_name, category, hint = app_entry
    record = {
        "app": app_name,
        "category": category,
        "status": "ok",
        "report": None,
        "needs_human_review": False,
        "error": None,
    }

    if BUDGET_EXCEEDED.is_set():
        record["status"] = "skipped_budget"
        record["error"] = "Token budget reached before this app started; not researched."
        record["needs_human_review"] = True
        log(f"[{app_name}] skipped - token budget reached")
    else:
        try:
            research_state = run_research_loop(app_name, category, hint, log)
            report, parse_error = generate_final_report(research_state)

            if report is None:
                record["status"] = "parse_error"
                record["error"] = parse_error
                record["needs_human_review"] = True
                record["raw_research_state"] = research_state
            else:
                verification = verify_report(app_name, report)
                report, needs_human = apply_verification(report, verification)
                record["report"] = report

                is_empty = (
                    not report.get("evidence")
                    and not report.get("description")
                    and not report.get("auth_methods")
                )
                if is_empty:
                    record["needs_human_review"] = True
                    record["error"] = (record.get("error") or "") + \
                        " | Report is empty: model exited loop without calling any research tool"
                else:
                    record["needs_human_review"] = needs_human

        except Exception as e:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
            record["needs_human_review"] = True

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", app_name)
    out_path = os.path.join(RESULTS_DIR, f"{safe_name}.json")
    write_err = None
    for attempt in range(3):
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            write_err = None
            break
        except OSError as e:
            write_err = e
            time.sleep(0.5 * (attempt + 1))
    if write_err:
        record["status"] = "error"
        record["error"] = f"Result file write failed after retries: {write_err}"
        record["needs_human_review"] = True

    log(f"[{app_name}] done -> status={record['status']} "
        f"needs_human_review={record['needs_human_review']}")

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only research the first N apps (for testing)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--app", type=str, default=None, help="research a single app by exact name")
    parser.add_argument(
        "--apps-file", type=str, default=None,
        help="path to a text file with one app name per line (exact registry "
             "names, '#' comments and blank lines ignored) - runs exactly "
             "that subset, in that order.",
    )
    parser.add_argument(
        "--token-budget", type=int, default=None,
        help="stop starting NEW apps once this many total tokens have been used "
             "(in-flight apps still finish). Leave unset for no cap.",
    )
    args = parser.parse_args()

    global TOKEN_BUDGET
    TOKEN_BUDGET = args.token_budget

    apps_to_run = APPS
    if args.app:
        apps_to_run = [a for a in APPS if a[0].lower() == args.app.lower()]
        if not apps_to_run:
            raise SystemExit(f"App not found in registry: {args.app}")
    elif args.apps_file:
        with open(args.apps_file, "r", encoding="utf-8") as f:
            wanted_names = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        by_name = {a[0].lower(): a for a in APPS}
        apps_to_run, missing = [], []
        for name in wanted_names:
            entry = by_name.get(name.lower())
            (apps_to_run.append(entry) if entry else missing.append(name))
        if missing:
            raise SystemExit(f"{len(missing)} name(s) in {args.apps_file} don't match the registry exactly: {missing}")
        if not apps_to_run:
            raise SystemExit(f"No apps found from {args.apps_file}")
    elif args.limit:
        apps_to_run = APPS[: args.limit]

    def log(msg):
        print(msg, flush=True)

    all_results = []
    needs_review = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_app, app, log): app for app in apps_to_run}
        for future in as_completed(futures):
            app_entry = futures[future]
            try:
                record = future.result()
            except Exception as e:  # noqa: BLE001
                record = {
                    "app": app_entry[0],
                    "category": app_entry[1],
                    "status": "fatal_error",
                    "error": str(e),
                    "needs_human_review": True,
                    "report": None,
                }
            all_results.append(record)
            if record.get("needs_human_review"):
                needs_review.append({
                    "app": record["app"],
                    "category": record["category"],
                    "status": record["status"],
                    "reason": record.get("error") or "verification contradiction/inconclusive",
                })

    with open("all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open("needs_human_review.json", "w", encoding="utf-8") as f:
        json.dump(needs_review, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in all_results if r["status"] == "ok")
    skipped_budget = sum(1 for r in all_results if r["status"] == "skipped_budget")

    with open("run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "apps_requested": len(apps_to_run),
            "apps_completed_ok": ok,
            "apps_skipped_budget": skipped_budget,
            "apps_flagged_for_review": len(needs_review),
            "token_usage": TOKEN_USAGE,
            "token_budget": TOKEN_BUDGET,
        }, f, indent=2)

    print()
    print("=" * 60)
    print(f"BATCH DONE: {ok}/{len(all_results)} apps researched successfully")
    if skipped_budget:
        print(f"Skipped due to token budget: {skipped_budget}")
    print(f"Flagged for human review: {len(needs_review)}")
    print(f"Total tokens used: {TOKEN_USAGE['total']}  (by model: {TOKEN_USAGE['by_model']})")
    print("Results: all_results.json, needs_human_review.json, run_summary.json, results/<app>.json")
    print("=" * 60)


if __name__ == "__main__":
    main()