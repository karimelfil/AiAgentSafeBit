import os
from typing import List

from app.schemas import ChatMessage


class LlmGenerationError(RuntimeError):
    pass


def _build_prompt(summary: str, memory: List[ChatMessage], question: str, use_memory: bool) -> str:
    sections = [
        "You are a health-aware food safety assistant.",
        "Use only the provided structured safety summary.",
        "Do not invent ingredients, diseases, restaurants, or scan results.",
        "If data is missing, say so clearly.",
        "Prefer deterministic findings, keep the answer concise, and mention uncertainty when confidence is low.",
        "Do not let prior memory override a fresh dish question.",
        "For questions like 'why is that dish safer?', explain the selected dish's actual ingredients, detected conflicts, or lack of conflicts.",
        "Avoid generic allergy-list responses unless no better evidence exists.",
        "",
        f"Structured safety summary:\n{summary}",
        "",
        f"User question:\n{question}",
    ]
    if use_memory:
        memory_lines = "\n".join(f"{m.role}: {m.content}" for m in memory[-4:]) or "No prior conversation."
        sections.extend(["", f"Conversation memory:\n{memory_lines}"])
    sections.extend([
        "",
        "Write a concise, plain-language explanation for the user.",
        "If there is a best restaurant, explain briefly why it ranked first.",
        "If there are risks, mention the top reasons only.",
    ])
    return "\n".join(sections)


async def generate_explanation(summary: str, memory: List[ChatMessage], question: str, use_memory: bool = False) -> str:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_API_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not api_key or not base_url or not model:
        raise LlmGenerationError("LLM_API_KEY, LLM_API_URL, and LLM_MODEL must be configured.")

    prompt = _build_prompt(summary, memory, question, use_memory)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer only from supplied safety data."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except ModuleNotFoundError as exc:
        raise LlmGenerationError("httpx is not installed. Install requirements to enable the LLM explanation layer.") from exc
    except Exception as exc:
        raise LlmGenerationError(f"LLM request failed: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise LlmGenerationError("LLM returned no choices.")

    message = choices[0].get("message", {})
    content = (message.get("content") or "").strip()
    if not content:
        raise LlmGenerationError("LLM returned an empty explanation.")
    return content
