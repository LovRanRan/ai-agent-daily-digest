"""
Daily AI Agent Digest Generator
Calls Claude API with web search to produce a daily markdown digest.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic

# Use Pacific Time for date consistency (matches schedule)
TODAY = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

PROMPT = f"""You are my daily intelligence officer for the AI Agent domain. Today is {TODAY}.

Use web search to scan the last 24 hours and produce a structured markdown digest covering the three sections below. Be selective — prioritize signal over volume.

## 1. Open Source Projects / GitHub Trending
Find trending repos from the last 24 hours related to AI agents, LLM agents, multi-agent systems, MCP, LangGraph, or RAG. Focus on repos with significant star growth, new releases, or real engineering value. For each, include: name + link, one-line description, and why it matters to an agent engineer. Limit to 3-5 high-signal picks. Do not pad.

## 2. Industry News & Big Tech Releases
Official announcements or significant updates from the last 24 hours from Anthropic, OpenAI, Google DeepMind, Meta AI, Microsoft, or Amazon, related to agents. Include new models, new APIs, new product features, and noteworthy blog posts. For each: company, title + link, and the impact point for an agent engineer.

## 3. Engineering Practice / Technical Blogs
Blog posts worth reading from the last 24 hours. Prioritize sources: LangChain Blog, Anthropic Engineering, OpenAI Cookbook, major company engineering blogs, well-known individual engineer blogs, and top Hacker News threads on agents. Selection criteria: posts with code, architecture diagrams, or real-world war stories. 3-5 entries, each with link + core takeaway (1-2 sentences).

## Deep Dive of the Day
From the content above, pick one technical point most valuable for someone preparing for a FAANG AI agent engineer interview. In 3-5 sentences, explain why it matters and what directions are worth digging deeper into.

Formatting requirements:
- All output in English
- Valid markdown, ready to render
- No emoji
- Separate each section with ---
- Start your response with a level-1 heading: # AI Agent Daily Digest - {TODAY}

Return ONLY the markdown content. Do not include any preamble, explanation, or closing remarks outside the digest itself.
"""


def generate_digest() -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": PROMPT}],
    )

    # Collect all text blocks from the response
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    markdown = "\n".join(parts).strip()

    if not markdown:
        raise RuntimeError("Claude returned empty content")

    return markdown


def main() -> None:
    digest = generate_digest()

    digests_dir = Path("digests")
    digests_dir.mkdir(exist_ok=True)

    output_path = digests_dir / f"{TODAY}.md"
    output_path.write_text(digest, encoding="utf-8")

    print(f"Digest written to: {output_path}")
    print(f"Length: {len(digest)} chars")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
