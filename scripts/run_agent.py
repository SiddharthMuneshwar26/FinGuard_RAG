"""Run the FinGuard agent once from the command line, optionally with an image/audio file.

  python scripts/run_agent.py "What does SR 26-2 require for model inventories?"
  python scripts/run_agent.py "Explain this chart" --image chart.png        # Gemini only
  python scripts/run_agent.py --audio question.wav                          # Gemini only

With a text-only model (Ollama/OpenAI) attachments are replaced by a note.
Set FINGUARD_CLOUD_TRACE=1 to export the run to Cloud Trace / Cloud Logging.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from google.adk.runners import InMemoryRunner  # noqa: E402

from finguard_agent.agent import root_agent  # noqa: E402
from finguard_agent.multimodal import build_user_content  # noqa: E402


async def run(text: str, image: str | None, audio: str | None) -> None:
    runner = InMemoryRunner(agent=root_agent, app_name="finguard")
    session = await runner.session_service.create_session(app_name="finguard", user_id="local")
    message = build_user_content(text, image_path=image, audio_path=audio)
    async for event in runner.run_async(user_id="local", session_id=session.id, new_message=message):
        for call in event.get_function_calls():
            print(f"[tool] {call.name}({dict(call.args or {})})")
        if event.is_final_response() and event.content and event.content.parts:
            print("".join(part.text or "" for part in event.content.parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    args = parser.parse_args()
    asyncio.run(run(args.question, args.image, args.audio))


if __name__ == "__main__":
    main()
