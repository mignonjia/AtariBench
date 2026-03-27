"""Trajectory persistence for Atari Gemini runs."""

from __future__ import annotations

import dataclasses
import datetime as dt
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

try:
    from ..games.env import extract_env_info
    from .clip import ParsedClipResponse
except ImportError:  # Running from inside the AtariBench folder.
    from games.env import extract_env_info
    from core.clip import ParsedClipResponse

FrameWriter = Callable[[Any, Path], None]


@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """One saved frame with environment metadata."""

    local_frame_index: int
    reward: float
    lives: int | None
    episode_frame_number: int | None
    frame_number: int | None
    frame_path: str


@dataclasses.dataclass(frozen=True)
class ActionRecord:
    """One high-level action expanded into one or more env frames."""

    action_name: str
    action_id: int
    start_frame_index: int
    end_frame_index: int
    reward_delta: float
    lost_life: bool
    end_frame_path: str
    end_info: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """Persistent record for one model turn."""

    turn_index: int
    prompt_path: str
    prompt_html_path: str
    response_path: str
    prompt_text: str
    raw_response: str
    parsed_thought: str
    planned_action_strings: list[str]
    planned_action_ids: list[int]
    parse_errors: list[str]
    referenced_image_paths: list[str]
    start_frame_index: int
    start_frame_path: str
    executed_frame_end: int
    reward_delta: float
    action_records: list[ActionRecord]
    new_game_started: bool


class Trajectory:
    """Append-only run storage with prompt/response artifacts."""

    def __init__(
        self,
        base_output_dir: str | Path,
        game_key: str,
        frame_writer: FrameWriter | None = None,
        include_game_key: bool = True,
    ):
        base_dir = Path(base_output_dir)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if include_game_key:
            self.run_dir = base_dir / game_key / timestamp
        else:
            self.run_dir = base_dir / timestamp
        self.frames_dir = self.run_dir / "frames"
        self.prompts_dir = self.run_dir / "prompts"
        self.responses_dir = self.run_dir / "responses"
        self.turns_path = self.run_dir / "turns.jsonl"
        self.summary_path = self.run_dir / "summary.json"

        for directory in (
            self.frames_dir,
            self.prompts_dir,
            self.responses_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.frame_writer = frame_writer or _default_frame_writer
        self.frame_records: list[FrameRecord] = []
        self.turn_records: list[TurnRecord] = []

    def record_frame(
        self,
        frame: Any,
        reward: float,
        info: dict[str, Any],
        local_frame_index: int,
    ) -> FrameRecord:
        """Persist one frame and metadata."""

        frame_path = self.frames_dir / f"frame_{local_frame_index:06d}.png"
        self.frame_writer(frame, frame_path)
        normalized = extract_env_info(info)
        record = FrameRecord(
            local_frame_index=local_frame_index,
            reward=float(reward),
            lives=normalized.lives,
            episode_frame_number=normalized.episode_frame_number,
            frame_number=normalized.frame_number,
            frame_path=str(frame_path),
        )
        self.frame_records.append(record)
        return record

    def latest_frame(self) -> FrameRecord:
        if not self.frame_records:
            raise RuntimeError("No frames recorded yet.")
        return self.frame_records[-1]

    def record_turn(
        self,
        prompt_text: str,
        raw_response: str,
        parsed_response: ParsedClipResponse,
        referenced_image_paths: list[str],
        start_frame_index: int,
        start_frame_path: str,
        executed_frame_end: int,
        reward_delta: float,
        action_records: list[ActionRecord],
        new_game_started: bool = False,
    ) -> TurnRecord:
        """Persist a model turn and append it to the trajectory."""

        turn_index = len(self.turn_records) + 1
        prompt_path = self.prompts_dir / f"turn_{turn_index:04d}.txt"
        prompt_html_path = self.prompts_dir / f"turn_{turn_index:04d}.html"
        response_path = self.responses_dir / f"turn_{turn_index:04d}.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        prompt_html_path.write_text(
            _render_prompt_html(
                prompt_text=prompt_text,
                referenced_image_paths=referenced_image_paths,
                html_path=prompt_html_path,
            ),
            encoding="utf-8",
        )
        response_path.write_text(raw_response, encoding="utf-8")

        record = TurnRecord(
            turn_index=turn_index,
            prompt_path=str(prompt_path),
            prompt_html_path=str(prompt_html_path),
            response_path=str(response_path),
            prompt_text=prompt_text,
            raw_response=raw_response,
            parsed_thought=parsed_response.thought,
            planned_action_strings=list(parsed_response.action_strings),
            planned_action_ids=list(parsed_response.action_ids),
            parse_errors=list(parsed_response.errors),
            referenced_image_paths=list(referenced_image_paths),
            start_frame_index=start_frame_index,
            start_frame_path=start_frame_path,
            executed_frame_end=executed_frame_end,
            reward_delta=float(reward_delta),
            action_records=list(action_records),
            new_game_started=bool(new_game_started),
        )
        self.turn_records.append(record)
        with self.turns_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_turn_to_dict(record)) + "\n")
        return record

    def finalize(
        self,
        stop_reason: str,
        total_reward: float,
        total_lost_lives: int,
        duration_seconds: int | None = None,
        model_name: str | None = None,
        thinking_mode: str | None = None,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        history_clips: int | None = None,
        non_zero_reward_clips: int | None = None,
        prompt_mode: str | None = None,
    ) -> dict[str, Any]:
        """Write and return the run summary."""

        summary = {
            "run_dir": str(self.run_dir),
            "stop_reason": stop_reason,
            "total_reward": float(total_reward),
            "total_lost_lives": int(total_lost_lives),
            "duration_seconds": duration_seconds,
            "frame_count": len(self.frame_records),
            "turn_count": len(self.turn_records),
            "model_name": model_name,
            "thinking_mode": thinking_mode,
            "thinking_budget": thinking_budget,
            "thinking_level": thinking_level,
            "history_clips": history_clips,
            "non_zero_reward_clips": non_zero_reward_clips,
            "prompt_mode": prompt_mode,
            "last_frame": dataclasses.asdict(self.frame_records[-1])
            if self.frame_records
            else None,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def replace_last_turn(self, turn: TurnRecord) -> None:
        """Update the most recent turn record and rewrite turns.jsonl."""

        if not self.turn_records:
            raise RuntimeError("No turns recorded yet.")
        self.turn_records[-1] = turn
        with self.turns_path.open("w", encoding="utf-8") as handle:
            for existing_turn in self.turn_records:
                handle.write(json.dumps(_turn_to_dict(existing_turn)) + "\n")


def _turn_to_dict(turn: TurnRecord) -> dict[str, Any]:
    payload = dataclasses.asdict(turn)
    return payload


def _default_frame_writer(frame: Any, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save trajectory frames.") from exc

    image = Image.fromarray(frame)
    image.save(path)


def _render_prompt_html(
    prompt_text: str,
    referenced_image_paths: list[str],
    html_path: Path,
) -> str:
    if _looks_like_chat_transcript(prompt_text):
        return _render_chat_prompt_html(
            prompt_text=prompt_text,
            referenced_image_paths=referenced_image_paths,
            html_path=html_path,
        )

    placeholder = "IMG_HOLDER"
    escaped_prompt = html.escape(prompt_text)
    parts = escaped_prompt.split(placeholder)

    rendered: list[str] = []
    for index, part in enumerate(parts):
        rendered.append(f"<pre>{part}</pre>")
        if index >= len(parts) - 1:
            continue

        image_path = referenced_image_paths[index] if index < len(referenced_image_paths) else None
        if image_path is None:
            rendered.append(
                '<div class="img-missing">IMG_HOLDER has no mapped image.</div>'
            )
            continue

        relative_image_path = os.path.relpath(image_path, start=html_path.parent)
        escaped_image_path = html.escape(relative_image_path)
        rendered.append(
            (
                '<figure class="img-block">'
                f'<div class="img-label">IMG_HOLDER #{index + 1}</div>'
                f'<img src="{escaped_image_path}" alt="IMG_HOLDER #{index + 1}" />'
                f'<figcaption>{escaped_image_path}</figcaption>'
                "</figure>"
            )
        )

    if len(referenced_image_paths) > len(parts) - 1:
        extras = referenced_image_paths[len(parts) - 1 :]
        extra_blocks = []
        for extra_index, image_path in enumerate(extras, start=len(parts)):
            relative_image_path = os.path.relpath(image_path, start=html_path.parent)
            escaped_image_path = html.escape(relative_image_path)
            extra_blocks.append(
                (
                    '<figure class="img-block">'
                    f'<div class="img-label">Unmapped Image #{extra_index}</div>'
                    f'<img src="{escaped_image_path}" alt="Unmapped Image #{extra_index}" />'
                    f'<figcaption>{escaped_image_path}</figcaption>'
                    "</figure>"
                )
            )
        rendered.append("<h2>Extra Referenced Images</h2>")
        rendered.extend(extra_blocks)

    body = "\n".join(rendered)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Prompt Render</title>
  <style>
    body {{
      background: #f7f7f5;
      color: #111;
      font-family: Menlo, Monaco, Consolas, monospace;
      line-height: 1.45;
      margin: 0;
      padding: 24px;
    }}
    pre {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin: 0 0 12px 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .img-block {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin: 0 0 16px 0;
      padding: 12px;
    }}
    .img-label {{
      color: #666;
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .img-missing {{
      background: #fff4f4;
      border: 1px solid #f0b8b8;
      border-radius: 8px;
      color: #900;
      margin: 0 0 16px 0;
      padding: 12px;
    }}
    img {{
      border: 1px solid #ddd;
      display: block;
      image-rendering: pixelated;
      margin-bottom: 8px;
      max-width: min(720px, 100%);
    }}
    figcaption {{
      color: #555;
      font-size: 12px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <h1>Prompt Render</h1>
  {body}
</body>
</html>
"""


def _looks_like_chat_transcript(prompt_text: str) -> bool:
    return "<user>" in prompt_text or "<assistant>" in prompt_text


def _render_chat_prompt_html(
    prompt_text: str,
    referenced_image_paths: list[str],
    html_path: Path,
) -> str:
    image_index = 0
    bubbles: list[str] = []

    for role, body in _iter_chat_blocks(prompt_text):
        bubble_html, image_index = _render_chat_bubble_html(
            role=role,
            body=body,
            referenced_image_paths=referenced_image_paths,
            html_path=html_path,
            image_index=image_index,
        )
        bubbles.append(bubble_html)

    if image_index < len(referenced_image_paths):
        bubbles.append("<h2>Extra Referenced Images</h2>")
        for extra_index, image_path in enumerate(referenced_image_paths[image_index:], start=image_index + 1):
            relative_image_path = os.path.relpath(image_path, start=html_path.parent)
            escaped_image_path = html.escape(relative_image_path)
            bubbles.append(
                (
                    '<figure class="img-block extra-image">'
                    f'<div class="img-label">Unmapped Image #{extra_index}</div>'
                    f'<img src="{escaped_image_path}" alt="Unmapped Image #{extra_index}" />'
                    f'<figcaption>{escaped_image_path}</figcaption>'
                    "</figure>"
                )
            )

    body = "\n".join(bubbles)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Prompt Render</title>
  <style>
    body {{
      background: #f4f1ea;
      color: #111;
      font-family: Menlo, Monaco, Consolas, monospace;
      line-height: 1.35;
      margin: 0;
      padding: 16px;
    }}
    h1 {{
      font-size: 16px;
      margin: 0 0 12px 0;
    }}
    .chat {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .row {{
      display: flex;
      width: 100%;
    }}
    .row.user {{
      justify-content: flex-start;
    }}
    .row.assistant {{
      justify-content: flex-end;
    }}
    .bubble {{
      border: 1px solid #d8d1c4;
      border-radius: 12px;
      max-width: min(780px, 92%);
      padding: 12px 14px;
    }}
    .row.user .bubble {{
      background: #fffaf0;
    }}
    .row.assistant .bubble {{
      background: #eef5ff;
    }}
    pre {{
      margin: 0 0 10px 0;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .img-block {{
      background: rgba(255,255,255,0.8);
      border: 1px solid #ddd;
      border-radius: 8px;
      margin: 10px 0 0 0;
      padding: 10px;
    }}
    .img-label {{
      color: #666;
      font-size: 11px;
      margin-bottom: 6px;
    }}
    .img-missing {{
      background: #fff4f4;
      border: 1px solid #f0b8b8;
      border-radius: 8px;
      color: #900;
      margin-top: 10px;
      padding: 10px;
    }}
    img {{
      border: 1px solid #ddd;
      display: block;
      image-rendering: pixelated;
      margin-bottom: 8px;
      max-width: min(680px, 100%);
    }}
    figcaption {{
      color: #555;
      font-size: 11px;
      word-break: break-all;
    }}
    .extra-image {{
      background: #fff;
      margin-top: 12px;
    }}
  </style>
</head>
<body>
  <h1>Prompt Render</h1>
  <div class="chat">
    {body}
  </div>
</body>
</html>
"""


def _iter_chat_blocks(prompt_text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"<(user|assistant)>\s*(.*?)\s*</\1>", re.DOTALL)
    return [(match.group(1), match.group(2)) for match in pattern.finditer(prompt_text)]


def _render_chat_bubble_html(
    role: str,
    body: str,
    referenced_image_paths: list[str],
    html_path: Path,
    image_index: int,
) -> tuple[str, int]:
    placeholder = "IMG_HOLDER"
    parts = body.split(placeholder)
    rendered: list[str] = []

    for index, part in enumerate(parts):
        escaped_part = html.escape(part)
        if escaped_part.strip():
            rendered.append(f"<pre>{escaped_part}</pre>")
        if index >= len(parts) - 1:
            continue

        image_path = referenced_image_paths[image_index] if image_index < len(referenced_image_paths) else None
        image_index += 1
        if image_path is None:
            rendered.append('<div class="img-missing">IMG_HOLDER has no mapped image.</div>')
            continue

        relative_image_path = os.path.relpath(image_path, start=html_path.parent)
        escaped_image_path = html.escape(relative_image_path)
        rendered.append(
            (
                '<figure class="img-block">'
                f'<div class="img-label">Image #{image_index}</div>'
                f'<img src="{escaped_image_path}" alt="Image #{image_index}" />'
                f'<figcaption>{escaped_image_path}</figcaption>'
                "</figure>"
            )
        )

    bubble = "\n".join(rendered)
    return (
        f'<div class="row {html.escape(role)}"><div class="bubble">{bubble}</div></div>',
        image_index,
    )
