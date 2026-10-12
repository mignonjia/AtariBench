"""Render a run directory into a side-by-side visualization video."""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


@dataclass(frozen=True)
class ActionWindow:
    """Frame range for one planned action."""

    action_name: str
    start_frame_index: int
    end_frame_index: int
    reward_delta: float
    lost_life: bool


@dataclass(frozen=True)
class TurnWindow:
    """Loaded turn metadata for visualization."""

    turn_index: int
    start_frame_index: int
    executed_frame_end: int
    thought: str
    planned_action_strings: list[str]
    action_windows: list[ActionWindow]


def render_run_video(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    fps: int = 30,
) -> Path:
    """Create a whiteboard-style mp4 from a stored run directory."""

    run_path = Path(run_dir)
    frames_dir = run_path / "frames"
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    turns = _load_turns(run_path / "turns.jsonl")
    if output_path is None:
        output_path = run_path / "visualization.mp4"
    output_path = Path(output_path)

    rendered_frames_dir = run_path / "visualization_frames"
    rendered_frames_dir.mkdir(parents=True, exist_ok=True)

    first_frame = Image.open(frame_paths[0]).convert("RGB")
    frame_width, frame_height = first_frame.size
    panel_width = 560
    canvas_width = frame_width + panel_width + 48
    canvas_height = max(frame_height + 40, 360)

    for frame_path in frame_paths:
        frame_index = int(frame_path.stem.split("_")[-1])
        composed = _compose_frame(
            source_frame_path=frame_path,
            frame_index=frame_index,
            turns=turns,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            panel_width=panel_width,
            fps=fps,
        )
        composed.save(rendered_frames_dir / f"viz_{frame_index:06d}.png")

    _encode_video(rendered_frames_dir, output_path, fps=fps)
    return output_path


def _load_turns(turns_path: Path) -> list[TurnWindow]:
    turns: list[TurnWindow] = []
    for line in turns_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        action_windows = [
            ActionWindow(
                action_name=action["action_name"],
                start_frame_index=action["start_frame_index"],
                end_frame_index=action["end_frame_index"],
                reward_delta=action["reward_delta"],
                lost_life=action["lost_life"],
            )
            for action in payload.get("action_records", [])
        ]
        turns.append(
            TurnWindow(
                turn_index=payload["turn_index"],
                start_frame_index=payload["start_frame_index"],
                executed_frame_end=payload["executed_frame_end"],
                thought=payload.get("parsed_thought", ""),
                planned_action_strings=payload.get("planned_action_strings", []),
                action_windows=action_windows,
            )
        )
    return turns


def _compose_frame(
    source_frame_path: Path,
    frame_index: int,
    turns: list[TurnWindow],
    canvas_width: int,
    canvas_height: int,
    panel_width: int,
    fps: int,
) -> Image.Image:
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(17)
    body_font = _load_font(12)
    mono_font = _load_font(11)

    game_frame = Image.open(source_frame_path).convert("RGB")
    game_x = 20
    game_y = 20
    canvas.paste(game_frame, (game_x, game_y))
    draw.rectangle(
        [game_x - 1, game_y - 1, game_x + game_frame.width, game_y + game_frame.height],
        outline=(210, 210, 210),
        width=2,
    )

    panel_x = game_x + game_frame.width + 28
    panel_y = 20
    panel_right = panel_x + panel_width

    draw.text((game_x, canvas_height - 26), f"frame {frame_index:03d}  |  t={frame_index / fps:.2f}s", fill="black", font=mono_font)
    draw.text((panel_x, panel_y), "LLM Response and Actions", fill="black", font=title_font)

    current_turn = _find_turn_for_frame(turns, frame_index)
    if current_turn is None:
        _draw_wrapped_block(
            draw,
            text="No turn metadata available for this frame.",
            x=panel_x,
            y=panel_y + 30,
            width=panel_width,
            font=body_font,
            fill="black",
        )
        return canvas

    draw.text(
        (panel_x, panel_y + 30),
        f"Turn {current_turn.turn_index}  |  clip {current_turn.start_frame_index}-{current_turn.executed_frame_end}",
        fill=(60, 60, 60),
        font=mono_font,
    )
    thought_y = panel_y + 54
    thought_label = "Thought (excerpt)"
    draw.text((panel_x, thought_y), thought_label, fill="black", font=body_font)
    next_y = _draw_wrapped_block(
        draw,
        text=current_turn.thought or "(empty)",
        x=panel_x,
        y=thought_y + 18,
        width=panel_width,
        font=body_font,
        fill=(20, 20, 20),
        max_lines=8,
    )

    actions_y = next_y + 10
    draw.text((panel_x, actions_y), "Actions", fill="black", font=body_font)
    active_action_index = _find_active_action_index(current_turn, frame_index)
    status_y = actions_y + 20
    column_count = 4
    column_gap = 14
    column_width = (panel_width - (column_gap * (column_count - 1))) // column_count
    line_height = _line_height(mono_font) + 5
    for index, action_name in enumerate(current_turn.planned_action_strings):
        prefix = ">"
        fill = (180, 180, 180)
        if active_action_index is None:
            prefix = "-"
            fill = (90, 90, 90)
        elif index < active_action_index:
            prefix = "✓"
            fill = (120, 120, 120)
        elif index == active_action_index:
            prefix = ">"
            fill = (15, 85, 230)
        else:
            prefix = "-"
            fill = (90, 90, 90)
        action_text = f"{prefix} {index + 1:02d}. {action_name}"
        column_index = index % column_count
        row_index = index // column_count
        action_x = panel_x + column_index * (column_width + column_gap)
        action_y = status_y + row_index * line_height
        draw.text((action_x, action_y), action_text, fill=fill, font=mono_font)

    footer_y = canvas_height - 66
    active_label = "Current action: none"
    if active_action_index is not None:
        active_label = f"Current action: {current_turn.planned_action_strings[active_action_index]}"
    draw.text((panel_x, footer_y), active_label, fill="black", font=mono_font)
    draw.text(
        (panel_x, footer_y + 24),
        "Blue = active, gray = past/future",
        fill=(90, 90, 90),
        font=mono_font,
    )
    draw.line((panel_x - 12, 14, panel_x - 12, canvas_height - 14), fill=(225, 225, 225), width=2)
    draw.rectangle([panel_x - 14, 12, panel_right + 6, canvas_height - 12], outline=(235, 235, 235), width=2)
    return canvas


def _draw_wrapped_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    font: ImageFont.ImageFont,
    fill,
    max_lines: int | None = None,
) -> int:
    lines = []
    for paragraph in text.splitlines():
        clean = paragraph.strip()
        if not clean:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(clean, width=max(20, width // 8)))
    if not lines:
        lines = [""]

    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines[-1]:
            lines[-1] = _truncate_line(lines[-1])
        else:
            lines[-1] = "..."

    line_height = _line_height(font) + 4
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=fill, font=font)
        current_y += line_height
    return current_y


def _line_height(font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


def _truncate_line(text: str, max_chars: int = 72) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _find_turn_for_frame(turns: list[TurnWindow], frame_index: int) -> TurnWindow | None:
    for turn in turns:
        if turn.start_frame_index <= frame_index <= turn.executed_frame_end:
            return turn
    if turns and frame_index < turns[0].start_frame_index:
        return turns[0]
    if turns:
        return turns[-1]
    return None


def _find_active_action_index(turn: TurnWindow, frame_index: int) -> int | None:
    if not turn.action_windows:
        return None
    for index, action in enumerate(turn.action_windows):
        if action.start_frame_index < frame_index <= action.end_frame_index:
            return index
    if frame_index <= turn.action_windows[0].start_frame_index:
        return 0
    return len(turn.action_windows) - 1


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("Menlo.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _encode_video(rendered_frames_dir: Path, output_path: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(rendered_frames_dir / "viz_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a run directory into a video.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--fps", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = render_run_video(
        run_dir=args.run_dir,
        output_path=args.output_path,
        fps=args.fps,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
