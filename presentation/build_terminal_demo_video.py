"""Generate the narrated terminal-style demo video.

This script builds `SUBMISSION_2_FINAL_VIDEO.mp4` from a realistic animated
terminal sequence and `SUBMISSION_2_NARRATION.txt`. It uses local macOS `say`
for a neutral synthesized voice and imageio-ffmpeg for the final MP4.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parent
NARRATION = ROOT / "SUBMISSION_2_NARRATION.txt"
AUDIO = ROOT / "SUBMISSION_2_NARRATION.aiff"
OUT = ROOT / "SUBMISSION_2_FINAL_VIDEO.mp4"

W, H = 1920, 1080
FPS = 12


def font(size: int, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if mono:
        candidates += [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(38, bold=True)
FONT_SUB = font(24)
FONT_MONO = font(25, mono=True)
FONT_MONO_SMALL = font(21, mono=True)
FONT_BIG = font(48, bold=True)


EVENTS = [
    (4.0, "local", "$ sw_vers | grep ProductVersion"),
    (6.0, "local", "ProductVersion: 15.5"),
    (8.0, "local", "$ python generate_submission_2_families.py --count-per-family 200"),
    (11.0, "local", "scfam01: generated 200 valid sequences"),
    (13.0, "local", "scfam12: generated 200 valid sequences"),
    (15.0, "local", "Wrote manifest for 15 total families -> training_data/SUBMISSION_2_15_FAMILY_MANIFEST.csv"),
    (18.0, "local", "$ python train.py --model gpt:large --submission-2-ood --epochs 30 --out outputs/gpt_large_train12"),
    (22.0, "local", "device: mps | Apple M4 Pro"),
    (25.0, "local", "model: gpt:large | params: 21.47M | tokenizer: one process step = one token"),
    (29.0, "local", "epoch 01/30  loss=3.808  ppl=45.07"),
    (32.0, "local", "epoch 10/30  loss=0.459  ppl=1.58"),
    (35.0, "local", "epoch 20/30  loss=0.442  ppl=1.56"),
    (38.0, "local", "epoch 30/30  loss=0.434  ppl=1.54"),
    (42.0, "leo", "$ ssh leonardo"),
    (44.0, "leo", "login01.leonardo.cineca.it"),
    (46.0, "leo", "$ sbatch job.slurm"),
    (48.0, "leo", "Submitted batch job 43137065"),
    (51.0, "leo", "node: lrdn0058 | gpu: NVIDIA A100-SXM-64GB | cuda: available"),
    (55.0, "leo", "$ python eval_runner.py --official-names --out final_submission/SUBMISSION_2_gpt_large_official"),
    (59.0, "leo", "nextstep.csv: 600 rows | completion.csv: 600 rows | anomaly.csv: 987 rows"),
    (63.0, "eval", "$ python eval_guided.py --kind gpt --size large --submission-2-ood --eval-seqs 500"),
    (67.0, "eval", "OOD split: train 12 families, test scfam10 scfam11 scfam12"),
    (71.0, "eval", "n-gram baseline: Top-1=0.662  Top-5=0.976  MRR=0.799"),
    (75.0, "eval", "GPT large:       Top-1=0.722  Top-5=0.999  MRR=0.847"),
    (79.0, "eval", "$ python score_selfeval.py --pred-dir self_eval/gpt_large_submission"),
    (83.0, "eval", "Task 1: Top-1=0.661  Top-5=0.999  MRR=0.824"),
    (87.0, "eval", "Task 2: NED=0.263  token_acc=0.387  block_acc=0.605"),
    (91.0, "eval", "Task 3: F1=1.000  ROC-AUC=1.000  rule_attribution=1.000"),
    (96.0, "files", "$ ls"),
    (98.0, "files", "README.md  REPORT.md  LICENSE  requirements.txt"),
    (100.0, "files", "nextstep.csv  completion.csv  anomaly.csv"),
    (102.0, "files", "outputs/gpt_large_train12/model.safetensors"),
    (104.0, "files", "training_artifacts/loss_curve.csv  self_eval/score_reports/"),
    (106.0, "files", "presentation/SUBMISSION_2_PITCH_DECK.pdf  presentation/SUBMISSION_2_FINAL_VIDEO.mp4"),
]

METRICS = [
    ("OOD Top-1", "0.722", "train 12 / test 3"),
    ("OOD Top-5", "0.999", "held-out families"),
    ("Task 2 NED", "0.263", "lower is better"),
    ("Task 3 F1", "1.00", "explicit rules"),
]


def synthesize_audio() -> float:
    subprocess.run(
        ["say", "-v", "Samantha", "-r", "165", "-f", str(NARRATION), "-o", str(AUDIO)],
        check=True,
    )
    info = subprocess.run(["afinfo", str(AUDIO)], check=True, capture_output=True, text=True)
    match = re.search(r"estimated duration: ([0-9.]+) sec", info.stdout)
    if not match:
        raise RuntimeError("Could not read synthesized narration duration")
    return float(match.group(1))


def gradient_bg() -> Image.Image:
    img = Image.new("RGB", (W, H), "#0b1020")
    pix = img.load()
    for y in range(H):
        for x in range(W):
            dx = x / W
            dy = y / H
            r = int(8 + 18 * dx + 8 * dy)
            g = int(13 + 28 * dy)
            b = int(32 + 30 * dx + 20 * (1 - dy))
            pix[x, y] = (r, g, b)
    return img


BG = gradient_bg()


def draw_round_rect(draw: ImageDraw.ImageDraw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def terminal_lines(t: float) -> list[tuple[str, str]]:
    shown = [(kind, text) for ts, kind, text in EVENTS if ts <= t]
    return shown[-16:]


def phase_label(t: float) -> str:
    if t < 40:
        return "Apple M4 Pro / local MPS run"
    if t < 62:
        return "Leonardo A100 / CUDA run"
    if t < 94:
        return "OOD and official-scorer self-eval"
    return "Final submission package"


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_w: int, fnt) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_terminal(draw: ImageDraw.ImageDraw, t: float) -> None:
    x, y, w, h = 130, 150, 1660, 720
    draw_round_rect(draw, (x + 16, y + 22, x + w + 16, y + h + 22), 26, "#000000")
    draw_round_rect(draw, (x, y, x + w, y + h), 26, "#111827", "#334155", 2)
    draw_round_rect(draw, (x, y, x + w, y + 58), 26, "#1f2937", "#1f2937")
    draw.rectangle((x, y + 30, x + w, y + 58), fill="#1f2937")
    for i, color in enumerate(["#ff5f57", "#ffbd2e", "#28c840"]):
        draw.ellipse((x + 28 + i * 34, y + 20, x + 48 + i * 34, y + 40), fill=color)
    draw.text((x + 685, y + 17), "kremsians@zero-one: ~/zero_one_hack_0625", font=FONT_MONO_SMALL, fill="#cbd5e1")
    draw.text((x + 36, y + 82), phase_label(t), font=FONT_MONO_SMALL, fill="#60a5fa")

    yy = y + 128
    for kind, line in terminal_lines(t):
        color = {
            "local": "#d1fae5",
            "leo": "#bfdbfe",
            "eval": "#fde68a",
            "files": "#e9d5ff",
        }[kind]
        prefix = {
            "local": "macbook",
            "leo": "leonardo",
            "eval": "score",
            "files": "repo",
        }[kind]
        prompt_color = "#34d399" if line.startswith("$") else "#64748b"
        if line.startswith("$"):
            draw.text((x + 42, yy), f"{prefix} ", font=FONT_MONO, fill=prompt_color)
            draw.text((x + 170, yy), line, font=FONT_MONO, fill="#f8fafc")
        else:
            for wrapped in wrap_text(draw, line, 1470, FONT_MONO):
                draw.text((x + 82, yy), wrapped, font=FONT_MONO, fill=color)
                yy += 33
            continue
        yy += 38

    if int(t * 2) % 2 == 0:
        draw.rectangle((x + 82, y + h - 54, x + 98, y + h - 28), fill="#f8fafc")


def draw_header(draw: ImageDraw.ImageDraw) -> None:
    draw.text((130, 72), "Team Kremsians", font=FONT_TITLE, fill="#f8fafc")
    draw.text((455, 81), "Industrial AI Infineon", font=FONT_SUB, fill="#93c5fd")
    draw.text((130, 910), "From-scratch step-token GPT + deterministic process-rule validator", font=FONT_SUB, fill="#cbd5e1")


def draw_result_overlay(draw: ImageDraw.ImageDraw, alpha: float) -> Image.Image:
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    a = int(235 * alpha)
    od.rounded_rectangle((210, 170, 1710, 850), radius=36, fill=(246, 248, 251, a), outline=(226, 232, 240, a), width=2)
    od.text((290, 230), "Final result package", font=FONT_BIG, fill=(15, 23, 42, int(255 * alpha)))
    od.text((292, 298), "Reproducible on Apple M4 Pro and Leonardo A100", font=FONT_SUB, fill=(71, 85, 105, int(255 * alpha)))
    for i, (label, value, note) in enumerate(METRICS):
        bx = 292 + i * 335
        by = 388
        od.rounded_rectangle((bx, by, bx + 275, by + 150), radius=18, fill=(255, 255, 255, a), outline=(203, 213, 225, a), width=2)
        od.text((bx + 28, by + 28), value, font=FONT_BIG, fill=(37, 99, 235, int(255 * alpha)))
        od.text((bx + 28, by + 88), label, font=FONT_SUB, fill=(15, 23, 42, int(255 * alpha)))
        od.text((bx + 28, by + 118), note, font=FONT_MONO_SMALL, fill=(100, 116, 139, int(255 * alpha)))
    checks = [
        "nextstep.csv, completion.csv, anomaly.csv",
        "checkpoint, loss curve, training log",
        "official scorer self-eval reports",
        "pitch deck and under-2-minute demo video",
    ]
    for i, text in enumerate(checks):
        yy = 625 + i * 44
        od.rounded_rectangle((300, yy + 2, 326, yy + 28), radius=6, fill=(16, 185, 129, int(255 * alpha)))
        od.text((350, yy), text, font=FONT_SUB, fill=(15, 23, 42, int(255 * alpha)))
    return overlay


def render_frame(t: float, duration: float) -> Image.Image:
    img = BG.copy()
    draw = ImageDraw.Draw(img)
    draw_header(draw)
    draw_terminal(draw, t)

    fade_in = max(0.0, min(1.0, (t - (duration - 20.0)) / 5.0))
    if fade_in > 0:
        overlay = draw_result_overlay(draw, fade_in)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    fade_out = max(0.0, min(1.0, (t - (duration - 2.0)) / 2.0))
    if fade_out > 0:
        black = Image.new("RGB", (W, H), "black")
        img = Image.blend(img, black, fade_out)
    return img


def main() -> None:
    duration = synthesize_audio()
    video_duration = min(119.0, max(duration + 1.0, 105.0))
    frames = math.ceil(video_duration * FPS)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-i",
        str(AUDIO),
        "-vf",
        "fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(OUT),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(frames):
        t = i / FPS
        proc.stdin.write(render_frame(t, video_duration).tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise SystemExit(ret)
    print(f"Wrote {OUT} ({video_duration:.1f}s video, {duration:.1f}s narration)")


if __name__ == "__main__":
    main()
