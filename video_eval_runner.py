"""One-click evaluation runner for panorama videos (.mp4).

This script evaluates *sampled frames* from videos using the existing image metrics in this folder.
It is not a temporal video metric (no motion/consistency metrics).

Supported metrics (default):
- Discontinuity Score (DS) on generated frames
- CLIPScore on generated frames (+ prompts)
- Inception Score (IS) on generated frames
- FID/KID between real vs generated sampled frames

Optional metrics:
- OmniFID: requires cubemap face directories (U/F/R/B/L/D). This runner can generate them
  only if `convert360` is installed and available in PATH (see e2c_converter.py).
- FAED: requires FAED checkpoint + external modules used by FAED_cal.py.

Windows notes:
- Requires `ffmpeg` available in PATH for frame extraction.

Sampling:
- Default/recommended: `--fps` (e.g. `--fps 1`)
- Alternative for long videos: `--num_frames_per_video N`

Example (PowerShell):
    python video_eval_runner.py --real_dir D:\\data\\gt_videos --fake_dir D:\\data\\gen_videos --out_dir D:\\data\\_eval_cache --fps 1 --prompt "a 360 panorama of a living room"

"""

import argparse
import importlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as torch_f
from torchvision import transforms
from torchvision.io import read_image


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found in PATH. Please install ffmpeg and make sure `ffmpeg` is callable from PowerShell."
        )
    return ffmpeg


def _require_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError(
            "ffprobe not found in PATH. It usually ships with ffmpeg. Please ensure `ffprobe` is callable."
        )
    return ffprobe


def _run(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nstdout:\n"
            + proc.stdout
            + "\n\nstderr:\n"
            + proc.stderr
        )


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    fps: float,
    max_frames: Optional[int],
) -> List[Path]:
    """Extract frames from a video using ffmpeg.

    Output naming: 000001.png ...

    Args:
        video_path: .mp4 path
        output_dir: directory to write frames
        fps: sampling fps
        max_frames: if provided, cap extracted frames by selecting first N frames

    Returns:
        List of extracted frame paths (sorted).
    """

    ffmpeg = _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract at a fixed FPS.
    # -hide_banner/-loglevel error keeps output clean.
    out_pattern = str(output_dir / "%06d.png")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-start_number",
        "1",
        out_pattern,
    ]
    _run(cmd)

    frames = sorted(output_dir.glob("*.png"))
    if max_frames is not None:
        frames = frames[: max_frames]
        # If we capped, delete extras to keep cache consistent.
        extras = sorted(output_dir.glob("*.png"))[max_frames:]
        for p in extras:
            try:
                p.unlink()
            except OSError:
                pass

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frames


def extract_frames_all(
    video_path: Path,
    output_dir: Path,
    max_frames: Optional[int],
) -> List[Path]:
    """Extract all frames from a video.

    This is the most faithful option when you want to evaluate every frame.
    """

    ffmpeg = _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)

    out_pattern = str(output_dir / "%06d.png")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-start_number",
        "1",
        out_pattern,
    ]
    _run(cmd)

    frames = sorted(output_dir.glob("*.png"))
    if max_frames is not None:
        frames = frames[: max_frames]
        extras = sorted(output_dir.glob("*.png"))[max_frames:]
        for p in extras:
            try:
                p.unlink()
            except OSError:
                pass

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frames


def extract_frames_every_k(
    video_path: Path,
    output_dir: Path,
    k: int,
    max_frames: Optional[int],
) -> List[Path]:
    """Extract every k-th frame from a video.

    This samples by decoded frame index n: 0, k, 2k, ...
    """

    if k <= 0:
        raise ValueError("k must be > 0")

    ffmpeg = _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg select filter on frame index n.
    # Note: the comma inside mod() must be escaped for ffmpeg filter syntax.
    vf = f"select=not(mod(n\\,{k}))"

    out_pattern = str(output_dir / "%06d.png")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-start_number",
        "1",
        out_pattern,
    ]
    _run(cmd)

    frames = sorted(output_dir.glob("*.png"))
    if max_frames is not None:
        frames = frames[: max_frames]
        extras = sorted(output_dir.glob("*.png"))[max_frames:]
        for p in extras:
            try:
                p.unlink()
            except OSError:
                pass

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frames


def _probe_duration_seconds(video_path: Path) -> float:
    ffprobe = _require_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError as e:
        raise RuntimeError(f"Unable to parse duration from ffprobe output: {proc.stdout!r}") from e


def extract_frames_uniform(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
) -> List[Path]:
    """Uniformly sample N frames by timestamp using repeated ffmpeg seeks.

    This avoids extracting many intermediate frames for long videos.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    ffmpeg = _require_ffmpeg()
    duration = _probe_duration_seconds(video_path)
    if not (duration > 0.0) or math.isinf(duration) or math.isnan(duration):
        raise RuntimeError(f"Invalid duration for {video_path}: {duration}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample at centers of N equal segments: (i+0.5)/N * duration
    timestamps = [((i + 0.5) * duration) / num_frames for i in range(num_frames)]
    for idx, t in enumerate(timestamps, start=1):
        out_path = output_dir / f"{idx:06d}.png"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{t:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(out_path),
        ]
        _run(cmd)

    frames = sorted(output_dir.glob("*.png"))
    if len(frames) != num_frames:
        raise RuntimeError(f"Expected {num_frames} frames but got {len(frames)} for {video_path}")
    return frames


def list_videos(folder: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def load_prompts(
    fake_videos: List[Path],
    prompt: Optional[str],
    prompts_file: Optional[Path],
) -> Dict[str, str]:
    """Return mapping: video_stem -> prompt.

    Priority:
    1) --prompts_file (one line per video, matched by sorted fake video list)
    2) --prompt (single prompt for all videos)
    """

    if prompts_file is not None:
        lines = prompts_file.read_text(encoding="utf-8").splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        if len(lines) != len(fake_videos):
            raise ValueError(
                f"prompts_file has {len(lines)} non-empty lines, but fake_dir has {len(fake_videos)} videos."
            )
        return {v.stem: t for v, t in zip(fake_videos, lines)}

    if prompt is not None:
        return {v.stem: prompt for v in fake_videos}

    return {}


def chunked(seq: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    for i in range(0, len(seq), batch_size):
        yield list(seq[i : i + batch_size])


def read_images_batch(paths: List[Path], transform: transforms.Compose) -> torch.Tensor:
    imgs: List[torch.Tensor] = []
    for p in paths:
        img = read_image(str(p))  # uint8, (C,H,W)
        if img.shape[0] != 3:
            img = img.expand(3, -1, -1)
        imgs.append(transform(img))
    return torch.stack(imgs, dim=0)


def _resize_uint8_image(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a single image tensor to `size` and return uint8.

    Args:
        img: uint8 tensor (C,H,W) in [0,255]
        size: (H,W)
    """

    if img.dtype != torch.uint8:
        img = img.to(torch.uint8)
    x = img.float().unsqueeze(0)  # (1,C,H,W)
    x = torch_f.interpolate(x, size=size, mode="bilinear", align_corners=False)
    x = x.round().clamp(0, 255).to(torch.uint8)
    return x.squeeze(0)


def read_images_batch_uint8(paths: List[Path], size_hw: Tuple[int, int]) -> torch.Tensor:
    imgs: List[torch.Tensor] = []
    for p in paths:
        img = read_image(str(p))  # uint8, (C,H,W)
        if img.shape[0] != 3:
            img = img.expand(3, -1, -1)
        img = _resize_uint8_image(img, size=size_hw)
        imgs.append(img)
    return torch.stack(imgs, dim=0)


def _as_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, (float, int)):
        return float(x)
    raise TypeError(f"Cannot convert to float: {type(x)}")


# ------------------- Discontinuity Score (DS) -------------------


def detect_seam(image: torch.Tensor) -> torch.Tensor:
    """Detect vertical seam around left/right boundary.

    Args:
        image: torch.Tensor (C,H,W) in [0,1] float.

    Returns:
        torch.Tensor (H,4)
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor.")
    if image.ndim != 3:
        raise ValueError("Input image must be 3-dimensional (C, H, W).")
    if image.shape[2] < 6:
        raise ValueError("Input image width must be at least 6 pixels.")

    if image.shape[0] == 3:
        grayscale = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).unsqueeze(0)
    else:
        grayscale = image

    left_boundary = grayscale[:, :, -3:]
    right_boundary = grayscale[:, :, :3]
    boundary = torch.cat([left_boundary, right_boundary], dim=2)

    scharr = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=boundary.dtype, device=boundary.device)
    scharr = scharr.view(1, 1, 3, 3)
    conv = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=(1, 0)).to(boundary.device)
    with torch.no_grad():
        conv.weight.copy_(scharr)
    out = conv(boundary.unsqueeze(0))
    return out.squeeze(0).squeeze(0)


def compute_ds(a_hat: torch.Tensor, c: float = 0.1) -> float:
    if a_hat.ndim != 2 or a_hat.shape[1] != 4:
        raise ValueError("Input tensor a_hat must have shape (H, 4).")
    h = a_hat.size(0)
    term1 = torch.abs(a_hat[:, 1]) / (torch.abs(a_hat[:, 0]) + c)
    term2 = torch.abs(a_hat[:, 2]) / (torch.abs(a_hat[:, 3]) + c)
    return _as_float((1.0 / (2 * h)) * torch.sum(term1 + term2))


def average_ds(frame_paths: List[Path], device: torch.device) -> float:
    total = 0.0
    for p in frame_paths:
        img = read_image(str(p)).to(device)
        if img.shape[0] != 3:
            img = img.expand(3, -1, -1)
        img = img.float() / 255.0
        seam = detect_seam(img)
        total += compute_ds(seam)
    return total / max(1, len(frame_paths))


# ------------------- Main runner -------------------


@dataclass
class EvalConfig:
    real_dir: Optional[Path]
    fake_dir: Path
    out_dir: Path
    fps: float
    num_frames_per_video: Optional[int]
    every_frame: bool
    every_k_frames: Optional[int]
    max_frames_per_video: Optional[int]
    batch_size: int
    device: torch.device

    inception_resize: int
    clip_resize: int

    compute_clip: bool
    compute_is: bool
    compute_ds: bool
    compute_fid: bool
    compute_kid: bool

    prompt: Optional[str]
    prompts_file: Optional[Path]

    pairing: str

    # Optional
    compute_omnifid: bool
    compute_faed: bool
    faed_ckpt_dir: Optional[Path]
    cmp_face_width: int


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="One-click evaluation for .mp4 panorama videos (frame-based).")
    parser.add_argument("--fake_dir", type=str, required=True, help="Folder containing generated videos (.mp4, etc).")
    parser.add_argument("--real_dir", type=str, default=None, help="Folder containing GT videos (for FID/KID).")
    parser.add_argument("--out_dir", type=str, default="_eval_cache", help="Cache dir for extracted frames.")

    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling FPS.")
    parser.add_argument(
        "--num_frames_per_video",
        type=int,
        default=None,
        help="Uniformly sample N frames per video by timestamp (recommended for long videos). If set, overrides --fps.",
    )
    parser.add_argument(
        "--every_frame",
        action="store_true",
        help="Extract every frame from each video (most faithful; overrides --fps/--num_frames_per_video).",
    )
    parser.add_argument(
        "--every_k_frames",
        type=int,
        default=None,
        help="Extract every k-th frame (by frame index): 0,k,2k,... Overrides --fps/--num_frames_per_video.",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Cap extracted frames per video (after FPS sampling).",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for torchmetrics updates.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda|cpu. Default auto-detect.",
    )

    parser.add_argument(
        "--inception_resize",
        type=int,
        default=299,
        help=(
            "Resize size for Inception-based metrics (FID/KID/IS and OmniFID faces). "
            "Default 299 (standard setting). You may set 256/512, but results may not be directly comparable to standard FID."
        ),
    )
    parser.add_argument(
        "--clip_resize",
        type=int,
        default=224,
        help="Resize size for CLIPScore images. Default 224 (CLIP ViT-L/14 standard).",
    )

    parser.add_argument("--prompt", type=str, default=None, help="Single prompt applied to all fake videos.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Text file: one prompt per video, matched by sorted fake videos.",
    )

    parser.add_argument(
        "--pairing",
        type=str,
        choices=["auto", "stem", "sorted"],
        default="auto",
        help=(
            "How to pair fake vs real videos when --real_dir is provided. "
            "auto: try stem match; if mismatch but counts equal, fall back to sorted pairing. "
            "stem: require same filename stem. sorted: pair by sorted order."
        ),
    )

    parser.add_argument("--no_clip", action="store_true", help="Disable CLIPScore.")
    parser.add_argument("--no_is", action="store_true", help="Disable Inception Score.")
    parser.add_argument("--no_ds", action="store_true", help="Disable Discontinuity Score.")
    parser.add_argument("--no_fid", action="store_true", help="Disable FID.")
    parser.add_argument("--no_kid", action="store_true", help="Disable KID.")

    parser.add_argument("--omnifid", action="store_true", help="Compute OmniFID (requires cubemap conversion).")
    parser.add_argument("--faed", action="store_true", help="Compute FAED (requires external modules + ckpt).")
    parser.add_argument("--faed_ckpt_dir", type=str, default=None, help="Directory containing faed.ckpt")
    parser.add_argument(
        "--cmp_face_width",
        type=int,
        default=256,
        help="Cubemap face width/height used by convert360 e2c when --omnifid is enabled.",
    )

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    return EvalConfig(
        real_dir=Path(args.real_dir) if args.real_dir else None,
        fake_dir=Path(args.fake_dir),
        out_dir=Path(args.out_dir),
        fps=float(args.fps),
        num_frames_per_video=int(args.num_frames_per_video) if args.num_frames_per_video is not None else None,
        every_frame=bool(args.every_frame),
        every_k_frames=int(args.every_k_frames) if args.every_k_frames is not None else None,
        max_frames_per_video=int(args.max_frames_per_video) if args.max_frames_per_video is not None else None,
        batch_size=int(args.batch_size),
        device=device,
        inception_resize=int(args.inception_resize),
        clip_resize=int(args.clip_resize),
        compute_clip=not args.no_clip,
        compute_is=not args.no_is,
        compute_ds=not args.no_ds,
        compute_fid=not args.no_fid,
        compute_kid=not args.no_kid,
        prompt=args.prompt,
        prompts_file=Path(args.prompts_file) if args.prompts_file else None,
        pairing=str(args.pairing),
        compute_omnifid=bool(args.omnifid),
        compute_faed=bool(args.faed),
        faed_ckpt_dir=Path(args.faed_ckpt_dir) if args.faed_ckpt_dir else None,
        cmp_face_width=int(args.cmp_face_width),
    )


def _collect_face_images(root: Path, face: str) -> List[Path]:
    # Accept nested structure: */<video_stem>/<face>/*.png
    # and flat structure: */<face>/*.png
    return sorted([p for p in root.rglob("*") if p.is_file() and p.parent.name == face])


def _ensure_convert360() -> None:
    if not shutil.which("convert360"):
        raise RuntimeError(
            "convert360 not found in PATH. OmniFID requires ERP->cubemap conversion via convert360. "
            "Please install convert360 and ensure it is callable from PowerShell."
        )


def extract_all_frames(
    videos: List[Path],
    out_root: Path,
    fps: float,
    num_frames_per_video: Optional[int],
    every_frame: bool,
    every_k_frames: Optional[int],
    max_frames: Optional[int],
) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for v in videos:
        out_dir = out_root / v.stem
        existing = sorted(out_dir.glob("*.png")) if out_dir.exists() else []

        meta_path = out_dir / "_extract_meta.json"
        if every_frame:
            mode = "every_frame"
        elif every_k_frames is not None:
            mode = "every_k_frames"
        elif num_frames_per_video is not None:
            mode = "uniform"
        else:
            mode = "fps"

        desired_meta: Dict[str, object] = {
            "mode": mode,
            "fps": float(fps),
            "num_frames_per_video": int(num_frames_per_video) if num_frames_per_video is not None else None,
            "every_k_frames": int(every_k_frames) if every_k_frames is not None else None,
            "max_frames_per_video": int(max_frames) if max_frames is not None else None,
        }

        existing_meta: Optional[Dict[str, object]] = None
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_meta = None

        # Decide expected count for cache reuse.
        expected: Optional[int] = None
        if every_frame:
            expected = None  # unknown without probing; reuse if any frames exist
        elif every_k_frames is not None:
            expected = None  # unknown without probing; reuse if any frames exist
        elif num_frames_per_video is not None:
            expected = num_frames_per_video
            if max_frames is not None:
                expected = min(expected, max_frames)
        elif max_frames is not None:
            expected = max_frames

        reuse_ok = bool(existing) and (existing_meta == desired_meta)
        if expected is not None:
            reuse_ok = reuse_ok and len(existing) >= expected

        if not reuse_ok:
            if out_dir.exists():
                shutil.rmtree(out_dir)
            if every_frame:
                frames = extract_frames_all(v, out_dir, max_frames=max_frames)
            elif every_k_frames is not None:
                frames = extract_frames_every_k(v, out_dir, k=every_k_frames, max_frames=max_frames)
            elif num_frames_per_video is not None:
                n = num_frames_per_video
                if max_frames is not None:
                    n = min(n, max_frames)
                frames = extract_frames_uniform(v, out_dir, num_frames=n)
            else:
                frames = extract_frames_ffmpeg(v, out_dir, fps=fps, max_frames=max_frames)

            try:
                meta_path.write_text(json.dumps(desired_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
        else:
            frames = existing
            if expected is not None:
                frames = frames[:expected]
        mapping[v.stem] = frames
    return mapping


def flatten_frames(frames_by_video: Dict[str, List[Path]]) -> List[Path]:
    out: List[Path] = []
    for _, frames in frames_by_video.items():
        out.extend(frames)
    return out


def main() -> None:
    cfg = parse_args()

    if not cfg.fake_dir.exists():
        raise FileNotFoundError(f"fake_dir not found: {cfg.fake_dir}")
    if cfg.real_dir is not None and not cfg.real_dir.exists():
        raise FileNotFoundError(f"real_dir not found: {cfg.real_dir}")

    fake_videos = list_videos(cfg.fake_dir)
    if not fake_videos:
        raise RuntimeError(f"No videos found under fake_dir: {cfg.fake_dir}")

    real_videos: List[Path] = []
    if cfg.real_dir is not None:
        real_videos = list_videos(cfg.real_dir)
        if not real_videos:
            raise RuntimeError(f"No videos found under real_dir: {cfg.real_dir}")

        if cfg.pairing in {"auto", "stem"}:
            # Match by filename stem
            real_by_stem = {p.stem: p for p in real_videos}
            missing = [p.stem for p in fake_videos if p.stem not in real_by_stem]
            if not missing:
                real_videos = [real_by_stem[p.stem] for p in fake_videos]
            else:
                if cfg.pairing == "stem":
                    raise RuntimeError(
                        "Stem mismatch between fake and real videos. Missing in real_dir: " + ", ".join(missing[:20])
                    )
                # auto fallback: if counts equal, pair by sorted order
                if len(real_videos) != len(fake_videos):
                    raise RuntimeError(
                        "Stem mismatch and video counts differ; cannot auto-pair. "
                        f"fake={len(fake_videos)}, real={len(real_videos)}. Missing stems: " + ", ".join(missing[:20])
                    )
                print(
                    "[warn] Video stems differ between real/fake. Falling back to sorted pairing. "
                    "Use --pairing stem to enforce exact matching."
                )
                real_videos = sorted(real_videos)
                fake_videos = sorted(fake_videos)
        elif cfg.pairing == "sorted":
            if len(real_videos) != len(fake_videos):
                raise RuntimeError(
                    f"--pairing sorted requires equal counts, but fake={len(fake_videos)} real={len(real_videos)}"
                )
            real_videos = sorted(real_videos)
            fake_videos = sorted(fake_videos)
        else:
            raise RuntimeError(f"Unknown pairing mode: {cfg.pairing}")

    prompts = load_prompts(fake_videos, cfg.prompt, cfg.prompts_file)
    if cfg.compute_clip and not prompts:
        raise RuntimeError("CLIPScore enabled but no prompts provided. Use --prompt or --prompts_file.")

    sampling_flags = int(cfg.every_frame) + int(cfg.every_k_frames is not None) + int(cfg.num_frames_per_video is not None)
    if sampling_flags > 1:
        raise RuntimeError("Please set only one of --every_frame, --every_k_frames, or --num_frames_per_video")

    cache_root = cfg.out_dir
    frames_root = cache_root / "frames"
    fake_frames_root = frames_root / "fake"
    real_frames_root = frames_root / "real"

    print(f"[1/3] Extracting frames to: {frames_root}")
    fake_frames_by_video = extract_all_frames(
        fake_videos,
        fake_frames_root,
        cfg.fps,
        cfg.num_frames_per_video,
        cfg.every_frame,
        cfg.every_k_frames,
        cfg.max_frames_per_video,
    )
    fake_frame_paths = flatten_frames(fake_frames_by_video)

    real_frame_paths: List[Path] = []
    if cfg.real_dir is not None:
        real_frames_by_video = extract_all_frames(
            real_videos,
            real_frames_root,
            cfg.fps,
            cfg.num_frames_per_video,
            cfg.every_frame,
            cfg.every_k_frames,
            cfg.max_frames_per_video,
        )
        real_frame_paths = flatten_frames(real_frames_by_video)

    results: Dict[str, object] = {
        "fps": cfg.fps,
        "num_frames_per_video": cfg.num_frames_per_video,
        "every_frame": cfg.every_frame,
        "every_k_frames": cfg.every_k_frames,
        "max_frames_per_video": cfg.max_frames_per_video,
        "inception_resize": cfg.inception_resize,
        "clip_resize": cfg.clip_resize,
        "num_fake_videos": len(fake_videos),
        "num_fake_frames": len(fake_frame_paths),
        "num_real_videos": len(real_videos) if cfg.real_dir is not None else 0,
        "num_real_frames": len(real_frame_paths) if cfg.real_dir is not None else 0,
        "pairing": cfg.pairing,
        "video_pairs": (
            [{"fake": f.name, "real": r.name} for f, r in zip(fake_videos, real_videos)] if cfg.real_dir is not None else []
        ),
    }

    print("[2/3] Computing metrics...")

    # DS on generated frames
    if cfg.compute_ds:
        ds_value = average_ds(fake_frame_paths, device=cfg.device)
        results["DS"] = ds_value
        print(f"DS: {ds_value:.6f}")

    # Torchmetrics metrics
    # We import lazily so users can disable metrics without installing heavy deps.
    if cfg.compute_fid or cfg.compute_kid or cfg.compute_is:
        tm = "torchmetrics"
        FrechetInceptionDistance = importlib.import_module(tm + ".image.fid").FrechetInceptionDistance
        KernelInceptionDistance = importlib.import_module(tm + ".image.kid").KernelInceptionDistance
        InceptionScore = importlib.import_module(tm + ".image.inception").InceptionScore

        size_inception = (cfg.inception_resize, cfg.inception_resize)

        # KID has a strict requirement: subset_size must be smaller than the number of samples
        # (for both real and fake). Auto-adjust for small frame counts.
        effective_kid_subset_size: Optional[int] = None
        if cfg.compute_kid:
            if cfg.real_dir is None:
                results["KID"] = None
                results["KID_skip_reason"] = "no_real_dir"
            else:
                n_samples = min(len(fake_frame_paths), len(real_frame_paths))
                # torchmetrics requires at least 2 samples and subset_size < n_samples
                if n_samples <= 1:
                    results["KID"] = None
                    results["KID_skip_reason"] = f"too_few_samples:{n_samples}"
                else:
                    # Default subset_size mirrors KID_cal.py (50)
                    effective_kid_subset_size = min(50, n_samples - 1)
                    results["KID_subset_size"] = effective_kid_subset_size

        fid = FrechetInceptionDistance(feature=2048).to(cfg.device) if cfg.compute_fid else None
        kid = (
            KernelInceptionDistance(subset_size=effective_kid_subset_size).to(cfg.device)
            if (cfg.compute_kid and effective_kid_subset_size is not None)
            else None
        )
        inception = InceptionScore().to(cfg.device) if cfg.compute_is else None

        # Update fake stats
        for batch_paths in chunked(fake_frame_paths, cfg.batch_size):
            imgs = read_images_batch_uint8(batch_paths, size_hw=size_inception).to(cfg.device)
            if fid is not None:
                fid.update(imgs, real=False)
            if kid is not None:
                kid.update(imgs, real=False)
            if inception is not None:
                inception.update(imgs)

        # Update real stats
        if cfg.real_dir is not None:
            for batch_paths in chunked(real_frame_paths, cfg.batch_size):
                imgs = read_images_batch_uint8(batch_paths, size_hw=size_inception).to(cfg.device)
                if fid is not None:
                    fid.update(imgs, real=True)
                if kid is not None:
                    kid.update(imgs, real=True)

        if inception is not None:
            is_value = inception.compute()
            # torchmetrics may return (mean, std)
            if isinstance(is_value, (tuple, list)) and len(is_value) == 2:
                results["IS_mean"] = _as_float(is_value[0])
                results["IS_std"] = _as_float(is_value[1])
                print(f"IS: mean={results['IS_mean']:.6f}, std={results['IS_std']:.6f}")
            else:
                results["IS"] = _as_float(is_value)
                print(f"IS: {results['IS']:.6f}")

        if fid is not None:
            if cfg.real_dir is None:
                print("FID skipped (no real_dir provided).")
            else:
                fid_value = fid.compute()
                results["FID"] = _as_float(fid_value)
                print(f"FID: {results['FID']:.6f}")

        if cfg.compute_kid:
            if kid is None:
                # Already recorded skip reason above.
                if "KID_skip_reason" in results:
                    print(f"KID skipped ({results['KID_skip_reason']})")
                else:
                    print("KID skipped")
            else:
                try:
                    kid_mean, kid_std = kid.compute()
                    results["KID_mean"] = _as_float(kid_mean)
                    results["KID_std"] = _as_float(kid_std)
                    print(f"KID: mean={results['KID_mean']:.6f}, std={results['KID_std']:.6f}")
                except ValueError as e:
                    results["KID"] = None
                    results["KID_error"] = str(e)
                    print(f"KID skipped: {e}")

    # CLIPScore
    if cfg.compute_clip:
        tm = "torchmetrics"
        CLIPScore = importlib.import_module(tm + ".multimodal.clip_score").CLIPScore

        clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(cfg.device)
        clip_tf = transforms.Compose(
            [
                transforms.Resize((cfg.clip_resize, cfg.clip_resize)),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

        # Build per-frame prompt list matched to frame order
        frame_prompts: List[str] = []
        ordered_frames: List[Path] = []
        for v in fake_videos:
            frames = fake_frames_by_video[v.stem]
            ordered_frames.extend(frames)
            frame_prompts.extend([prompts[v.stem]] * len(frames))

        scores: List[float] = []
        for i in range(0, len(ordered_frames), cfg.batch_size):
            batch_frames = ordered_frames[i : i + cfg.batch_size]
            batch_texts = frame_prompts[i : i + cfg.batch_size]
            imgs = read_images_batch(batch_frames, clip_tf).to(cfg.device)
            # torchmetrics CLIPScore supports list[str] texts
            s = clip(imgs, batch_texts)
            # s is a scalar tensor for batch average
            scores.append(_as_float(s))

        avg_clip = sum(scores) / max(1, len(scores))
        results["CLIPScore"] = float(avg_clip)
        print(f"CLIPScore: {avg_clip:.6f}")

    # Optional metrics are intentionally conservative: skip with clear messages.
    if cfg.compute_omnifid:
        try:
            _ensure_convert360()
            # Import converter lazily.
            from e2c_converter import convert_and_split_cubemap

            cmp_root = cache_root / "cmp"
            cmp_fake = cmp_root / "fake"
            cmp_real = cmp_root / "real"

            # Convert fake frames
            for v in fake_videos:
                in_dir = fake_frames_root / v.stem
                out_dir = cmp_fake / v.stem
                if not out_dir.exists() or not any((out_dir / "F").glob("*.png")):
                    if out_dir.exists():
                        shutil.rmtree(out_dir)
                    convert_and_split_cubemap(str(in_dir), str(out_dir), width=cfg.cmp_face_width)

            # Convert real frames
            if cfg.real_dir is None:
                raise RuntimeError("OmniFID requires --real_dir (GT videos).")

            for v in real_videos:
                in_dir = real_frames_root / v.stem
                out_dir = cmp_real / v.stem
                if not out_dir.exists() or not any((out_dir / "F").glob("*.png")):
                    if out_dir.exists():
                        shutil.rmtree(out_dir)
                    convert_and_split_cubemap(str(in_dir), str(out_dir), width=cfg.cmp_face_width)

            # Compute per-face FID across all converted frames
            FrechetInceptionDistance = importlib.import_module("torchmetrics" + ".image.fid").FrechetInceptionDistance
            size_inception = (cfg.inception_resize, cfg.inception_resize)

            face_regions = ["U", "F", "R", "B", "L", "D"]
            fid_scores: Dict[str, float] = {}
            for face in face_regions:
                real_face_imgs = _collect_face_images(cmp_real, face)
                fake_face_imgs = _collect_face_images(cmp_fake, face)
                if not real_face_imgs or not fake_face_imgs:
                    raise RuntimeError(f"OmniFID missing face images for {face}.")

                fid = FrechetInceptionDistance(feature=2048).to(cfg.device)

                for batch_paths in chunked(fake_face_imgs, cfg.batch_size):
                    imgs = read_images_batch_uint8(batch_paths, size_hw=size_inception).to(cfg.device)
                    fid.update(imgs, real=False)
                for batch_paths in chunked(real_face_imgs, cfg.batch_size):
                    imgs = read_images_batch_uint8(batch_paths, size_hw=size_inception).to(cfg.device)
                    fid.update(imgs, real=True)

                fid_scores[face] = _as_float(fid.compute())

            frontal_fid = 0.25 * (fid_scores["F"] + fid_scores["R"] + fid_scores["B"] + fid_scores["L"])
            omni_fid = (fid_scores["U"] + frontal_fid + fid_scores["D"]) / 3.0

            results["OmniFID_faces"] = fid_scores
            results["OmniFID_frontal"] = float(frontal_fid)
            results["OmniFID"] = float(omni_fid)
            print(f"OmniFID: {omni_fid:.6f}")
        except (RuntimeError, FileNotFoundError, ImportError, ValueError, OSError) as e:
            print(f"OmniFID skipped: {e}")
            results["OmniFID"] = None

    if cfg.compute_faed:
        print("FAED: not implemented in this runner yet (requires FAED project modules + checkpoint wiring).")
        results["FAED"] = None

    # Write results
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[3/3] Done. Results written to: {out_path}")


if __name__ == "__main__":
    main()
