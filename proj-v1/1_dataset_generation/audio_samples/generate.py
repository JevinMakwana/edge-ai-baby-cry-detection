import ast
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
HF_HOME = ROOT_DIR / ".hf-cache"

os.environ.setdefault("AUDIOLDM_CACHE_DIR", str(ROOT_DIR / ".audioldm-cache"))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(ROOT_DIR / ".torch-kernels"))

Path(os.environ["AUDIOLDM_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["PYTORCH_KERNEL_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)

from audioldm import text_to_audio, build_model


def patch_audioldm_for_cpu():
    """Patch AudioLDM's DDIMSampler to work on CPU (no-op if CUDA is available)."""
    if torch.cuda.is_available():
        return

    from audioldm.latent_diffusion.ddim import DDIMSampler

    def register_buffer_cpu_safe(self, name, attr):
        if isinstance(attr, torch.Tensor):
            target_device = getattr(self.model, "device", torch.device("cpu"))
            if attr.device != target_device:
                attr = attr.to(target_device)
        setattr(self, name, attr)

    DDIMSampler.register_buffer = register_buffer_cpu_safe


patch_audioldm_for_cpu()

# Load model once (expensive — keep this outside generate_batch)
ldm = build_model(model_name="audioldm-s-full")


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompts_from_txt(file_path, variable_name="prompts"):
    """Parse a .txt file containing a Python list assignment and return the list."""
    file_path = Path(file_path)
    content = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            content = file_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise UnicodeDecodeError(
            "prompt_loader",
            b"",
            0,
            1,
            f"Unsupported encoding in {file_path}",
        )

    module = ast.parse(content, filename=str(file_path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)

    raise ValueError(f"Variable '{variable_name}' not found in {file_path}")


# ---------------------------------------------------------------------------
# Waveform utilities
# ---------------------------------------------------------------------------

def prepare_waveform_for_wav(waveform):
    """Convert any waveform type to a clean float32 numpy array suitable for sf.write."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().float().cpu().numpy()
    else:
        waveform = np.asarray(waveform, dtype=np.float32)

    waveform = np.squeeze(waveform)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
    waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
    return waveform


def compute_rms(waveform: np.ndarray) -> float:
    """Return the RMS energy of a waveform. Values below 0.005 indicate silence."""
    return float(np.sqrt(np.mean(waveform ** 2)))


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def generate_batch(
    prompts,
    output_dir,
    samples_per_prompt: int = 3,
    base_seed: int = 42,
    rms_threshold: float = 0.005,
    log_path: Path = None,
):
    """
    Generate `samples_per_prompt` WAV files for every prompt in `prompts`.

    Improvements over original:
    - Always uses distinct seeds (base_seed, base_seed+1, ...) so outputs truly differ.
    - Skips files that already exist — safe to re-run after a cluster crash.
    - Inline RMS check flags silent/near-silent files immediately.
    - Appends a JSON line to `log_path` for every file saved.

    Args:
        prompts:            List of text prompts.
        output_dir:         Directory to write WAV files into.
        samples_per_prompt: How many distinct audio clips to generate per prompt.
        base_seed:          Starting seed; each sample j uses seed = base_seed + j.
        rms_threshold:      Files with RMS below this are flagged as silent.
        log_path:           Path to a .jsonl file for generation logs (optional).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_prompts = len(prompts)
    saved = 0
    skipped = 0
    silent_files = []

    for i, prompt in enumerate(prompts):
        # --- Resume support: skip prompt entirely if all samples already exist ---
        all_exist = all(
            (output_dir / f"{i:03d}_{j}.wav").exists()
            for j in range(samples_per_prompt)
        )
        if all_exist:
            print(f"[{i+1}/{total_prompts}] Skipping (all {samples_per_prompt} samples exist): {prompt}")
            skipped += samples_per_prompt
            continue

        # --- Generate each sample with a distinct seed ---
        for j in range(samples_per_prompt):
            filename = output_dir / f"{i:03d}_{j}.wav"

            # Resume support: skip individual file if it already exists
            if filename.exists():
                print(f"  [{j+1}/{samples_per_prompt}] Skipping (exists): {filename.name}")
                skipped += 1
                continue

            seed = base_seed + j
            waveforms = text_to_audio(
                ldm,
                text=prompt,
                seed=seed,
                duration=5.0,           # 5-second clips (Edge Impulse sweet spot)
                guidance_scale=2.5,     # Higher = closer to prompt; range 2.5–3.5 recommended
                n_candidate_gen_per_text=1,
            )

            if waveforms is None or (hasattr(waveforms, '__len__') and len(waveforms) == 0):
                print(f"  WARNING: No waveform returned for prompt {i}, sample {j}. Skipping.")
                continue

            wav_array = prepare_waveform_for_wav(waveforms[0])

            # --- Inline RMS quality check ---
            rms = compute_rms(wav_array)
            if rms < rms_threshold:
                warning = f"SILENT/QUIET (rms={rms:.5f}): {filename}"
                print(f"  WARNING: {warning}")
                silent_files.append(str(filename))

            # --- Save WAV ---
            sf.write(str(filename), wav_array, samplerate=16000, format="WAV", subtype="PCM_16")
            saved += 1
            print(f"  [{j+1}/{samples_per_prompt}] seed={seed} rms={rms:.4f} → {filename.name}  |  {prompt}")

            # --- Append to generation log ---
            if log_path is not None:
                log_entry = {
                    "prompt_index": i,
                    "sample_index": j,
                    "prompt": prompt,
                    "seed": seed,
                    "rms": round(rms, 6),
                    "file": str(filename),
                    "silent": rms < rms_threshold,
                }
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(log_entry) + "\n")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Done — {output_dir.name}")
    print(f"  Saved  : {saved}")
    print(f"  Skipped: {skipped} (already existed)")
    if silent_files:
        print(f"  Silent files ({len(silent_files)}) — review and delete if needed:")
        for sf_path in silent_files:
            print(f"    {sf_path}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baby-cry and background-noise WAV datasets using AudioLDM."
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=3,
        help="Number of distinct audio clips to generate per prompt (default: 3).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed. Sample j uses seed = base_seed + j (default: 42).",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=0.005,
        help="RMS energy below which a file is flagged as silent (default: 0.005).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable generation logging to generation_log.jsonl.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    output_root = Path(__file__).resolve().parent

    baby_cry_prompts = load_prompts_from_txt(prompts_dir / "baby_cry_prompts.txt")
    background_prompts = load_prompts_from_txt(prompts_dir / "background_noise_prompts.txt")

    print(f"Loaded {len(baby_cry_prompts)} baby_cry prompts")
    print(f"Loaded {len(background_prompts)} background_noise prompts")
    print(f"Samples per prompt : {args.samples_per_prompt}")
    print(f"Base seed          : {args.base_seed}")
    print(f"Expected output    : {(len(baby_cry_prompts) + len(background_prompts)) * args.samples_per_prompt} WAV files total\n")

    log_path = None if args.no_log else output_root / "generation_log.jsonl"

    generate_batch(
        baby_cry_prompts,
        output_root / "baby_cry",
        samples_per_prompt=args.samples_per_prompt,
        base_seed=args.base_seed,
        rms_threshold=args.rms_threshold,
        log_path=log_path,
    )

    generate_batch(
        background_prompts,
        output_root / "background_noise",
        samples_per_prompt=args.samples_per_prompt,
        base_seed=args.base_seed,
        rms_threshold=args.rms_threshold,
        log_path=log_path,
    )

    print("All done! Run the silent-file checker if needed:")
    print("  python -c \"import json; [print(e['file']) for e in (json.loads(l) for l in open('generation_log.jsonl')) if e['silent']]\"")