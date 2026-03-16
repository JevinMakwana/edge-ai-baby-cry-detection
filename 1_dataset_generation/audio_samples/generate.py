from audioldm import text_to_audio, build_model
import os, soundfile as sf
import ast
from pathlib import Path
import torch

def patch_audioldm_for_cpu():
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

# Load model once (expensive)
ldm = build_model(model_name="audioldm-s-full")

def load_prompts_from_txt(file_path, variable_name="prompts"):
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


prompts_dir = Path(__file__).resolve().parents[1] / "prompts"

baby_cry_prompts = load_prompts_from_txt(prompts_dir / "baby_cry_prompts.txt")
background_prompts = load_prompts_from_txt(prompts_dir / "background_noise_prompts.txt")

def generate_batch(prompts, output_dir, samples_per_prompt=3):
    os.makedirs(output_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        waveforms = text_to_audio(
            ldm,
            text=prompt,
            seed=42,
            duration=5.0,       # 5 second clips
            guidance_scale=2.5,
            n_candidate_gen_per_text=samples_per_prompt
        )
        for j, wav in enumerate(waveforms):
            filename = f"{output_dir}/{i:03d}_{j}.wav"
            sf.write(filename, wav, samplerate=16000)
            print(f"Generated: {prompt} [{j+1}/{samples_per_prompt}]")

generate_batch(baby_cry_prompts,     "audio_samples/baby_cry")
generate_batch(background_prompts,   "audio_samples/background_noise")
