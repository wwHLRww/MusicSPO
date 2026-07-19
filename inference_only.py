import os
import json
import torch
import argparse
import soundfile as sf
from tqdm import tqdm
from diffusers import DDPMScheduler
from models import build_pretrained_models, DPOMusicAudioDiffusion
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only: text-to-music generation.")

    # Model & config paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory (contains model_2.safetensors).")
    parser.add_argument("--unet_config", type=str, required=True,
                        help="UNet model config JSON path.")
    parser.add_argument("--text_encoder_name", type=str,
                        default="stabilityai/stable-diffusion-2-1",
                        help="Text encoder name or path.")
    parser.add_argument("--scheduler_name", type=str,
                        default="stabilityai/stable-diffusion-2-1",
                        help="Scheduler name or path for DDPMScheduler.")
    parser.add_argument("--vae_ckpt", type=str,
                        default="/public/zhoumingtong/Music/pretrained/audioldm-s-full",
                        help="Path to the pretrained VAE checkpoint.")

    # Input (either --prompts or --input_json)
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="One or more text prompts for generation.")
    parser.add_argument("--input_json", type=str, default=None,
                        help="JSON file containing test prompts, one JSON object per line.")
    parser.add_argument("--text_key", type=str, default="main_caption",
                        help="Key for the text prompt in the JSON file.")
    parser.add_argument("--beats_key", type=str, default="beats",
                        help="Key for beats data in JSON.")
    parser.add_argument("--chords_key", type=str, default="chords",
                        help="Key for chords data in JSON.")
    parser.add_argument("--chords_time_key", type=str, default="chords_time",
                        help="Key for chords time data in JSON.")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional prefix prepended to every text prompt.")

    # Generation parameters
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--guidance", type=float, default=3.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per prompt.")
    parser.add_argument("--test_limit", type=int, default=-1,
                        help="Limit the number of test instances (-1 for all).")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated audio files.")
    parser.add_argument("--samplerate", type=int, default=16000,
                        help="Sample rate for saved audio.")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on.")

    # Music feature flags
    parser.add_argument("--use_beats", action="store_true", default=True,
                        help="Use beats features.")
    parser.add_argument("--use_chords", action="store_true", default=True,
                        help="Use chords features.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 1. Load VAE and STFT ----
    print("Loading VAE and STFT...")
    vae, stft = build_pretrained_models("audioldm-s-full")
    vae = vae.to(device)
    stft = stft.to(device)
    vae.eval()
    stft.eval()

    # ---- 2. Load diffusion model ----
    print("Initializing DPOMusicAudioDiffusion...")
    model = DPOMusicAudioDiffusion(
        text_encoder_name=args.text_encoder_name,
        scheduler_name=args.scheduler_name,
        unet_model_config_path=args.unet_config,
        use_beats=args.use_beats,
        use_chords=args.use_chords,
    ).to(device)

    # Load weights
    model_path = os.path.join(args.model_path, "model_2.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    print(f"Loading model weights from: {model_path}")
    state_dict = load_file(model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded {len(state_dict)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.eval()
    print("Model loaded successfully!\n")

    # ---- 3. Set up scheduler ----
    scheduler = DDPMScheduler.from_pretrained(args.scheduler_name, subfolder="scheduler")

    # ---- 4. Prepare input prompts ----
    if args.prompts:
        text_prompts = [args.prefix + p for p in args.prompts]
        beats_list = [[]] * len(text_prompts)
        chords_list = [[]] * len(text_prompts)
        chords_time_list = [[]] * len(text_prompts)
        output_filenames = [f"gen_{i:04d}.wav" for i in range(len(text_prompts))]
    elif args.input_json:
        with open(args.input_json, "r") as f:
            test_lines = [json.loads(line) for line in f.readlines()]
        if args.test_limit > 0:
            test_lines = test_lines[:args.test_limit]

        text_prompts = [args.prefix + line[args.text_key] for line in test_lines]
        beats_list = [line.get(args.beats_key, [[], []]) for line in test_lines]
        chords_list = [line.get(args.chords_key, []) for line in test_lines]
        chords_time_list = [line.get(args.chords_time_key, []) for line in test_lines]
        output_filenames = [os.path.basename(line.get("location", f"gen_{i:04d}.wav"))
                            for i, line in enumerate(test_lines)]
    else:
        raise ValueError("Please provide either --prompts or --input_json.")

    print(f"Number of prompts: {len(text_prompts)}")
    print(f"First prompt: {text_prompts[0][:80]}...")

    # ---- 5. Generation loop ----
    os.makedirs(args.output_dir, exist_ok=True)
    all_wavs = []
    all_filenames = []

    for k in tqdm(range(0, len(text_prompts), args.batch_size), desc="Generating"):
        text_batch = text_prompts[k:k + args.batch_size]
        beats_batch = beats_list[k:k + args.batch_size]
        chords_batch = chords_list[k:k + args.batch_size]
        chords_time_batch = chords_time_list[k:k + args.batch_size]
        filename_batch = output_filenames[k:k + args.batch_size]

        with torch.no_grad():
            latents = model.inference(
                text_batch,
                beats_batch,
                chords_batch,
                chords_time_batch,
                scheduler,
                num_steps=args.num_steps,
                guidance_scale=args.guidance,
                num_samples_per_prompt=args.num_samples,
                disable_progress=True,
            )
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)

        for wav, fname in zip(wave, filename_batch):
            save_path = os.path.join(args.output_dir, fname)
            sf.write(save_path, wav.cpu().numpy(), samplerate=args.samplerate)
            all_wavs.append(wav)
            all_filenames.append(fname)

    print(f"\nSaved {len(all_filenames)} audio files to: {args.output_dir}")


if __name__ == "__main__":
    main()
