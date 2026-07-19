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

    # ---- Input source (pick one) ----
    # Option A: direct text prompts
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="One or more text prompts. Beats/chords are disabled; use --features_json to add them.")
    # Option B: JSON lines file (same format as training)
    parser.add_argument("--input_json", type=str, default=None,
                        help="JSON lines file with prompts and optional beats/chords. One JSON object per line.")
    parser.add_argument("--text_key", type=str, default="main_caption",
                        help="Key for the text prompt in --input_json.")
    parser.add_argument("--beats_key", type=str, default="beats",
                        help="Key for beats data in --input_json.")
    parser.add_argument("--chords_key", type=str, default="chords",
                        help="Key for chords data in --input_json.")
    parser.add_argument("--chords_time_key", type=str, default="chords_time",
                        help="Key for chords time data in --input_json.")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional prefix prepended to every text prompt.")

    # Option C: separate music-feature JSON for use with --prompts
    parser.add_argument("--features_json", type=str, default=None,
                        help="JSON lines file with beats/chords/chords_time for each --prompts entry "
                             "(same order, one object per line, keys: beats, chords, chords_time).")

    # ---- Generation parameters ----
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

    # ---- Output ----
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated audio files.")
    parser.add_argument("--samplerate", type=int, default=16000,
                        help="Sample rate for saved audio.")

    # ---- Device ----
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on.")

    args = parser.parse_args()

    # Validation
    if args.prompts is None and args.input_json is None:
        parser.error("Either --prompts or --input_json must be provided.")
    if args.prompts is not None and args.input_json is not None:
        parser.error("Provide only one of --prompts or --input_json, not both.")
    if args.features_json is not None and args.prompts is None:
        parser.error("--features_json can only be used together with --prompts.")

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

    # ---- 2. Prepare input prompts & determine music-feature flags ----
    if args.prompts:
        text_prompts = [args.prefix + p for p in args.prompts]
        n_prompts = len(text_prompts)

        if args.features_json:
            # Read beats/chords from a separate JSON lines file (one line per prompt)
            with open(args.features_json, "r") as f:
                feat_lines = [json.loads(line) for line in f.readlines()]
            if len(feat_lines) != n_prompts:
                raise ValueError(
                    f"--features_json has {len(feat_lines)} entries but --prompts has {n_prompts}. "
                    "They must match 1:1 in order."
                )
            beats_list = [line.get(args.beats_key, [[], []]) for line in feat_lines]
            chords_list = [line.get(args.chords_key, []) for line in feat_lines]
            chords_time_list = [line.get(args.chords_time_key, []) for line in feat_lines]
            use_beats = True
            use_chords = True
            print("Music features loaded from --features_json.")
        else:
            # Text-only: disable music features
            beats_list = [[[], []] for _ in range(n_prompts)]
            chords_list = [[] for _ in range(n_prompts)]
            chords_time_list = [[] for _ in range(n_prompts)]
            use_beats = False
            use_chords = False
            print("No music features provided -- beats/chords conditioning disabled.")

        output_filenames = [f"gen_{i:04d}.wav" for i in range(n_prompts)]

    else:
        # --input_json mode
        with open(args.input_json, "r") as f:
            test_lines = [json.loads(line) for line in f.readlines()]
        if args.test_limit > 0:
            test_lines = test_lines[:args.test_limit]

        n_prompts = len(test_lines)
        text_prompts = [args.prefix + line[args.text_key] for line in test_lines]
        beats_list = [line.get(args.beats_key, [[], []]) for line in test_lines]
        chords_list = [line.get(args.chords_key, []) for line in test_lines]
        chords_time_list = [line.get(args.chords_time_key, []) for line in test_lines]

        # Auto-detect: enable features if any sample has non-empty beats/chords
        use_beats = any(len(b) > 0 and len(b[0]) > 0 for b in beats_list)
        use_chords = any(len(c) > 0 for c in chords_list)
        if not use_beats and not use_chords:
            print("No music features detected in JSON -- beats/chords conditioning disabled.")

        output_filenames = [os.path.basename(line.get("location", f"gen_{i:04d}.wav"))
                            for i, line in enumerate(test_lines)]

    print(f"use_beats={use_beats}, use_chords={use_chords}")
    print(f"Number of prompts: {len(text_prompts)}")
    print(f"First prompt: {text_prompts[0][:80]}...")

    # ---- 3. Load diffusion model ----
    print("\nInitializing DPOMusicAudioDiffusion...")
    model = DPOMusicAudioDiffusion(
        text_encoder_name=args.text_encoder_name,
        scheduler_name=args.scheduler_name,
        unet_model_config_path=args.unet_config,
        use_beats=use_beats,
        use_chords=use_chords,
    ).to(device)

    # Load weights
    model_file = os.path.join(args.model_path, "model_2.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model weights not found: {model_file}")

    print(f"Loading model weights from: {model_file}")
    state_dict = load_file(model_file)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded {len(state_dict)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.eval()
    print("Model loaded successfully!\n")

    # ---- 4. Set up scheduler ----
    scheduler = DDPMScheduler.from_pretrained(args.scheduler_name, subfolder="scheduler")

    # ---- 5. Generation loop ----
    os.makedirs(args.output_dir, exist_ok=True)

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

    print(f"\nSaved {len(output_filenames)} audio files to: {args.output_dir}")


if __name__ == "__main__":
    main()
