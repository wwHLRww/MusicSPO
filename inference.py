import os
import json
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import argparse
import soundfile as sf
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models_forzen import build_pretrained_models, DPOMusicAudioDiffusion
import numpy as np
import laion_clap
from transformers import RobertaModel
import numpy as np
from safetensors.torch import load_file

def clap_score_computation(wav_output_dir, text_prompts):
    cos_sim = torch.nn.CosineSimilarity()
    local_roberta_path = "/public/zhoumingtong/Model/huggingface/hub/models--roberta-base"
    local_clap_ckpt = "/public/zhoumingtong/Music/tango-master/tango2/ckpt/clap/630k-audioset-best.pt"
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(local_clap_ckpt)
    model.text_branch = RobertaModel.from_pretrained(
        local_roberta_path,
        local_files_only=True,
        trust_remote_code=True
    )
    model.eval()

    output_dir = wav_output_dir
    wav_files = [f for f in os.listdir(wav_output_dir) if f.endswith(".wav")]
    audio_file = [os.path.join(wav_output_dir, f) for f in wav_files]

    with torch.no_grad():
        audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True).cpu()
        text_embed = model.get_text_embedding(text_prompts, use_tensor=True).cpu()

    clap_score = torch.mean(cos_sim(audio_embed, text_embed)).item()
    return clap_score

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to music generation task.")
    parser.add_argument("--original_args", type=str, default="/public/zhoumingtong/Music/tango-master/tango2/output_/output_sft/summary.jsonl",
                        help="Path for summary jsonl file saved during training.")
    parser.add_argument("--unet_model_config", type=str, default="/public/zhoumingtong/Music/tango-master/tango2/configs/diffusion_model_config_munet.json",  #choose between configs/diffusion_model_config.json for Tango and configs/diffusion_model_config_munet.json for Mustango
                        help="unet model config json path")
    parser.add_argument("--model", type=str, default="/public/zhoumingtong/Music/tango-master/tango2/output_/output_sft/epoch_3",
                        help="Path for saved model bin file.")
    parser.add_argument("--test_file", type=str, default="/public/zhoumingtong/Dataset/FMACaps/FMACaps_eval_set/FMACaps_B.json",
                        help="json file containing the test prompts for generation.")
    parser.add_argument("--text_key", type=str, default="main_caption",
                        help="Key containing the text in the json file.")
    parser.add_argument("--test_references", type=str, default="Dataset/FMACaps/FMACaps_eval_set/audiodata",
                        help="Folder containing the test reference wav files.")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="How many denoising steps for generation.")
    parser.add_argument("--guidance", type=float, default=3,
                        help="Guidance scale for classifier free guidance.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="How many samples per prompt.")
    parser.add_argument("--test_limit", type=int, default=-1,
                        help="Number of test instances to use (default: 10).")
    parser.add_argument("--location_key", type=str, default="location",
                        help="Key containing the audio file path in the json file (default: location).")
    # Music feature column names
    parser.add_argument("--beats_key", type=str, default="beats",
                        help="Key containing beats data in json.")
    parser.add_argument("--chords_key", type=str, default="chords",
                        help="Key containing chords data in json.")
    parser.add_argument("--chords_time_key", type=str, default="chords_time",
                        help="Key containing chords time data in json.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))


    name = "audioldm-s-full"
    vae, stft = build_pretrained_models(name)
    vae, stft = vae.cuda(), stft.cuda()
    vae.eval()
    stft.eval()

    # Initialize model with same parameters as training
    model = DPOMusicAudioDiffusion(
        text_encoder_name=train_args.text_encoder_name,
        scheduler_name=train_args.scheduler_name,
        unet_model_name=train_args.unet_model_name,
        unet_model_config_path=train_args.unet_model_config,
        snr_gamma=train_args.snr_gamma,
        freeze_text_encoder=train_args.freeze_text_encoder,
        uncondition=train_args.uncondition,
        beta_dpo=2000,  # not used during inference

        d_fme=1024,
        fme_type="se",
        base=1,
        if_trainable=True,
        translation_bias_type="nd",
        emb_nn=True,
        d_pe=1024,
        if_index=True,
        if_global_timing=True,
        if_modulo_timing=False,
        d_beat=1024,
        d_oh_beat_type=7,
        beat_len=50,
        d_chord=1024,
        d_oh_chord_type=12,
        d_oh_inv_type=4,
        chord_len=20,
        use_beats = True,
        use_chords = True,
        lambda_sft=0,

    ).cuda()

    print(f"\nLoading checkpoint from: {args.model}")

        # Method A: load only diffusion model weights (recommended)
    model_2_path = os.path.join(args.model, "model_2.safetensors")
    if os.path.exists(model_2_path):
        print("Loading model_2.safetensors (diffusion model weights)...")
        state_dict = load_file(model_2_path)

        # Load weights (strict=False ignores mismatched keys)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights: {len(state_dict)} keys")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        if len(missing) > 100:
            print("\nWarning: Many missing keys. Model structure may not match.")
            print("First 5 missing keys:", missing[:5])
    else:
        # Method B: load full checkpoint with Accelerate
        print("Loading full checkpoint with Accelerate...")
        accelerator = Accelerator()
        vae, stft, model = accelerator.prepare(vae, stft, model)
        accelerator.load_state(args.checkpoint_dir)
        vae = accelerator.unwrap_model(vae)
        stft = accelerator.unwrap_model(stft)
        model = accelerator.unwrap_model(model)

    model.eval()
    print("Model loaded successfully!\n")


    # Set up scheduler
    stable_diffusion_path = "/public/zhoumingtong/Music/pretrained/stable-diffusion-2-1"
    scheduler = DDPMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

    # Evaluator
    evaluator = EvaluationHelper(16000, "cuda:0")

    # Load test data
    test_lines = [json.loads(line) for line in open(args.test_file).readlines()]
    if args.test_limit > 0:
        test_lines = test_lines[:args.test_limit]

    prefix = train_args.get("prefix", "") or ""

    text_prompts = []
    target_filenames = []
    beats_list = []
    chords_list = []
    chords_time_list = []

    for line in test_lines:
        text = prefix + line[args.text_key]
        text_prompts.append(text)

        location_path = line[args.location_key]
        pure_filename = os.path.basename(location_path)
        target_filenames.append(pure_filename)

        # Read music features (set to empty list if not present)
        beats = line.get(args.beats_key, [[], []])
        chords = line.get(args.chords_key, [])
        chords_time = line.get(args.chords_time_key, [])

        beats_list.append(beats)
        chords_list.append(chords)
        chords_time_list.append(chords_time)

    print(f"Number of test prompts (limited): {len(text_prompts)}")
    print(f"First 3 target filenames: {target_filenames[:3]}")

    # Generation
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []

    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text_batch = text_prompts[k: k+batch_size]
        beats_batch = beats_list[k: k+batch_size]
        chords_batch = chords_list[k: k+batch_size]
        chords_time_batch = chords_time_list[k: k+batch_size]

        with torch.no_grad():
            # Call Mustango inference with music features
            latents = model.inference(
                text_batch,
                beats_batch,
                chords_batch,
                chords_time_batch,
                scheduler,
                num_steps=num_steps,
                guidance_scale=guidance,
                num_samples_per_prompt=num_samples,
                disable_progress=True
            )
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)
            all_outputs += [item for item in wave]

    # Save audio files
    exp_id = str(int(time.time()))
    output_dir = "/public/zhoumingtong/Music/tango-master/tango2/FMACaps/outputs_SFT/ourputs_SFT".format(
        exp_id,
        "_".join(args.model.split("/")[1:-1]),
        num_steps,
        guidance
    )
    os.makedirs(output_dir, exist_ok=True)

    for j, (wav, filename) in enumerate(zip(all_outputs, target_filenames)):
        save_path = os.path.join(output_dir, filename)
        sf.write(save_path, wav, samplerate=16000)
    print(f"Saved {len(target_filenames)} audio files to {output_dir} with target filenames")

    # Compute CLAP score
    print("Computing CLAP score...")
    data = [json.loads(line) for line in open(args.test_file).readlines()]
    file_to_text = {}
    for item in data:
        filename = item["location"].split("/")[-1]
        text = item[args.text_key]
        file_to_text[filename] = text

    wav_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
    text_prompts_sort = [file_to_text[f] for f in wav_files]

    clap_score = clap_score_computation(output_dir, text_prompts_sort)
    print(f"CLAP score: {np.round(clap_score, 4)}")

    # Compute objective metrics (FD, IS, KL, etc.)
    result = evaluator.main(output_dir, args.test_references)
    result["Steps"] = num_steps
    result["Guidance Scale"] = guidance
    result["Test Instances"] = len(text_prompts)
    result["Clap Score"] = np.round(clap_score, 4)

    result["scheduler_config"] = dict(scheduler.config)
    result["args"] = dict(vars(args))
    result["output_dir"] = output_dir

    print("\n===== Evaluation Summary =====\n")
    for key, value in result.items():
        if key not in ["scheduler_config", "args"]:
            print(f"{key}: {value}")

    # Save results
    if not os.path.exists("/public/zhoumingtong/Music/tango-master/tango2/FMACaps/outputs_SFT"):
        os.makedirs("/public/zhoumingtong/Music/tango-master/tango2/FMACaps/outputs_SFT")
    with open("/public/zhoumingtong/Music/tango-master/tango2/FMACaps/outputs_SFT/summary.jsonl", "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n\n")

if __name__ == "__main__":
    main()
