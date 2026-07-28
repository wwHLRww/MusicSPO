# MusicAlign: Preference Optimization with Structural Conditioning for Text-to-Music Diffusion

Welcome to MusicAlign's GitHub repository! This repository hosts the code, data and model weight of MusicAlign.

**Abstract**: Text-to-music generation models have made substantial progress in generation quality, yet their outputs often fail to align with human auditory preferences and suffer from melodic instability when conditioned on complex text descriptions. In this work, we propose MusicAlign, a preference alignment framework for text-to-music latent diffusion models. First, we construct MusicPre, a music-theory-aware preference dataset built by generating inferior text descriptions via LLM-driven music-theory perturbations and applying CLAP-based bidirectional filtering to select high-quality preference pairs. Second, building upon existing diffusion-DPO alignment methods, we introduce an IPO-based alignment variant and incorporate an auxiliary SFT loss to mitigate generation quality degradation during preference optimization. We further employ explicit beat and chord encoders to inject musical structural information, preserving controllability and acoustic fidelity throughout alignment. Extensive experiments demonstrate that MusicAlign outperforms text-to-music baselines including Mustango and MusicGen鈥攐n both objective and subjective metrics such as CLAP score, overall musical quality, and text relevance. The IPO variant, in particular, exhibits additional advantages in spectral fidelity and cross-domain generalization.  

![](\img\model.png)  

Paper: coming soon

<!-- ### Key Features

- **DPO / IPO Training** 锟?Align a frozen Mustango diffusion model using human preference pairs (chosen vs. rejected)
- **Beat & Chord Conditioning** 锟?Encode beat timestamps/types and chord progressions as auxiliary inputs to the UNet
- **Frozen Encoders** 锟?Text encoder, beat encoder, chord encoder, and VAE are all frozen during training; only the UNet is optimized
- **SFT Warmup** 锟?Optional supervised fine-tuning phase before preference optimization
- **Adaptive Beta** 锟?Dynamic DPO beta scheduling to stabilize training

--- -->
## 1 Installation

```bash
git clone https://github.com/your-username/MusicAlign.git
cd MusicAlign

conda create -n music_align python=3.10 -y
conda activate music_align

pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt

cd diffusers
pip install -e .
```

## 2 Main Results Reproduce
### 2.1 Training Preparation
Download datasets and place them under `./dataset`.

### 2.2 Path Configuration
| Argument | Default Value | Description |
|----------|-------------|-------------|
| `--dataset_dir` | `~/.cache/huggingface/datasets` | Dataset root directory |
| `--train_file` | `{PROJECT_ROOT}/Dataset/MusicPre/train.json` | Training data JSON file |
| `--validation_file` | `{PROJECT_ROOT}/Dataset/MusicPre/validation.json` | Validation data JSON file |
| `--test_file` | `{PROJECT_ROOT}/Dataset/MusicPre/test.json` | Test data JSON file |
| `--output_dir` | `None` | Model & checkpoint output directory |
| `--unet_model_config` | `None` | UNet config JSON |
| `--learning_rate` | `9.6e-7` | learning rate |
| `--num_train_epochs` | `40` | Total training epochs |
| `--per_device_train_batch_size` | `1` | Batch size per GPU |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `--num_warmup_steps` | `0` | Linear warmup steps |
| `--max_train_samples` | `-1` | Max training samples (-1 = all) |
| `--align_type` | `dpo` | Alignment method: `dpo` or `ipo` |
| `--dpo_beta` | `0.5` | DPO temperature coefficient |
| `--eta_ipo` | `0.5` | IPO eta parameter |
| `--zeta` | `0.5` | Weight of Alignment loss |
| `--sft_first_epochs` | `1` | Number of SFT-only warmup epochs |
| `--d_fme` | `1024` | Fourier Music Encoder dimension |
| `--d_pe` | `1024` | Positional encoding dimension |
| `--d_beat` | `1024` | Beat embedding dimension |

### Training a Model

```bash
bash MusicAlign-train.sh
```

---

## Checkpoint *(coming soon)*

---


## Inference

Use `inference_only.py` for text-to-music generation. It supports three input modes:

### Mode 1: Direct text prompts (no beats/chords)

Best for quick generation from plain text. Beats/chords conditioning is automatically disabled.

```bash
python inference_only.py \
    --model_path ./output_dpo/epoch_3 \
    --unet_config configs/diffusion_model_config_munet.json \
    --prompts "A cheerful piano melody" "A calm jazz piece with saxophone" \
    --output_dir ./outputs
```

### Mode 2: Direct prompts + separate music-feature file

When you want beats/chords conditioning with `--prompts`, supply a JSON lines file
via `--features_json`. Each line must correspond 1:1 to a prompt in `--prompts`.

```bash
python inference_only.py \
    --model_path ./output_dpo/epoch_3 \
    --unet_config configs/diffusion_model_config_munet.json \
    --prompts "A cheerful piano melody" "A calm jazz piece" \
    --features_json ./my_features.jsonl \
    --output_dir ./outputs
```

Example `my_features.jsonl`:

```json
{"beats": [[0.5, 1.0], [4, 4]], "chords": ["C:maj", "G:maj"], "chords_time": [0.0, 2.0]}
{"beats": [[0.33, 0.67], [3, 4]], "chords": ["D:min", "A:min"], "chords_time": [0.0, 1.5]}
```

### Mode 3: JSON lines input file (full pipeline)

Same format as training data. Beats/chords are auto-detected from the file.

```bash
python inference_only.py \
    --model_path ./output_dpo/epoch_3 \
    --unet_config configs/diffusion_model_config_munet.json \
    --input_json /path/to/test_prompts.json \
    --text_key main_caption \
    --output_dir ./outputs
```
---
### Dataset *(coming soon)*
---
<!-- ## Citation

If you use this work, please cite:

```bibtex
@misc{musicalign2026,
  title={MusicDPO: Direct Preference Optimization for Text-to-Music Generation},
  author={Your Name},
  year={2026},
  note={Coming soon}
}
``` -->
<!-- 
## License

[Specify your license] -->
