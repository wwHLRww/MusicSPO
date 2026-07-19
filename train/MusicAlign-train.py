import argparse
import json
import logging
import math
import os
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HuggingFaceDataset
from tqdm.auto import tqdm
import diffusers
import transformers
import tools.torch_tools as torch_tools
from huggingface_hub import snapshot_download
from models import build_pretrained_models, DPOMusicAudioDiffusion, IPOMusicAudioDiffusion
from transformers import SchedulerType, get_scheduler
import copy
from tools.get_dataset import load_custom_dataset, load_eval_dataset

from diffusers.models.attention_processor import AttnProcessor2_0
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/path/to/your/project")  # TODO: set this to your project root directory


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
    parser.add_argument(
        "--train_file", type=str, default=f"{PROJECT_ROOT}/Dataset/MusicPre/train.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=f"{PROJECT_ROOT}/Dataset/MusicPre/validation.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=f"{PROJECT_ROOT}/Dataset/MusicPre/test.json",
        help="A csv or a json file containing the test data for generation."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-base",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1",
        help="Scheduler identifier.",
    )
    parser.add_argument(
        "--unet_model_name", type=str, default=None,
        help="UNet model identifier from huggingface.co/models.",
    )
    parser.add_argument(            #choose between configs/diffusion_model_config.json for Tango and configs/diffusion_model_config_munet.json for Mustango
        "--unet_model_config", type=str, default=None,
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--hf_model", type=str, default=None,
        help="Tango model identifier from huggingface: declare-lab/tango",
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=True,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--audio_w_column", type=str, default="chosen",
        help="The name of the column in the dpo dataset containing the winner audio paths.",
    )
    parser.add_argument(
        "--audio_l_column", type=str, default="rejected",
        help="The name of the column in the dpo dataset containing the loser audio paths.",
    )
    parser.add_argument(
        "--text_column_dpo", type=str, default="prompt",
        help="The name of the column in the dpo dataset containing the input texts.",
    )
    parser.add_argument("--beats_column", type=str, default="beats")

    parser.add_argument("--chords_column", type=str, default="chords")

    parser.add_argument("--chords_time_column", type=str, default="chords_time")

    parser.add_argument("--freeze_beat_encoder", action="store_true", default=True,
                        help="Freeze beat encoder parameters")
    
    parser.add_argument("--freeze_chord_encoder", action="store_true", default=True,
                        help="Freeze chord encoder parameters")

    parser.add_argument("--dpo_beta", type=float, default=0.5,
                        help="Initial beta value for DPO.")
    
    parser.add_argument("--dpo_beta_max", type=float, default=0.5,
                        help="Beta value for dpo.")
    
    parser.add_argument("--dpo_beta_min", type=float, default=0.5,
                        help="Minimum beta value for dpo.")
    
    parser.add_argument("--target_diff_ema", type=float, default=0.1,
                        help="Target value for (model_diff - ref_diff) EMA.")
    
    parser.add_argument("--beta_adjust_lr", type=float, default=0.02,
                        help="Update speed for adaptive beta.")
    
    parser.add_argument("--diff_ema_decay", type=float, default=0.9,
                        help="Decay factor for diff exponential moving average.")
    
    parser.add_argument(
        "--eta_ipo", type=float, default=0.5,
                        help="Eta value for IPO")

    parser.add_argument(
        "--uncondition", action="store_true", default=False,
        help="10% uncondition for training.")
    
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.")
    
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1,
        help="Batch size (per device) for the training dataloader.")
    
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=1,
        help="Batch size (per device) for the validation dataloader.")
    
    parser.add_argument(
        "--learning_rate", type=float, default=9.6e-7,
        help="Initial learning rate (after the potential warmup period) to use.")
    
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
        help="Weight decay to use.")
    
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform.")
    
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer.")
    
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer.")
    
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
        help="Weight decay to use.")
    
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer")
    
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model.")
    
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training.")
    
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.")
    
    parser.add_argument(
        "--save_every", type=int, default=5,
        help="Save model after every how many epochs when checkpointing_steps is set to best.")
    
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.")
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.")
    
    parser.add_argument(
        "--sft_first_epochs", type=int, default=1,
        help="Whether to perform sft on preference data first")
    
    parser.add_argument("--dataset_dir",type=str,default="~/.cache/huggingface/datasets",
                        help="Location to store the downloaded audio")
    
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ))
    
    parser.add_argument(
        "--max_train_samples", type=int, default=-1,
        help="Maximum number of training samples (-1 for all). Used for quick validation of the training pipeline."
    )

    parser.add_argument(
        "--max_eval_samples", type=int, default=-1,
        help="Maximum number of validation samples (-1 for all). Used for quick validation of the training pipeline."
    )

    parser.add_argument("--d_fme", type=int, default=1024, help="FME dimension")

    parser.add_argument("--fme_type", type=str, default="se", help="FME type")

    parser.add_argument("--base", type=int, default=1)

    parser.add_argument("--if_trainable", action="store_true", default=True)

    parser.add_argument("--translation_bias_type", type=str, default="nd")

    parser.add_argument("--emb_nn", action="store_true", default=True)

    parser.add_argument("--d_pe", type=int, default=1024, help="Positional encoding dimension")

    parser.add_argument("--if_index", action="store_true", default=True)

    parser.add_argument("--if_global_timing", action="store_true", default=True)

    parser.add_argument("--if_modulo_timing", action="store_false", default=False)

    parser.add_argument("--d_beat", type=int, default=1024, help="Beat embedding dimension")

    parser.add_argument("--d_oh_beat_type", type=int, default=7)

    parser.add_argument("--beat_len", type=int, default=50)

    parser.add_argument("--d_chord", type=int, default=1024, help="Chord embedding dimension")
    
    parser.add_argument("--d_oh_chord_type", type=int, default=12)

    parser.add_argument("--d_oh_inv_type", type=int, default=4)

    parser.add_argument("--chord_len", type=int, default=20)

    parser.add_argument(
        "--align_type", type = str, default = "dpo", choices = ["dpo", "ipo"],
        help = "Choose the alignment method for preference learning, DPO or IPO."
    )

    parser.add_argument(
        "--remove_beats", action="store_true", default=False,
        help="Remove beat encoder for ablation study."
    )
    parser.add_argument(
        "--remove_chords", action="store_true", default=False,
        help="Remove chord encoder for ablation study."
    )
    parser.add_argument(
        "--zeta", type=float, default=0.5,
        help="Weight for SFT loss in combined loss: zeta * sft_loss + (1-zeta) * dpo/ipo_loss."
    )

    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


class Text2AudioDataset(Dataset):
    """Standard dataset for validation and SFT stages."""
    def __init__(self, dataset, prefix, text_column, audio_column, beats_column, 
                 chords_column, chords_time_column, num_examples=-1):
        
        # Extract texts
        if text_column in dataset.column_names:
            inputs = list(dataset[text_column])
        elif "main_caption" in dataset.column_names:
            inputs = list(dataset["main_caption"])
        else:
            raise ValueError(f"Text column {text_column} or main_caption not found")

        self.inputs = [prefix + inp for inp in inputs]

        # Audio paths
        if audio_column in dataset.column_names:
            self.audios = list(dataset[audio_column])
        elif "location" in dataset.column_names:
            self.audios = list(dataset["location"])
        else:
            raise ValueError(f"Audio column {audio_column} or location not found")
        
    # Music feature parameters
        self.beats = []
        self.chords = []
        self.chords_time = []
        
        for i in range(len(inputs)):
            if beats_column and beats_column in dataset.column_names:
                beat_data = dataset[beats_column][i]
                self.beats.append(beat_data if isinstance(beat_data, list) else [[], []])
            else:
                self.beats.append([[], []])
            
            if chords_column and chords_column in dataset.column_names:
                chord_data = dataset[chords_column][i]
                self.chords.append(chord_data if isinstance(chord_data, list) else [])
            else:
                self.chords.append([])
            
            if chords_time_column and chords_time_column in dataset.column_names:
                chord_time_data = dataset[chords_time_column][i]
                self.chords_time.append(chord_time_data if isinstance(chord_time_data, list) else [])
            else:
                self.chords_time.append([])


        self.indices = list(range(len(self.inputs)))

        if num_examples != -1:
            self.inputs = self.inputs[:num_examples]
            self.audios = self.audios[:num_examples]
            self.beats = self.beats[:num_examples]
            self.chords = self.chords[:num_examples]
            self.chords_time = self.chords_time[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (self.inputs[index], self.audios[index], self.beats[index], 
                self.chords[index], self.chords_time[index], self.indices[index])


    def collate_fn(self, data):
        batch = list(zip(*data))
        texts = batch[0]
        audios = batch[1]
        beats = batch[2]
        chords = batch[3]
        chords_times = batch[4]
        indices = batch[5]
        return [texts, audios, beats, chords, chords_times, indices]

        
class DPOText2AudioDataset(Dataset):
    def __init__(self, dataset, prefix, text_column, audio_w_column, audio_l_column, 
    beats_column, chords_column, chords_time_column, num_examples=-1, dataset_dir = "~/.cache/huggingface/datasets"):

        if text_column in dataset.column_names:
            inputs = list(dataset[text_column])
        elif "main_caption" in dataset.column_names:
            inputs = list(dataset["main_caption"])
        else:
            raise ValueError(f"Text column {text_column} or main_caption not found")

        self.inputs = [prefix + inp for inp in inputs]

        if audio_w_column in dataset.column_names:
            self.audios_w = list(dataset[audio_w_column])
        else:
            self.audios_w = [f"{dataset_dir}/Specific_Prompt_music/{path}" for path in dataset["winner_location"]]
        
        if audio_l_column in dataset.column_names:
            self.audios_l = list(dataset[audio_l_column])
        else:
            self.audios_l = [f"{dataset_dir}/Specific_Prompt_music/{path}" for path in dataset["loser_location"]]

        self.beats = []
        self.chords = []
        self.chords_time = []
        
        for i in range(len(inputs)):
            # Parse beats format: [[timestamps], [beat_types]]
            if beats_column and beats_column in dataset.column_names:
                beat_data = dataset[beats_column][i]
                if isinstance(beat_data, list) and len(beat_data) == 2:
                    self.beats.append(beat_data)
                else:
                    self.beats.append([[], []])  # empty beats
            else:
                self.beats.append([[], []])
            
            # Parse chords
            if chords_column and chords_column in dataset.column_names:
                chord_data = dataset[chords_column][i]
                if isinstance(chord_data, list):
                    self.chords.append(chord_data)
                else:
                    self.chords.append([])
            else:
                self.chords.append([])
            
            # Parse chords
            if chords_time_column and chords_time_column in dataset.column_names:
                chord_time_data = dataset[chords_time_column][i]
                if isinstance(chord_time_data, list):
                    self.chords_time.append(chord_time_data)
                else:
                    self.chords_time.append([])
            else:
                self.chords_time.append([])

        self.indices = list(range(len(self.inputs)))


        if num_examples != -1:
            self.inputs, self.audios_w, self.audios_l = self.inputs[:num_examples], self.audios_w[:num_examples], self.audios_l[:num_examples]
            self.indices = self.indices[:num_examples]
            self.beats = self.beats[:num_examples]
            self.chords = self.chords[:num_examples]
            self.chords_time = self.chords_time[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (self.inputs[index], self.audios_w[index], self.audios_l[index], 
                self.beats[index], self.chords[index], self.chords_time[index], self.indices[index])

    def collate_fn(self, data):
        batch = list(zip(*data))
        
                # Handle regular fields
        texts = batch[0]
        audios_w = batch[1]
        audios_l = batch[2]
        indices = batch[6]
        
                # beats, chords, chords_time kept as lists (no stacking needed)
        beats = batch[3]
        chords = batch[4]
        chords_times = batch[5]
        
        return [texts, audios_w, audios_l, beats, chords, chords_times, indices]

        


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

  
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
     **accelerator_log_kwargs,
     mixed_precision = "bf16",    #TODO: fp16 causes NaN training loss; use bf16 for now.
     device_placement = True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)


    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        # os.environ["WANDB_DISABLED"] = "true"
        wandb.init(project="Text to Music Diffusion with DPO", mode = "offline")
        
        
        # logger.info(f"***** Writing audio file to {args.dataset_dir}/audio_alpaca *****")
        # if not os.path.exists(f"{args.dataset_dir}/audio_alpaca"):
        #     os.makedirs(f"{args.dataset_dir}/audio_alpaca")
            
        print("dataset has been loaded")

    accelerator.wait_for_everyone()
    
    audio_base_dir = args.dataset_dir


    
    # Get the datasets
    accelerator.print(f"Loading training data from JSONL: {args.train_file}")
    raw_dataset_train = load_custom_dataset(args.train_file)

    if args.max_train_samples > 0 and args.max_train_samples < len(raw_dataset_train):
        raw_dataset_train = raw_dataset_train.select(range(args.max_train_samples))
        accelerator.print(f"Limited to {args.max_train_samples} training samples")

    accelerator.print(f"Training data loaded: {len(raw_dataset_train)} samples")

        # Load validation and test sets. TODO: could refactor into a JSONL loader consistent with the training set
    accelerator.print(f"Loading validation data from JSONL: {args.validation_file}")
    raw_dataset_valid = load_eval_dataset(args.validation_file)
    if args.max_eval_samples > 0 and args.max_eval_samples < len(raw_dataset_valid):
        raw_dataset_valid = raw_dataset_valid.select(range(args.max_eval_samples))

    accelerator.print(f"Loading test data from JSONL: {args.test_file}")
    raw_dataset_test = load_eval_dataset(args.test_file)
    if args.max_eval_samples > 0 and args.max_eval_samples < len(raw_dataset_test):
        raw_dataset_test = raw_dataset_test.select(range(args.max_eval_samples))
   
 
    # Initialize models
    pretrained_model_name = "audioldm-s-full"
    vae, stft = build_pretrained_models(pretrained_model_name)
    vae.eval()
    stft.eval()

    use_beats = not args.remove_beats
    use_chords = not args.remove_chords

    if args.align_type == "dpo":
        accelerator.print("Using DPO alignment for training.")
        model = DPOMusicAudioDiffusion(
            text_encoder_name=args.text_encoder_name,
            scheduler_name=args.scheduler_name,
            unet_model_name=args.unet_model_name,
            unet_model_config_path=args.unet_model_config,
            snr_gamma=args.snr_gamma,
            freeze_text_encoder=args.freeze_text_encoder,
            uncondition=args.uncondition,
            beta_dpo=args.dpo_beta,
            zeta=args.zeta,
            # music feature parameters
            d_fme=args.d_fme,
            fme_type="se",
            base=1,
            if_trainable=True,
            translation_bias_type="nd",
            emb_nn=True,
            d_pe=args.d_pe,
            if_index=True,
            if_global_timing=True,
            if_modulo_timing=False,
            d_beat=args.d_beat,
            d_oh_beat_type=7,
            beat_len=50,
            d_chord=1024,
            d_oh_chord_type=12,
            d_oh_inv_type=4,
            chord_len=20,
            use_beats = use_beats,
            use_chords = use_chords
        )
    elif args.align_type == "ipo":
        accelerator.print("Using IPO alignment for training.")
        model = IPOMusicAudioDiffusion(
            text_encoder_name=args.text_encoder_name,
            scheduler_name=args.scheduler_name,
            unet_model_name=args.unet_model_name,
            unet_model_config_path=args.unet_model_config,
            snr_gamma=args.snr_gamma,
            freeze_text_encoder=args.freeze_text_encoder,
            uncondition=args.uncondition,
            beta_dpo=args.dpo_beta,
            eta_ipo=args.eta_ipo,
            zeta=args.zeta,
            # music feature parameters
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
        )
        

    if args.hf_model:
        local_model_path = f"{PROJECT_ROOT}/Model/huggingface/hub/models--mustango/ldm/pytorch_model_ldm.bin"
        # hf_model_path = snapshot_download(repo_id=args.hf_model)
        model.load_state_dict(torch.load(local_model_path, map_location="cpu"))
        accelerator.print("Successfully loaded checkpoint from:", local_model_path)
        
        # Freeze specified encoders
    if args.freeze_beat_encoder:
        for param in model.beat_embedding_layer.parameters():
            param.requires_grad = False
        accelerator.print("Frozen beat encoder parameters")
    
    if args.freeze_chord_encoder:
        for param in model.chord_embedding_layer.parameters():
            param.requires_grad = False
        for param in model.FME.parameters():
            param.requires_grad = False
        accelerator.print("Frozen chord encoder parameters")


    model.unet.enable_gradient_checkpointing()
    accelerator.print("Enabled gradient checkpointing for UNet.")
    
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""
        
    
    
        
    with accelerator.main_process_first():

        train_num_examples = args.max_train_samples if args.max_train_samples > 0 else args.num_examples
        train_dataset = DPOText2AudioDataset(
            raw_dataset_train, 
            prefix, 
            args.text_column_dpo,          
            args.audio_w_column,          
            args.audio_l_column,           
            beats_column=args.beats_column,       
            chords_column=args.chords_column,      
            chords_time_column=args.chords_time_column,  
            num_examples=train_num_examples,
            dataset_dir=args.dataset_dir
        ) 


        eval_num_examples = args.max_eval_samples if args.max_eval_samples > 0 else args.num_examples
        eval_dataset = Text2AudioDataset(
            raw_dataset_valid, 
            prefix, 
            args.text_column,         
            args.audio_column, 
            beats_column=args.beats_column,       
            chords_column=args.chords_column,      
            chords_time_column=args.chords_time_column,  
            num_examples=eval_num_examples  
        )
        
        test_num_examples = args.max_eval_samples if args.max_eval_samples > 0 else args.num_examples
        test_dataset = Text2AudioDataset(
            raw_dataset_test, 
            prefix, 
            args.text_column, 
            args.audio_column, 
            beats_column=args.beats_column,        
            chords_column=args.chords_column,      
            chords_time_column=args.chords_time_column,  
            num_examples=test_num_examples
        )
        
        accelerator.print("Num instances in train: {}, validation: {}, test: {}".format(train_dataset.get_num_instances(), eval_dataset.get_num_instances(), test_dataset.get_num_instances()))
        
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=eval_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=test_dataset.collate_fn)
    
   
    # Initialize reference Unet
    model.unet.set_attn_processor(AttnProcessor2_0())
    accelerator.print("�?Enabled PyTorch 2.0 scaled_dot_product_attention for UNet")
    ref_unet = copy.deepcopy(model.unet)
    ref_unet.requires_grad_(False)
    ref_unet.to(accelerator.device)
        
    if args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            model.text_encoder.eval()
        if args.unet_model_config:
            optimizer_parameters = model.unet.parameters()
            accelerator.print("Optimizing UNet parameters.")
        else:
            optimizer_parameters = list(model.unet.parameters()) + list(model.group_in.parameters()) + list(model.group_out.parameters())
            accelerator.print("Optimizing UNet and channel transformer parameters.")
    else:
        optimizer_parameters = model.parameters()
        accelerator.print("Optimizing Text Encoder and UNet parameters.")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in ref_unet.parameters() if p.requires_grad)
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))
    
    # Optimizer 
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs+args.sft_first_epochs) * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, stft, model, optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"*****  Running {args.align_type} *****")
    logger.info(f" Learning rate = {args.learning_rate}")
    logger.info(f" Number of warmup steps = {args.num_warmup_steps}")
    logger.info(f" DPO beta value = {args.dpo_beta}")
    logger.info(f" IPO eta value = {args.eta_ipo}")
    logger.info(f" Doing {args.sft_first_epochs} epoch of SFT first")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint)
            # path = os.path.basename(args.resume_from_checkpoint)
            accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            
    # Duration of the audio clips in seconds
    duration, best_loss = 10, np.inf
    
                
    sft_epochs = args.sft_first_epochs
    #adaptive beta parameters
    ema_diff = 0.0
    target_diff_ema = args.target_diff_ema   #keep (model_diff-ref_diff) around 0.1
    beta_adjust_lr = args.beta_adjust_lr    #beta lr
    integral_error = 0.0
    beta_min, beta_max = args.dpo_beta_min, args.dpo_beta_max
   
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):
            # Each batch contains the prompt, chosen audio, reject audio

            with accelerator.accumulate(model):
                device = accelerator.device
                target_length = int(duration * 102.4)


                if len(batch) == 7:  # DPO dataset with music features
                    text, audio_w, audio_l, beats, chords, chords_time, _ = batch
                    has_music_features = True
                elif len(batch) == 4:  # Original DPO dataset (no music features)
                    text, audio_w, audio_l, _ = batch
                    beats, chords, chords_time = None, None, None
                    has_music_features = False
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                
                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    mel_w, _, waveform_w = torch_tools.wav_to_fbank(audio_w, target_length, stft)
                    mel_l, _, waveform_l = torch_tools.wav_to_fbank(audio_l, target_length, stft)
                    mel_w = mel_w.unsqueeze(1).to(device)
                    mel_l = mel_l.unsqueeze(1).to(device)
                    
                    
                    #true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))
                    latent_w =  unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel_w))
                    latent_l =  unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel_l))
                    
                if sft_epochs > 0: 
                    # Perform SFT on the prompt and preferred audio if sft_epochs > 0
                    
                    latents = latent_w
                    if has_music_features:
                        loss = model(latents, text, ref_unet = ref_unet, validation_mode=False , sft = True, beats=beats, chords=chords, chords_time=chords_time)
                    else:
                        loss = model(latents, text, ref_unet = ref_unet, validation_mode=False , sft = True)
                    
                else:
                    latents = torch.cat((latent_w,latent_l),dim=0)
                    # ref_unet = ref_unet.to(device)
                    # ref_unet.eval()

                    if has_music_features:
                        beats_dpo = beats
                        chords_dpo = chords
                        chords_time_dpo = chords_time
                        loss, diff = model(
                            latents, text, beats_dpo, chords_dpo, chords_time_dpo,
                            ref_unet=ref_unet, validation_mode=False, sft=False,
                            return_diff = True
                        )
                        diff_tensor = torch.tensor([diff], device = device)
                        gathered_diffs = accelerator.gather(diff_tensor)   # shape: (world_size,)
                        mean_diff = gathered_diffs.mean().item()

                        ema_diff = 0.9 * ema_diff + 0.1 * mean_diff
                        # if abs(ema_diff) > 1e-8:
                        #     beta_factor = target_diff_ema / abs(ema_diff)   # TODO: unclear logic, needs review
                        # else:
                        #     beta_factor = 1.0
                        #     # Smoothly update beta to avoid oscillation
                        unwrapped_model = accelerator.unwrap_model(model)
                        if completed_steps >=4:
                            old_beta = unwrapped_model.beta_dpo
                            error = abs(ema_diff) - target_diff_ema
                            integral_error = 0.9 * integral_error + 0.1 * error
                            delta_beta = 2.0 * error + 0.1 * integral_error
                            new_beta = old_beta + delta_beta * beta_adjust_lr 
                            new_beta = max(beta_min, min(beta_max, new_beta))
                        else:
                            new_beta = unwrapped_model.beta_dpo

                        unwrapped_model.beta_dpo = new_beta
                        
                    else:
                        loss = model(latents, text, ref_unet = ref_unet, validation_mode=False , sft = False)
                            
                    # ref_unet = ref_unet.to("cpu")
                    torch.cuda.empty_cache()
                    
                
            
    
                total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if (completed_steps+1) % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps+1 }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        


        model.eval()
        model.uncondition = False
        

        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            
            with accelerator.accumulate(model) and torch.no_grad():
                # device = model.device
                device = model.text_encoder.device #
                # text, audios, _ = batch
                target_length = int(duration * 102.4)

                # Parse validation batch
                if len(batch) == 6:  # with music features
                    text, audios, beats, chords, chords_time, _ = batch
                    has_music_features = True
                elif len(batch) == 3:  # without music features
                    text, audios, _ = batch
                    beats, chords, chords_time = None, None, None
                    has_music_features = False
                else:
                    raise ValueError(f"Unexpected eval batch size: {len(batch)}")

                unwrapped_vae = accelerator.unwrap_model(vae)
                
                mel, _, waveform = torch_tools.wav_to_fbank(audios, target_length, stft)
                mel = mel.unsqueeze(1).to(device)
                true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

                if has_music_features:
                    val_loss = accelerator.unwrap_model(model).diffusion_forward(
                        true_latent, text, beats, chords, chords_time, validation_mode=True
                    )
                else:
                    val_loss = accelerator.unwrap_model(model).diffusion_forward(
                        true_latent, text, validation_mode=True
                    )

                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        model.uncondition = args.uncondition

        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1,
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
            result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
            result["mode"] = "SFT" if sft_epochs > 0 else args.align_type.upper()
            sft_epochs -= 1
            wandb.log(result)

            result_string = "Epoch: {}, Mode: {}, Loss Train: {}, Val: {}\n".format(epoch, result["mode"], result["train_loss"], result["val_loss"])
            
            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["val_loss"] < best_loss:
                best_loss = result["val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False
    
        
        if args.with_tracking:
            accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()
        
        if epoch >= args.sft_first_epochs:  # Only start checkpointing after SFT epochs
            if accelerator.is_main_process and args.checkpointing_steps == "best":
                if save_checkpoint:
                    accelerator.save_state("{}/{}".format(args.output_dir, "best"))
                    
                if (epoch + 1) % args.save_every == 0 and sft_epochs < 0: ## Only save state after doing sft_first
                    accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))

            if accelerator.is_main_process and args.checkpointing_steps == "epoch" and sft_epochs < 0: ## Only save state after doing sft_first
                accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))
                
            
            
if __name__ == "__main__":
    main()
