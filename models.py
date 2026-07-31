import yaml
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from tools.torch_tools import wav_to_fbank

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata

from transformers import CLIPTokenizer, AutoTokenizer
from transformers import CLIPTextModel, T5EncoderModel, AutoModel

import diffusers
from diffusers.utils import randn_tensor
from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DConditionModelMusic
from diffusers import AutoencoderKL as DiffuserAutoencoderKL


def build_pretrained_models(name):
    # checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
    ckpt_path = "/path/to/your/models/Music/pretrained/audioldm-s-full" #
    checkpoint = torch.load(ckpt_path, map_location = "cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae.eval()
    fn_STFT.eval()
    return vae, fn_STFT


class AudioDiffusion(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,

    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.freeze_text_encoder = freeze_text_encoder
        self.uncondition = uncondition

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")


        if unet_model_config_path:
            unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
            self.unet = UNet2DConditionModel.from_config(unet_config, subfolder="unet")
            self.set_from = "random"
            print("UNet initialized randomly.")
        else:
            self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
            self.set_from = "pre-trained"
            self.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
            self.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))
            print("UNet initialized from stable diffusion checkpoint.")

        if "stable-diffusion" in self.text_encoder_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name, subfolder="text_encoder")
        elif "t5" in self.text_encoder_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def forward(self, latents, prompt, validation_mode=False):
        device = self.text_encoder.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)

        if self.uncondition:
            mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.set_from == "random":
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            ).sample

        elif self.set_from == "pre-trained":
            compressed_latents = self.group_in(noisy_latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            model_pred = self.unet(
                compressed_latents, timesteps, encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            ).sample
            model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def inference(self, prompt, inference_scheduler, num_steps=20, guidance_scale=3, num_samples_per_prompt=1,
                  disable_progress=True):
        device = self.text_encoder.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt

        if classifier_free_guidance:
            prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
        else:
            prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask
            ).sample

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        if self.set_from == "pre-trained":
            latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 256, 16)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask


from layers.layers import chord_tokenizer, beat_tokenizer, Chord_Embedding, Beat_Embedding, Music_PositionalEncoding, Fundamental_Music_Embedding

class MusicAudioDiffusion(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,

        d_fme = 1024,  #FME
        fme_type = "se",
        base = 1,
        if_trainable = True,
        translation_bias_type = "nd",
        emb_nn = True,
        d_pe = 1024, #PE
        if_index = True,
        if_global_timing = True,
        if_modulo_timing = False,
        d_beat = 1024, #Beat
        d_oh_beat_type = 7,
        beat_len = 50,
        d_chord = 1024, #Chord
        d_oh_chord_type = 12,
        d_oh_inv_type = 4,
        chord_len = 20,
        use_beats = True,
        use_chords = True
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.text_encoder_name = text_encoder_name
        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.freeze_text_encoder = freeze_text_encoder
        self.uncondition = uncondition
        self.use_beats = use_beats
        self.use_chords = use_chords

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")


        if unet_model_config_path:
            unet_config = UNet2DConditionModelMusic.load_config(unet_model_config_path)
            self.unet = UNet2DConditionModelMusic.from_config(unet_config, subfolder="unet")
            self.set_from = "random"
            print("UNet initialized randomly.")
        else:
            self.unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
            self.set_from = "pre-trained"
            self.group_in = nn.Sequential(nn.Linear(8, 512), nn.Linear(512, 4))
            self.group_out = nn.Sequential(nn.Linear(4, 512), nn.Linear(512, 8))
            print("UNet initialized from stable diffusion checkpoint.")

        if "stable-diffusion" in self.text_encoder_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name, subfolder="text_encoder")
        elif "t5" in self.text_encoder_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)

        self.device = self.text_encoder.device
        #Music Feature Encoder
        self.FME = Fundamental_Music_Embedding(d_model = d_fme, base= base, if_trainable = False, type = fme_type,emb_nn=emb_nn,translation_bias_type = translation_bias_type)
        self.PE = Music_PositionalEncoding(d_model = d_pe, if_index = if_index, if_global_timing = if_global_timing, if_modulo_timing = False, device = self.device)
        # self.PE2 = Music_PositionalEncoding(d_model = d_pe, if_index = if_index, if_global_timing = if_global_timing, if_modulo_timing = False, device = self.device)
        self.beat_tokenizer = beat_tokenizer(seq_len_beat=beat_len, if_pad = True)
        self.beat_embedding_layer = Beat_Embedding(self.PE, d_model = d_beat, d_oh_beat_type = d_oh_beat_type)
        self.chord_embedding_layer = Chord_Embedding(self.FME, self.PE, d_model = d_chord, d_oh_type = d_oh_chord_type, d_oh_inv = d_oh_inv_type)
        self.chord_tokenizer = chord_tokenizer(seq_len_chord=chord_len, if_pad = True)

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        # Debug code for NaN detection: NaN occurs at the residual connection in the last layer of T5 encoder
        # with torch.cuda.amp.autocast(enabled=True): # Disable automatic mixed precision; this caused NaN
        if self.freeze_text_encoder: # TODO
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_beats(self, beats):
        # device = self.beat_embedding_layer.device
        out_beat = []
        out_beat_timing = []
        out_mask = []
        for beat in beats:
            tokenized_beats,tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer(beat)
            out_beat.append(tokenized_beats)
            out_beat_timing.append(tokenized_beats_timing)
            out_mask.append(tokenized_beat_mask)
        out_beat, out_beat_timing, out_mask = torch.tensor(out_beat).cuda(), torch.tensor(out_beat_timing).cuda(), torch.tensor(out_mask).cuda() #batch, len_beat
        embedded_beat = self.beat_embedding_layer(out_beat, out_beat_timing)

        return embedded_beat, out_mask

    def encode_chords(self, chords,chords_time):
        out_chord_root = []
        out_chord_type = []
        out_chord_inv = []
        out_chord_timing = []
        out_mask = []
        for chord, chord_time in zip(chords,chords_time): #batch loop
            tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer(chord, chord_time)
            out_chord_root.append(tokenized_chord_root)
            out_chord_type.append(tokenized_chord_type)
            out_chord_inv.append(tokenized_chord_inv)
            out_chord_timing.append(tokenized_chord_time)
            out_mask.append(tokenized_chord_mask)
        #chords: (B, LEN, 4)
        out_chord_root, out_chord_type, out_chord_inv, out_chord_timing, out_mask = torch.tensor(out_chord_root).cuda(), torch.tensor(out_chord_type).cuda(), torch.tensor(out_chord_inv).cuda(), torch.tensor(out_chord_timing).cuda(), torch.tensor(out_mask).cuda()
        embedded_chord = self.chord_embedding_layer(out_chord_root, out_chord_type, out_chord_inv, out_chord_timing)
        return embedded_chord, out_mask
        # return out_chord_root, out_mask

    def forward(self, latents, prompt, beats, chords,chords_time, validation_mode=False):
        device = self.text_encoder.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)

        if self.use_beats:
        # with torch.no_grad():
            encoded_beats, beat_mask = self.encode_beats(beats) #batch, len_beats, dim; batch, len_beats
        else:
            # print("Freeze beats encoder")
            bsz = latents.shape[0]
            beat_len = self.beat_tokenizer.seq_len_beat  # assume tokenizer has this attribute
            d_beat = self.beat_embedding_layer.d_model
            encoded_beats = torch.zeros(bsz, beat_len, d_beat, device=device)
            beat_mask = torch.zeros(bsz, beat_len, device=device)

        if self.use_chords:
            encoded_chords, chord_mask = self.encode_chords(chords,chords_time)
        else:
            # print("Freeze chords encoder")
            bsz = latents.shape[0]
            chord_len = self.chord_tokenizer.seq_len_chord
            d_chord = self.chord_embedding_layer.d_model
            encoded_chords = torch.zeros(bsz, chord_len, d_chord, device=device)
            chord_mask = torch.zeros(bsz, chord_len, device=device)

        if self.uncondition:
            mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0
                encoded_chords[mask_indices] = 0
                encoded_beats[mask_indices] = 0

        bsz = latents.shape[0]

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)

        timesteps = timesteps.long()

        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.set_from == "random":
            # model_pred = torch.zeros((bsz,8,256,16)).to(device)
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states, encoded_beats, encoded_chords,
                encoder_attention_mask=boolean_encoder_mask, beat_attention_mask = beat_mask, chord_attention_mask = chord_mask
            ).sample

        elif self.set_from == "pre-trained":
            compressed_latents = self.group_in(noisy_latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            model_pred = self.unet(
                compressed_latents, timesteps, encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            ).sample
            model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def inference(self, prompt, beats, chords,chords_time, inference_scheduler, num_steps=20, guidance_scale=3, num_samples_per_prompt=1,
                  disable_progress=True):
        device = self.text_encoder.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) * num_samples_per_prompt

        if classifier_free_guidance:
            prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)

            if self.use_beats:
                encoded_beats, beat_mask = self.encode_beats_classifier_free(beats, num_samples_per_prompt) #batch, len_beats, dim; batch, len_beats
            else:
                encoded_beats = torch.zeros(
                    batch_size * 2,
                    self.beat_tokenizer.seq_len_beat,
                    self.beat_embedding_layer.d_model,
                    device=device
                )
                beat_mask = torch.zeros(
                    batch_size * 2,
                    self.beat_tokenizer.seq_len_beat,
                    device=device
                )

            if self.use_chords:
                encoded_chords, chord_mask = self.encode_chords_classifier_free(chords, chords_time, num_samples_per_prompt)
            else:
                encoded_chords = torch.zeros(
                    batch_size * 2,
                    self.chord_tokenizer.seq_len_chord,
                    self.chord_embedding_layer.d_model,
                    device=device
                )
                chord_mask = torch.zeros(
                    batch_size * 2,
                    self.chord_tokenizer.seq_len_chord,
                    device=device
                )
        else:
            prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
            boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

            if self.use_beats:
                encoded_beats, beat_mask = self.encode_beats(beats)
                encoded_beats = encoded_beats.repeat_interleave(num_samples_per_prompt, 0)
                beat_mask = beat_mask.repeat_interleave(num_samples_per_prompt, 0)
            else:
                encoded_beats = torch.zeros(
                    batch_size,
                    self.beat_tokenizer.seq_len_beat,
                    self.beat_embedding_layer.d_model,
                    device=device
                )
                beat_mask = torch.zeros(
                    batch_size,
                    self.beat_tokenizer.seq_len_beat,
                    device=device
                )

            if self.use_chords:
                encoded_chords, chord_mask = self.encode_chords(chords, chords_time)
                encoded_chords = encoded_chords.repeat_interleave(num_samples_per_prompt, 0)
                chord_mask = chord_mask.repeat_interleave(num_samples_per_prompt, 0)
            else:
                encoded_chords = torch.zeros(
                    batch_size,
                    self.chord_tokenizer.seq_len_chord,
                    self.chord_embedding_layer.d_model,
                    device=device
                )
                chord_mask = torch.zeros(
                    batch_size,
                    self.chord_tokenizer.seq_len_chord,
                    device=device
                )

        # print(f"encoded_chords:{encoded_chords.shape}, chord_mask:{chord_mask.shape}, prompt_embeds:{prompt_embeds.shape},boolean_prompt_mask:{boolean_prompt_mask.shape} ")
        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask,
                beat_features = encoded_beats, beat_attention_mask = beat_mask, chord_features = encoded_chords,chord_attention_mask = chord_mask
            ).sample

            # perform guidance
            if classifier_free_guidance: #should work for beats and chords too
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        if self.set_from == "pre-trained":
            latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return latents

    def prepare_latents(self, batch_size, inference_scheduler, num_channels_latents, dtype, device):
        shape = (batch_size, num_channels_latents, 256, 16)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents

    def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():

            prompt_embeds = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # get unconditional embeddings for classifier free guidance
        # print(len(prompt), 'this is prompt len')
        uncond_tokens = [""] * len(prompt)

        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)

        return prompt_embeds, boolean_prompt_mask

    def encode_beats_classifier_free(self, beats, num_samples_per_prompt):
        with torch.no_grad():
            out_beat = []
            out_beat_timing = []
            out_mask = []
            for beat in beats:
                tokenized_beats,tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer(beat)
                out_beat.append(tokenized_beats)
                out_beat_timing.append(tokenized_beats_timing)
                out_mask.append(tokenized_beat_mask)
            out_beat, out_beat_timing, out_mask = torch.tensor(out_beat).cuda(), torch.tensor(out_beat_timing).cuda(), torch.tensor(out_mask).cuda() #batch, len_beat
            embedded_beat = self.beat_embedding_layer(out_beat, out_beat_timing)

        embedded_beat = embedded_beat.repeat_interleave(num_samples_per_prompt, 0)
        out_mask = out_mask.repeat_interleave(num_samples_per_prompt, 0)

        uncond_beats = [[[],[]]] * len(beats)

        max_length = embedded_beat.shape[1]
        with torch.no_grad():
            out_beat_unc = []
            out_beat_timing_unc = []
            out_mask_unc = []
            for beat in uncond_beats:
                tokenized_beats, tokenized_beats_timing, tokenized_beat_mask = self.beat_tokenizer(beat)
                out_beat_unc.append(tokenized_beats)
                out_beat_timing_unc.append(tokenized_beats_timing)
                out_mask_unc.append(tokenized_beat_mask)
            out_beat_unc, out_beat_timing_unc, out_mask_unc = torch.tensor(out_beat_unc).cuda(), torch.tensor(out_beat_timing_unc).cuda(), torch.tensor(out_mask_unc).cuda() #batch, len_beat
            embedded_beat_unc = self.beat_embedding_layer(out_beat_unc, out_beat_timing_unc)

        embedded_beat_unc = embedded_beat_unc.repeat_interleave(num_samples_per_prompt, 0)
        out_mask_unc = out_mask_unc.repeat_interleave(num_samples_per_prompt, 0)

        embedded_beat = torch.cat([embedded_beat_unc, embedded_beat])
        out_mask = torch.cat([out_mask_unc, out_mask])

        return embedded_beat, out_mask

    def encode_chords_classifier_free(self, chords, chords_time, num_samples_per_prompt):
        with torch.no_grad():
            out_chord_root = []
            out_chord_type = []
            out_chord_inv = []
            out_chord_timing = []
            out_mask = []
            for chord, chord_time in zip(chords,chords_time): #batch loop
                tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer(chord, chord_time)
                out_chord_root.append(tokenized_chord_root)
                out_chord_type.append(tokenized_chord_type)
                out_chord_inv.append(tokenized_chord_inv)
                out_chord_timing.append(tokenized_chord_time)
                out_mask.append(tokenized_chord_mask)
            out_chord_root, out_chord_type, out_chord_inv, out_chord_timing, out_mask = torch.tensor(out_chord_root).cuda(), torch.tensor(out_chord_type).cuda(), torch.tensor(out_chord_inv).cuda(), torch.tensor(out_chord_timing).cuda(), torch.tensor(out_mask).cuda()
            embedded_chord = self.chord_embedding_layer(out_chord_root, out_chord_type, out_chord_inv, out_chord_timing)

        embedded_chord = embedded_chord.repeat_interleave(num_samples_per_prompt, 0)
        out_mask = out_mask.repeat_interleave(num_samples_per_prompt, 0)

        chords_unc=[[]] * len(chords)
        chords_time_unc=[[]] * len(chords_time)

        max_length = embedded_chord.shape[1]

        with torch.no_grad():
            out_chord_root_unc = []
            out_chord_type_unc = []
            out_chord_inv_unc = []
            out_chord_timing_unc = []
            out_mask_unc = []
            for chord, chord_time in zip(chords_unc,chords_time_unc): #batch loop
                tokenized_chord_root, tokenized_chord_type, tokenized_chord_inv, tokenized_chord_time, tokenized_chord_mask = self.chord_tokenizer(chord, chord_time)
                out_chord_root_unc.append(tokenized_chord_root)
                out_chord_type_unc.append(tokenized_chord_type)
                out_chord_inv_unc.append(tokenized_chord_inv)
                out_chord_timing_unc.append(tokenized_chord_time)
                out_mask_unc.append(tokenized_chord_mask)
            out_chord_root_unc, out_chord_type_unc, out_chord_inv_unc, out_chord_timing_unc, out_mask_unc = torch.tensor(out_chord_root_unc).cuda(), torch.tensor(out_chord_type_unc).cuda(), torch.tensor(out_chord_inv_unc).cuda(), torch.tensor(out_chord_timing_unc).cuda(), torch.tensor(out_mask_unc).cuda()
            embedded_chord_unc = self.chord_embedding_layer(out_chord_root_unc, out_chord_type_unc, out_chord_inv_unc, out_chord_timing_unc)

        embedded_chord_unc = embedded_chord_unc.repeat_interleave(num_samples_per_prompt, 0)
        out_mask_unc = out_mask_unc.repeat_interleave(num_samples_per_prompt, 0)

        embedded_chord = torch.cat([embedded_chord_unc, embedded_chord])
        out_mask = torch.cat([out_mask_unc, out_mask])

        return embedded_chord, out_mask


# Inherit music diffusion model, add DPO training logic
class DPOMusicAudioDiffusion(MusicAudioDiffusion):
    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
        beta_dpo=2000,
        zeta=0.2,
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
        use_chords = True
    ):

        super().__init__(
            text_encoder_name=text_encoder_name,
            scheduler_name=scheduler_name,
            unet_model_name=unet_model_name,
            unet_model_config_path=unet_model_config_path,
            snr_gamma=snr_gamma,
            freeze_text_encoder=freeze_text_encoder,
            uncondition=uncondition,
            d_fme=d_fme,
            fme_type=fme_type,
            base=base,
            if_trainable=if_trainable,
            translation_bias_type=translation_bias_type,
            emb_nn=emb_nn,
            d_pe=d_pe,
            if_index=if_index,
            if_global_timing=if_global_timing,
            if_modulo_timing=if_modulo_timing,
            d_beat=d_beat,
            d_oh_beat_type=d_oh_beat_type,
            beat_len=beat_len,
            d_chord=d_chord,
            d_oh_chord_type=d_oh_chord_type,
            d_oh_inv_type=d_oh_inv_type,
            chord_len=chord_len,
            use_beats=use_beats,
            use_chords=use_chords
        )

        # DPO training hyperparameters
        self.beta_dpo = beta_dpo
        self.zeta = zeta

    def forward(self, latents, prompt, beats, chords, chords_time, ref_unet=None, validation_mode=False, sft=False, return_diff=False):
        """
        DPO forward pass for music diffusion model

        Args:
            latents: Latent representations (batch, channels, height, width)
            prompt: Text prompts
            beats: Beat features
            chords: Chord features
            chords_time: Chord timing information
            ref_unet: Reference UNet model for DPO (required when sft=False)
            validation_mode: Whether in validation mode
            sft: If True, use standard supervised fine-tuning (MSE loss)
                 If False, use DPO loss with reference model
        """

        # SFT mode: standard diffusion training
        if sft:
            return super().forward(
                latents=latents,
                prompt=prompt,
                beats=beats,
                chords=chords,
                chords_time=chords_time,
                validation_mode=validation_mode
            )

        # DPO mode: Direct Preference Optimization
        # Ensure reference model is provided
        if ref_unet is None:
            raise ValueError("ref_unet must be provided when sft=False (DPO mode)")

        device = self.text_encoder.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        # Encode inputs: text, beats, chords
        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt) # zhanle

        bsz = len(prompt)
        if self.use_beats:
            encoded_beats, beat_mask = self.encode_beats(beats)
        else:
            beat_len = self.beat_tokenizer.seq_len_beat
            d_beat = self.beat_embedding_layer.d_model
            encoded_beats = torch.zeros(bsz, beat_len, d_beat, device=device)
            beat_mask = torch.zeros(bsz, beat_len, device=device)

        if self.use_chords:
            encoded_chords, chord_mask = self.encode_chords(chords, chords_time)
        else:
            chord_len = self.chord_tokenizer.seq_len_chord
            d_chord = self.chord_embedding_layer.d_model
            encoded_chords = torch.zeros(bsz, chord_len, d_chord, device=device)
            chord_mask = torch.zeros(bsz, chord_len, device=device)

        # Duplicate for chosen and rejected samples (assuming latents contains [chosen, rejected])
        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
        boolean_encoder_mask = boolean_encoder_mask.repeat(2, 1)
        encoded_beats = encoded_beats.repeat(2, 1, 1)
        beat_mask = beat_mask.repeat(2, 1)
        encoded_chords = encoded_chords.repeat(2, 1, 1)
        chord_mask = chord_mask.repeat(2, 1)

        # Unconditional dropout (optional)
        if self.uncondition:
            batch_size_2x = len(prompt) * 2
            mask_indices = [k for k in range(batch_size_2x) if random.random() < 0.1]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0
                encoded_beats[mask_indices] = 0
                encoded_chords[mask_indices] = 0

        bsz = latents.shape[0] // 2

        # Sample timesteps
        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps) * torch.ones((bsz,), dtype=torch.int64, device=device).repeat(2)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.int64
            ).repeat(2)

        timesteps = timesteps.long()

        # Add noise to latents (use same noise for chosen and rejected)
        noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)
        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Target based on prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Forward pass through current model
        if self.set_from == "random":
            model_pred = self.unet(
                noisy_model_input, timesteps, encoder_hidden_states, encoded_beats, encoded_chords,
                encoder_attention_mask=boolean_encoder_mask,
                beat_attention_mask=beat_mask,
                chord_attention_mask=chord_mask
            ).sample
            # print("set_from random")
        else:  # pre-trained mode with group_in/group_out
            compressed_latents = self.group_in(noisy_model_input.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            model_pred = self.unet(
                compressed_latents, timesteps, encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask,
                beat_attention_mask=beat_mask,
                chord_attention_mask=chord_mask
            ).sample
            print("set_from pre-trained")
            model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        # Calculate model losses for chosen and rejected
        model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
        model_losses_w, model_losses_l = model_losses.chunk(2)  # chosen (winning) vs rejected (losing)
        model_diff = model_losses_w - model_losses_l


        # Forward pass through reference model (no gradients)
        with torch.no_grad():
            if self.set_from == "random":
                ref_preds = ref_unet(
                    noisy_model_input, timesteps, encoder_hidden_states, encoded_beats, encoded_chords,
                    encoder_attention_mask=boolean_encoder_mask,
                    beat_attention_mask=beat_mask,
                    chord_attention_mask=chord_mask
                ).sample
            else:  # pre-trained mode
                # Note: ref_unet should have group_in/group_out layers if it's also pre-trained
                # Assuming ref_unet has the same structure
                if hasattr(ref_unet, 'group_in') and hasattr(ref_unet, 'group_out'):
                    ref_compressed = ref_unet.group_in(noisy_model_input.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
                    ref_preds = ref_unet.unet(
                        ref_compressed, timesteps, encoder_hidden_states,
                        encoder_attention_mask=boolean_encoder_mask,
                        beat_attention_mask=beat_mask,
                        chord_attention_mask=chord_mask
                    ).sample
                    ref_preds = ref_unet.group_out(ref_preds.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
                else:
                    # Fallback: direct forward through ref_unet (might not work for pre-trained mode)
                    ref_preds = ref_unet(
                        noisy_model_input, timesteps, encoder_hidden_states,
                        encoder_attention_mask=boolean_encoder_mask,
                        beat_attention_mask=beat_mask,
                        chord_attention_mask=chord_mask
                    ).sample

            # Calculate reference model losses
            ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
            ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))
            ref_losses_w, ref_losses_l = ref_loss.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l

        # DPO loss (Equation from https://arxiv.org/abs/2311.12908)
        scale_term = -0.5 * self.beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        loss_dpo = -F.logsigmoid(inside_term).mean()

        #new add SFT_loss
        loss_sft = model_losses_w.mean()

        loss = loss_sft + self.zeta * loss_dpo

        if return_diff:
            print(f"Model diff: {model_diff.mean().item():.4f}, Ref diff: {ref_diff.mean().item():.4f}")
            print(f"Model loss w: {model_losses_w.mean().item():.4f}, l: {model_losses_l.mean().item():.4f},loss_sft: {loss_sft.item():.4f}, loss_dpo: {loss_dpo.item():.4f}")
            diff_val = (model_diff - ref_diff).mean().detach()
            return loss, diff_val
        else:
            return loss

    def diffusion_forward(self, latents, prompt, beats, chords, chords_time, validation_mode=False):
        """
        Standard diffusion forward pass (without DPO loss)
        Useful for validation/evaluation
        """
        return super().forward(
            latents=latents,
            prompt=prompt,
            beats=beats,
            chords=chords,
            chords_time=chords_time,
            validation_mode=validation_mode
        )


class IPOMusicAudioDiffusion(MusicAudioDiffusion):
    def __init__(
        self,
        text_encoder_name,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        freeze_text_encoder=True,
        uncondition=False,
        beta_dpo=2000,
        eta_ipo = 0.5,
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
        zeta=0.2
    ):

        super().__init__(
            text_encoder_name=text_encoder_name,
            scheduler_name=scheduler_name,
            unet_model_name=unet_model_name,
            unet_model_config_path=unet_model_config_path,
            snr_gamma=snr_gamma,
            freeze_text_encoder=freeze_text_encoder,
            uncondition=uncondition,
            d_fme=d_fme,
            fme_type=fme_type,
            base=base,
            if_trainable=if_trainable,
            translation_bias_type=translation_bias_type,
            emb_nn=emb_nn,
            d_pe=d_pe,
            if_index=if_index,
            if_global_timing=if_global_timing,
            if_modulo_timing=if_modulo_timing,
            d_beat=d_beat,
            d_oh_beat_type=d_oh_beat_type,
            beat_len=beat_len,
            d_chord=d_chord,
            d_oh_chord_type=d_oh_chord_type,
            d_oh_inv_type=d_oh_inv_type,
            chord_len=chord_len,
            use_beats = use_beats,
            use_chords = use_chords,
        )

        # DPO training hyperparameters
        self.beta_dpo = beta_dpo
        # IPO training hyperparameters
        self.eta_ipo = eta_ipo
        self.zeta = zeta

    def forward(self, latents, prompt, beats, chords, chords_time, ref_unet=None, validation_mode=False, sft=False, return_diff=False):
        """
        IPO forward pass for music diffusion model

        Args:
            latents: Latent representations (batch, channels, height, width)
            prompt: Text prompts
            beats: Beat features
            chords: Chord features
            chords_time: Chord timing information
            ref_unet: Reference UNet model for IPO (required when sft=False)
            validation_mode: Whether in validation mode
            sft: If True, use standard supervised fine-tuning (MSE loss)
                 If False, use IPO loss with reference model
        """

        # SFT mode: standard diffusion training
        if sft:
            return super().forward(
                latents=latents,
                prompt=prompt,
                beats=beats,
                chords=chords,
                chords_time=chords_time,
                validation_mode=validation_mode
            )

        # IPO mode: Implicit Preference Optimization
        # Ensure reference model is provided

        if ref_unet is None:
            raise ValueError("ref_unet must be provided when sft=False (IPO mode)")

        device = self.text_encoder.device
        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        # Encode inputs: text, beats, chords
        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)

        bsz = len(prompt)
        if self.use_beats:
            encoded_beats, beat_mask = self.encode_beats(beats)
        else:
            beat_len = self.beat_tokenizer.seq_len_beat
            d_beat = self.beat_embedding_layer.d_model
            encoded_beats = torch.zeros(bsz, beat_len, d_beat, device=device)
            beat_mask = torch.zeros(bsz, beat_len, device=device)

        if self.use_chords:
            encoded_chords, chord_mask = self.encode_chords(chords, chords_time)
        else:
            chord_len = self.chord_tokenizer.seq_len_chord
            d_chord = self.chord_embedding_layer.d_model
            encoded_chords = torch.zeros(bsz, chord_len, d_chord, device=device)
            chord_mask = torch.zeros(bsz, chord_len, device=device)

        # Duplicate for chosen and rejected samples (assuming latents contains [chosen, rejected])
        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
        boolean_encoder_mask = boolean_encoder_mask.repeat(2, 1)
        encoded_beats = encoded_beats.repeat(2, 1, 1)
        beat_mask = beat_mask.repeat(2, 1)
        encoded_chords = encoded_chords.repeat(2, 1, 1)
        chord_mask = chord_mask.repeat(2, 1)

        # Unconditional dropout (optional)
        if self.uncondition:
            batch_size_2x = len(prompt) * 2
            mask_indices = [k for k in range(batch_size_2x) if random.random() < 0.1]
            if len(mask_indices) > 0:
                encoder_hidden_states[mask_indices] = 0
                encoded_beats[mask_indices] = 0
                encoded_chords[mask_indices] = 0

        bsz = latents.shape[0] // 2

        # Sample timesteps
        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps) * torch.ones((bsz,), dtype=torch.int64, device=device).repeat(2)
        else:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.int64
            ).repeat(2)

        timesteps = timesteps.long()

        # Add noise to latents (use same noise for chosen and rejected)
        noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)
        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Target based on prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Forward pass through current model
        if self.set_from == "random":
            model_pred = self.unet(
                noisy_model_input, timesteps, encoder_hidden_states, encoded_beats, encoded_chords,
                encoder_attention_mask=boolean_encoder_mask,
                beat_attention_mask=beat_mask,
                chord_attention_mask=chord_mask
            ).sample
        else:  # pre-trained mode with group_in/group_out
            compressed_latents = self.group_in(noisy_model_input.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            model_pred = self.unet(
                compressed_latents, timesteps, encoder_hidden_states,
                encoder_attention_mask=boolean_encoder_mask
            ).sample
            model_pred = self.group_out(model_pred.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        # Calculate model losses for chosen and rejected
        model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
        model_losses_w, model_losses_l = model_losses.chunk(2)  # chosen (winning) vs rejected (losing)
        model_diff = model_losses_w - model_losses_l

        # Forward pass through reference model (no gradients)
        with torch.no_grad():
            if self.set_from == "random":
                ref_preds = ref_unet(
                    noisy_model_input, timesteps, encoder_hidden_states, encoded_beats, encoded_chords,
                    encoder_attention_mask=boolean_encoder_mask,
                    beat_attention_mask=beat_mask,
                    chord_attention_mask=chord_mask
                ).sample
            else:  # pre-trained mode
                # Note: ref_unet should have group_in/group_out layers if it's also pre-trained
                # Assuming ref_unet has the same structure
                if hasattr(ref_unet, 'group_in') and hasattr(ref_unet, 'group_out'):
                    ref_compressed = ref_unet.group_in(noisy_model_input.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
                    ref_preds = ref_unet.unet(
                        ref_compressed, timesteps, encoder_hidden_states,
                        encoder_attention_mask=boolean_encoder_mask
                    ).sample
                    ref_preds = ref_unet.group_out(ref_preds.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
                else:
                    # Fallback: direct forward through ref_unet (might not work for pre-trained mode)
                    ref_preds = ref_unet(
                        noisy_model_input, timesteps, encoder_hidden_states,
                        encoder_attention_mask=boolean_encoder_mask
                    ).sample

            # Calculate reference model losses
            ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
            ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))
            ref_losses_w, ref_losses_l = ref_loss.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l

        # IPO loss
        reward_margin = model_diff - ref_diff
        target_value = 0.5/self.eta_ipo  # This is the target margin we want to achieve (can be tuned)
        scale_term = -0.5 * self.beta_dpo

        inside_term = scale_term * reward_margin
        ipo_loss = ((inside_term - target_value) ** 2).mean()   #IPO  loss
        # loss = -F.logsigmoid(inside_term).mean()  #DPO loss

        #new add SFT_loss
        loss_sft = model_losses_w.mean()
        # Combine losses (you can weight them differently if desired)
        loss = loss_sft + self.zeta * ipo_loss

        # del inter_term, reward_margin, model_diff, ref_diff, model_losses, ref_loss, model_pred, ref_preds, noisy_model_input, noise
        if return_diff:
            print(f"Model diff: {model_diff.mean().item():.4f}, Ref diff: {ref_diff.mean().item():.4f}")
            print(f"Model loss w: {model_losses_w.mean().item():.4f}, l: {model_losses_l.mean().item():.4f}, loss_sft: {loss_sft.item():.4f}, loss_ipo: {ipo_loss.item():.4f}")
            diff_val = (model_diff - ref_diff).mean().detach()
            return loss, diff_val
        else:
            return loss


    def diffusion_forward(self, latents, prompt, beats, chords, chords_time, validation_mode=False):
        """
        Standard diffusion forward pass (without IPO loss)
        Useful for validation/evaluation
        """
        return super().forward(
            latents=latents,
            prompt=prompt,
            beats=beats,
            chords=chords,
            chords_time=chords_time,
            validation_mode=validation_mode
        )
