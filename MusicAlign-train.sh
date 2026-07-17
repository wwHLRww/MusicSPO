#!/bin/bash
# ====================== Core Environment Configuration ======================
export CUDA_VISIBLE_DEVICES=2
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
export HF_HOME=/public/zhoumingtong/Model/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

LOCAL_MODEL_BIN="/public/zhoumingtong/Model/huggingface/hub/models--mustango/ldm/pytorch_model_ldm.bin"
DATASET_DIR="/public/zhoumingtong/Dataset/Music_dpo"
OUTPUT_DIR="/public/zhoumingtong/Music/tango-master/tango2/output_/output_sft"
LOG_FILE="/public/zhoumingtong/Music/tango-master/tango2/log/train_ipo_$(date +%Y%m%d_%H%M%S).log"
TRAIN_PY_PATH="/public/zhoumingtong/Music/tango-master/tango2/tango2-train_forzen.py"
ALIGN_TYPE="dpo"
CHECKPOINT_STEPS="best"  # best or epoch
REMOVE_BEATS=False
REMOVE_CHORDS=False
LAMBDA_SFT=0

# ====================== Create Directories ======================
mkdir -p ${OUTPUT_DIR}
mkdir -p /public/zhoumingtong/Music/tango-master/tango2/log

# ====================== Print Configuration ======================
echo -e "=================================================="
echo "Local model weights: ${LOCAL_MODEL_BIN}"
echo "Freeze Beat Encoder + Chord Encoder"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Remove beats encoder: ${REMOVE_BEATS}"
echo "Remove chords encoder: ${REMOVE_CHORDS}"
echo "lambda_sft: ${LAMBDA_SFT}"
echo -e "==================================================\n"

python "${TRAIN_PY_PATH}" \
  --hf_model Muisc_dpo  \
  --unet_model_config="configs/diffusion_model_config_munet.json" \
  --freeze_text_encoder \
  --freeze_beat_encoder \
  --freeze_chord_encoder \
  --learning_rate=9.6e-7 \
  --num_train_epochs=5 \
  --num_warmup_steps=50 \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dpo_beta=2000 \
  --dpo_beta_max=2000 \
  --dpo_beta_min=2000   \
  --sft_first_epochs=1 \
  --dataset_dir=${DATASET_DIR} \
  --output_dir=${OUTPUT_DIR} \
  --max_train_samples=-1 \
  --max_eval_samples=-1 \
  --align_type=${ALIGN_TYPE} \
  --eta_ipo=0.5 \
  --checkpointing_steps=${CHECKPOINT_STEPS}  \
  --lambda_sft=0  \
  2>&1 | tee ${LOG_FILE}

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo -e "\n Training successful! ${ALIGN_TYPE^^} + frozen encoders complete!"
  echo " Model output: ${OUTPUT_DIR}"
  echo " Log: ${LOG_FILE}"
else
  echo -e "\n Training failed!"
  exit 1
fi
