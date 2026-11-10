#!/bin/bash
PROJECT_ROOT="/path/to/your/project"  # TODO: set this to your project root directory

export CUDA_VISIBLE_DEVICES=0

LOCAL_MODEL_BIN="${PROJECT_ROOT}/Model/huggingface/hub/models--mustango/ldm/pytorch_model_ldm.bin"
DATASET_DIR="${PROJECT_ROOT}/Dataset/MusicPre"
OUTPUT_DIR="${PROJECT_ROOT}/output"
LOG_FILE="${PROJECT_ROOT}/log/train_$(date +%Y%m%d_%H%M%S).log"
TRAIN_PY_PATH="${PROJECT_ROOT}/MusicAlign-train.py"
ALIGN_TYPE="dpo"
CHECKPOINT_STEPS="best"
REMOVE_BEATS=False
REMOVE_CHORDS=False
ZETA=0

mkdir -p ${OUTPUT_DIR}
mkdir -p ${PROJECT_ROOT}/log

echo -e "=================================================="
echo "Local model weights: ${LOCAL_MODEL_BIN}"
echo "Freeze Beat Encoder + Chord Encoder"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Remove beats encoder: ${REMOVE_BEATS}"
echo "Remove chords encoder: ${REMOVE_CHORDS}"
echo "zeta: ${ZETA}"
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
  --zeta=0  \
  2>&1 | tee ${LOG_FILE}

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo -e "\n Training successful! ${ALIGN_TYPE^^} + frozen encoders complete!"
  echo " Model output: ${OUTPUT_DIR}"
  echo " Log: ${LOG_FILE}"
else
  echo -e "\n Training failed!"
  exit 1
fi
