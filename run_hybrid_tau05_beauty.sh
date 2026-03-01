#!/bin/bash
#SBATCH --job-name=hybrid_tau05
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/milkak/A-LLMRec/logs/hybrid_tau05_%j.out

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate allm_rec
cd /home/milkak/A-LLMRec

cp models/a_llmrec_model_HYBRID.py models/a_llmrec_model.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage 1 Hybrid tau=0.5 ==="
python main.py --pretrain_stage1 --rec_pre_trained_data All_Beauty --batch_size1 8

echo "=== Stage 2 ==="
python main.py --pretrain_stage2 --rec_pre_trained_data All_Beauty --batch_size2 8

echo "=== Inference ==="
python main.py --inference --rec_pre_trained_data All_Beauty

echo "=== METRICS tau=0.5 ==="
python eval.py
