#!/bin/bash
#SBATCH --job-name=hybrid_s2_beauty
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/milkak/A-LLMRec/logs/hybrid_beauty_s2_%j.out

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate allm_rec
cd /home/milkak/A-LLMRec

cp models/a_llmrec_model_HYBRID.py models/a_llmrec_model.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Stage 2 Hybrid Beauty ==="
python main.py --pretrain_stage2 --rec_pre_trained_data All_Beauty --batch_size2 8
echo "=== Inference Hybrid Beauty ==="
python main.py --inference --rec_pre_trained_data All_Beauty
echo "=== HYBRID FINAL METRICS ==="
python eval.py
