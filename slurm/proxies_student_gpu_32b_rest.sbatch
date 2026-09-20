#!/bin/bash
#SBATCH --job-name=trp-proxies-gpu-32b-rest
#SBATCH --partition=capella
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --exclude=c118,c46
#SBATCH --output=/data/cat/ws/frwe188h-otagent/teacher_ranking_proxy/runs/logs/%x-%j.out
# The 5 likelihood proxies the n=1000 32B job timed out before finishing
# (global_nll, local_nll_k1/k2/k4 already done). 24h + 2 GPUs (device_map=auto).
# Usage: sbatch proxies_student_gpu_32b_rest.sbatch [run_id]
set -euo pipefail
RUN_ID=${1:-terminal_lego-n1000-s42}
REPO=/data/cat/ws/frwe188h-otagent/OpenThoughts-Agent-trp
source $REPO/hpc/dotenv/zih_capella_franziska.env
VT=${TMPDIR:-/tmp}/venv_trp_$SLURM_JOB_ID
if [ ! -x $VT/bin/python ]; then
  mkdir -p $VT
  tar --warning=no-file-changed -C $WS_ROOT/teacher_ranking_proxy/venv -cf - . | tar -C $VT -xf - || [ ${PIPESTATUS[0]} -le 1 ]
  sed -i "1s|^#!.*/teacher_ranking_proxy/venv/bin/python.*|#!$VT/bin/python|" $VT/bin/* || true
fi
export TEACHER_PROXY_VENV=$VT PATH=$VT/bin:$PATH
export TRP_DEVICE_MAP=auto
cd $REPO
SAMPLE=data/teacher_ranking_proxy/runs/$RUN_ID/sampled_task_ids.json
$VT/bin/python data/teacher_ranking_proxy/compute_proxies.py --sample-file $SAMPLE \
  --student Qwen/Qwen3-32B \
  --proxy local_nll_k8 --proxy aslec_drop --proxy aslec_casl --proxy rsr --proxy scas
# --out: keep the 32B report separate so it does not overwrite the 8B ranking_report.*
$VT/bin/python data/teacher_ranking_proxy/evaluate_ranking.py --run-id $RUN_ID --students Qwen/Qwen3-32B \
  --out data/teacher_ranking_proxy/runs/$RUN_ID/ranking_report_qwen3-32b
