#!/bin/bash
#SBATCH --job-name=trp-proxies-gpu-32b
#SBATCH --partition=capella
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=c118,c46
#SBATCH --output=/data/cat/ws/frwe188h-otagent/teacher_ranking_proxy/runs/logs/%x-%j.out
# Student-specificity test (PROXY_SPEC.md §8): the student-dependent likelihood
# proxies for the Qwen3-32B student, scoring the SAME teacher trajectories.
# GRACE excluded here (TRAK fp32 gradients at 32B risk OOM on one 80GB GPU;
# run separately). SCRF excluded (needs generated 32B student traces).
# Usage: sbatch proxies_student_gpu_32b.sbatch [run_id]
set -euo pipefail
RUN_ID=${1:-terminal_lego-n1000-s42}; shift || true
REPO=/data/cat/ws/frwe188h-otagent/OpenThoughts-Agent-trp
source $REPO/hpc/dotenv/zih_capella_franziska.env
VT=${TMPDIR:-/tmp}/venv_trp_$SLURM_JOB_ID
if [ ! -x $VT/bin/python ]; then
  mkdir -p $VT
  tar --warning=no-file-changed -C $WS_ROOT/teacher_ranking_proxy/venv -cf - . | tar -C $VT -xf - || [ ${PIPESTATUS[0]} -le 1 ]
  sed -i "1s|^#!.*/teacher_ranking_proxy/venv/bin/python.*|#!$VT/bin/python|" $VT/bin/* || true
fi
export TEACHER_PROXY_VENV=$VT PATH=$VT/bin:$PATH
export TRP_DEVICE_MAP=auto   # shard 32B across the 2 GPUs
cd $REPO
SAMPLE=data/teacher_ranking_proxy/runs/$RUN_ID/sampled_task_ids.json
$VT/bin/python data/teacher_ranking_proxy/compute_proxies.py --sample-file $SAMPLE \
  --student Qwen/Qwen3-32B \
  --proxy global_nll \
  --proxy local_nll_k1 --proxy local_nll_k2 --proxy local_nll_k4 --proxy local_nll_k8 \
  --proxy aslec_drop --proxy aslec_casl --proxy rsr --proxy scas "$@"
$VT/bin/python data/teacher_ranking_proxy/evaluate_ranking.py --run-id $RUN_ID --students Qwen/Qwen3-32B
