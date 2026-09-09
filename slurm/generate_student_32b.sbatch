#!/bin/bash
#SBATCH --job-name=trp-gen-32b
#SBATCH --partition=capella
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --exclude=c118,c46
#SBATCH --output=/data/cat/ws/frwe188h-otagent/teacher_ranking_proxy/runs/logs/%x-%j.out
# Stage-2 student trace generation for the Qwen3-32B student (SCRF q_S source).
# Qwen3-32B served via vLLM TP=2 (2 GPUs); Harbor + Apptainer task sandboxes.
# Verified 8B launch pattern (teacher-proxy-cluster-setup SKILL.md, milestone
# repro) with the 32B serve YAML + --gpus 2. Resumable.
# Usage: sbatch generate_student_32b.sbatch [n_tasks] [run_id]
set -euo pipefail
N_TASKS=${1:-10}
RUN_ID=${2:-terminal_lego-n${N_TASKS}-s42-q32b}
REPO=/data/cat/ws/frwe188h-otagent/OpenThoughts-Agent-trp
source $REPO/hpc/dotenv/zih_capella_franziska.env
# Node-local venv copy + shebang rewrite (Lustre venv is pathologically slow)
VT=${TMPDIR:-/tmp}/venv_trp_$SLURM_JOB_ID
if [ ! -x $VT/bin/python ]; then
  mkdir -p $VT
  tar --warning=no-file-changed -C $WS_ROOT/teacher_ranking_proxy/venv -cf - . | tar -C $VT -xf - || [ ${PIPESTATUS[0]} -le 1 ]
  sed -i "1s|^#!.*/teacher_ranking_proxy/venv/bin/python.*|#!$VT/bin/python|" $VT/bin/* || true
fi
export TEACHER_PROXY_VENV=$VT PATH=$VT/bin:$PATH
export TRP_DATAGEN_YAML=$REPO/hpc/datagen_yaml/qwen3_32b_vllm_serve_32k_2xH100.yaml
export OTAGENT_HARBOR_NO_YES=1 OTAGENT_VLLM_NO_RAY=1 VLLM_SKIP_RAY_PROBE=1
cd $REPO
echo "[gen-32b] run_id=$RUN_ID n_tasks=$N_TASKS yaml=$TRP_DATAGEN_YAML gpus=2"
$VT/bin/python data/teacher_ranking_proxy/generate_trajectories.py \
  --dataset terminal_lego --model Qwen/Qwen3-32B --runtime apptainer \
  --n-tasks $N_TASKS --seed 42 --gpus 2 --run-id $RUN_ID --resume
echo "[gen-32b] DONE run_id=$RUN_ID"
ls -d data/teacher_ranking_proxy/runs/$RUN_ID/traces/Qwen__Qwen3-32B/apptainer/harbor_jobs/*/task_* 2>/dev/null | wc -l | xargs echo "trial dirs:"
