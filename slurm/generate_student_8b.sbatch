#!/bin/bash
#SBATCH --job-name=trp-gen-8b
#SBATCH --partition=capella
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --exclude=c118,c46
#SBATCH --output=/data/cat/ws/frwe188h-otagent/teacher_ranking_proxy/runs/logs/%x-%j.out
# Scale the Qwen3-8B student traces to 1000 tasks (a q_S sample-size check vs
# the n=200 q_S used in the headline). 8B serves on 1 GPU (default YAML).
# Resumable. Usage: sbatch generate_student_8b.sbatch [n_tasks] [run_id]
set -euo pipefail
N_TASKS=${1:-1000}
RUN_ID=${2:-terminal_lego-n${N_TASKS}-s42}
REPO=/data/cat/ws/frwe188h-otagent/OpenThoughts-Agent-trp
source $REPO/hpc/dotenv/zih_capella_franziska.env
VT=${TMPDIR:-/tmp}/venv_trp_$SLURM_JOB_ID
if [ ! -x $VT/bin/python ]; then
  mkdir -p $VT
  tar --warning=no-file-changed -C $WS_ROOT/teacher_ranking_proxy/venv -cf - . | tar -C $VT -xf - || [ ${PIPESTATUS[0]} -le 1 ]
  sed -i "1s|^#!.*/teacher_ranking_proxy/venv/bin/python.*|#!$VT/bin/python|" $VT/bin/* || true
fi
export TEACHER_PROXY_VENV=$VT PATH=$VT/bin:$PATH
export OTAGENT_HARBOR_NO_YES=1 OTAGENT_VLLM_NO_RAY=1 VLLM_SKIP_RAY_PROBE=1
cd $REPO
echo "[gen-8b] run_id=$RUN_ID n_tasks=$N_TASKS (default 8B serve YAML, 1 GPU)"
$VT/bin/python data/teacher_ranking_proxy/generate_trajectories.py \
  --dataset terminal_lego --model Qwen/Qwen3-8B --runtime apptainer \
  --n-tasks $N_TASKS --seed 42 --gpus 1 --run-id $RUN_ID --resume
echo "[gen-8b] DONE run_id=$RUN_ID"
ls -d data/teacher_ranking_proxy/runs/$RUN_ID/traces/Qwen__Qwen3-8B/apptainer/harbor_jobs/*/task_* 2>/dev/null | wc -l | xargs echo "trial dirs:"
