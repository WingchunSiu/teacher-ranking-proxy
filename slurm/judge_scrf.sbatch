#!/bin/bash
#SBATCH --job-name=trp-judge-scrf
#SBATCH --partition=capella
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --exclude=c118,c46
#SBATCH --output=/data/cat/ws/frwe188h-otagent/teacher_ranking_proxy/runs/logs/%x-%j.out
# SCRF with a chosen student q_S source (gpt-oss-120b judge served by judge.py).
# Usage: sbatch judge_scrf.sbatch RUN_ID STUDENT STUDENT_RUNS [JUDGE] [TAG]
#  8B q_S@1000: ... terminal_lego-n1000-s42 Qwen/Qwen3-8B  terminal_lego-n1000-s42
#  32B q_S@200: ... terminal_lego-n1000-s42 Qwen/Qwen3-32B terminal_lego-n200-s42-q32b
set -euo pipefail
RUN_ID=$1; STUDENT=$2; STUDENT_RUNS=$3
JUDGE=${4:-openai/gpt-oss-120b}; TAG=${5:-@gpt-oss-120b}
REPO=/data/cat/ws/frwe188h-otagent/OpenThoughts-Agent-trp
source $REPO/hpc/dotenv/zih_capella_franziska.env
VT=${TMPDIR:-/tmp}/venv_trp_$SLURM_JOB_ID
if [ ! -x $VT/bin/python ]; then
  mkdir -p $VT
  tar --warning=no-file-changed -C $WS_ROOT/teacher_ranking_proxy/venv -cf - . | tar -C $VT -xf - || [ ${PIPESTATUS[0]} -le 1 ]
  sed -i "1s|^#!.*/teacher_ranking_proxy/venv/bin/python.*|#!$VT/bin/python|" $VT/bin/* || true
fi
export TEACHER_PROXY_VENV=$VT PATH=$VT/bin:$PATH TRP_JUDGE_MODEL=$JUDGE
unset HF_HUB_DISABLE_XET; $VT/bin/hf download "$JUDGE" > $TMPDIR/hf_dl_$SLURM_JOB_ID.log 2>&1
export HF_HUB_OFFLINE=1
cd $REPO
SAMPLE=data/teacher_ranking_proxy/runs/$RUN_ID/sampled_task_ids.json
echo "[judge-scrf] run=$RUN_ID student=$STUDENT student_runs=$STUDENT_RUNS tag=$TAG"
$VT/bin/python data/teacher_ranking_proxy/compute_proxies.py --sample-file $SAMPLE \
  --student "$STUDENT" --proxy scrf --student-runs "$STUDENT_RUNS" --judge-tag "$TAG"
$VT/bin/python data/teacher_ranking_proxy/evaluate_ranking.py --run-id $RUN_ID --students "$STUDENT"
