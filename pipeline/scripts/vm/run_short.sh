#!/bin/bash
# Queue for the paper's VM experiments (run from /srv/gpu-server/qwen: nohup bash scripts/run_short.sh &).
# Order: the two pool runs of the re-ranker (needed first), the yes-probabilities of the prompted model, then the
# re-ranker on the benchmark pairs (long). Every step appends to its CSV and resumes, so the queue can be restarted.
cd /srv/gpu-server/qwen || exit 1
export HF_HUB_CACHE=/srv/gpu-server/huggingface
PY=/opt/anacondas/anaconda3/envs/qwen/bin/python
LOG=scripts/logs
mkdir -p "$LOG"
run() {  # run <log name> <command...>
  echo "$1 start $(date)" >> "$LOG/status.txt"
  "${@:2}" > "$LOG/$1.log" 2>&1
  echo "$1 rc=$? $(date)" >> "$LOG/status.txt"
}
run rerank_svo_pool    $PY scripts/qwen3_reranker.py svo_pool
run rerank_coco_pool   $PY scripts/qwen3_reranker.py coco_pool
run qwen25p_svo_pool   $PY scripts/qwen25_yes_probability.py svo_pool
run qwen25p_coco_pool  $PY scripts/qwen25_yes_probability.py coco_pool
run rerank_sugarcrepe  $PY scripts/qwen3_reranker.py sugarcrepe
run rerank_svo_bench   $PY scripts/qwen3_reranker.py svo_bench
echo "ALL DONE $(date)" >> "$LOG/status.txt"
