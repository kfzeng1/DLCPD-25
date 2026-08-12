#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/zkf/DLCPD-25
run_id=j3-joint-full-e67e96e-r2
run_dir="$repo_root/artifacts/training/detection/$run_id"
training_unit=dlcpd25-j3-e67e96e-r2.service
output="$run_dir/epoch-1-supervision.json"

while systemctl --user is-active --quiet "$training_unit"; do
  if [[ -s "$run_dir/history.json" ]] &&
    [[ "$(jq 'length' "$run_dir/history.json")" -ge 1 ]] &&
    [[ -s "$run_dir/joint-last.pt" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -s "$run_dir/history.json" ]] ||
  [[ "$(jq 'length' "$run_dir/history.json")" -lt 1 ]] ||
  [[ ! -s "$run_dir/joint-last.pt" ]]; then
  jq -n \
    --arg run_id "$run_id" \
    --arg checked_at "$(date --iso-8601=seconds)" \
    '{run_id: $run_id, checked_at: $checked_at, status: "failed_before_epoch_1_checkpoint"}' \
    > "$output"
  exit 1
fi

python_result=$(PYTHONPATH="$repo_root/project/src" /home/zkf/pytorch-env/bin/python - "$run_dir/history.json" <<'PY'
import json
import math
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))[0]
values = [
    record["train"]["classification_loss"],
    record["train"]["detection_loss"],
    record["classification"]["accuracy"],
    record["detection"]["map"],
]
if not all(math.isfinite(float(value)) for value in values):
    raise SystemExit("epoch 1 contains non-finite metrics")
if float(record["classification"]["accuracy"]) < 0.85:
    raise SystemExit("epoch 1 classification accuracy is below the safety floor")
print(json.dumps(record, ensure_ascii=False))
PY
)

jq -n \
  --arg run_id "$run_id" \
  --arg checked_at "$(date --iso-8601=seconds)" \
  --arg checkpoint_sha256 "$(sha256sum "$run_dir/joint-last.pt" | cut -d ' ' -f 1)" \
  --argjson epoch "$python_result" \
  '{
    run_id: $run_id,
    checked_at: $checked_at,
    status: "epoch_1_passed_training_continues_without_polling",
    epoch: $epoch,
    joint_last_sha256: $checkpoint_sha256,
    test_metrics_read: false
  }' > "$output"

loginctl lock-session 4
