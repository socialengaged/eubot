#!/usr/bin/env bash
# Finestra di manutenzione: promuove raw_staging -> data/raw prima di build_dataset.py.
# Eseguire con il training fermato (o dopo checkpoint) — vedi SETUP_SCRAPER.md.
#
# Uso:
#   bash promote_raw_staging.sh              # default paths
#   bash promote_raw_staging.sh --dry-run
#   SRC=/path/raw_staging DST=/path/raw bash promote_raw_staging.sh
#
# Non usa --delete su rsync: i file in raw non presenti in staging restano (merge additivo).
set -u
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; fi

SRC="${SRC:-/workspace/eurobot_baby/data/raw_staging}"
DST="${DST:-/workspace/eurobot_baby/data/raw}"

if [[ ! -d "$SRC" ]]; then
  echo "ERR: missing directory: $SRC" >&2
  exit 1
fi
mkdir -p "$DST"

echo "Promote: $SRC -> $DST (dry=$DRY)"
if [[ "$DRY" -eq 1 ]]; then
  rsync -avn "$SRC"/ "$DST"/
  exit 0
fi
rsync -av "$SRC"/ "$DST"/
echo "Done. Next: cd /workspace/eurobot_baby && python scripts/build_dataset.py"
echo "Then update configs/training.yaml max_steps and resume training (see RUNBOOK_PHASE0.md)."
