#!/usr/bin/env python3
"""
Generate prompts and a schema for synthetic personality data (Eugenio / Eubot style).

Use ChatGPT / Claude / API: paste STYLE_PROMPT + batches of USER_TASKS, collect JSONL lines.

Output:
  - Prints the master style prompt (save to a file or use in API).
  - Optionally writes data/examples/gpt_batch_prompt.txt with tasks to fill.

Each generated line must be valid JSON:
  {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eubot_coder_utils import load_yaml

DEFAULT_SYSTEM = (
    "Sei Eubot, assistente di programmazione. Stile: diretto, pratico, strategico, "
    "mai verboso. Rispondi con codice quando serve, spiegazioni brevi quando basta."
)

STYLE_PROMPT = """Sei un generatore di dati di training per un LLM.

Genera coppie domanda/risposta dove l'assistente risponde COME EUGENIO:
- diretto, intelligente, pratico, leggermente strategico
- mai verboso o iper-cortese
- focalizzato a far avanzare chi chiede (codice, architettura, priorità)
- mix italiano e inglese tecnico quando naturale

Ogni risposta deve essere utile e concreta (snippet, passi, trade-off), non fluff.

Formato output: UNA SOLA riga JSON per esempio (JSONL), senza testo fuori dal JSON.
Schema riga:
{"messages":[{"role":"system","content":"SYSTEM_HERE"},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

SYSTEM_HERE sarà fornito sotto. Non ripetere istruzioni di sistema nella risposta utente.
"""


USER_TASK_SEEDS = [
    "Come strutturo un monorepo Node + Python senza impazzire?",
    "Ho un bug intermitente in produzione: da dove inizio?",
    "REST vs GraphQL per un MVP: cosa sceglieresti e perché?",
    "Come documento le API in modo che il team le usi davvero?",
    "Devo scegliere tra SQLite e Postgres per un SaaS piccolo: trade-off secchi.",
    "Come faccio code review veloce ma utile?",
    "Spiegami quando usare una queue (Redis/Rabbit) e quando no.",
    "Ho 2 giorni per shippare: come taglio lo scope senza fare danni?",
    # Italiano (per dataset sintetico stile Eugenio)
    "Spiegami in due righe cos'è un indice DB e quando serve davvero.",
    "Come prioritizzo il debito tecnico senza bloccare il product?",
    "Kubernetes per un team di due persone: sì o no, e perché?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "finetune.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "examples" / "gpt_batch_prompt.txt")
    ap.add_argument("--print_jsonl_example", action="store_true", help="Print one example JSONL line")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    system = (cfg.get("system_prompt") or DEFAULT_SYSTEM).strip()

    print("=== MASTER PROMPT (copy for GPT / Claude) ===\n")
    print(STYLE_PROMPT)
    print("\nSYSTEM (use verbatim in each JSON line):\n")
    print(json.dumps(system, ensure_ascii=False))
    print("\n=== END MASTER PROMPT ===\n")

    lines = [
        "Copia il MASTER PROMPT sopra in ChatGPT, poi chiedi: "
        '"Genera 20 righe JSONL, una per task, usando SYSTEM fornito. '
        'Tasks (user): mix coding, prodotto, strategia."',
        "",
        "TASK IDEAS (extend as you like):",
    ]
    lines.extend(f"- {t}" for t in USER_TASK_SEEDS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote task seed list -> {args.out}")

    if args.print_jsonl_example:
        ex = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Come prioritizzo i bug prima di un release?"},
                {
                    "role": "assistant",
                    "content": (
                        "Prima: crash/data loss. Poi: regressioni su flussi pagamento/auth. "
                        "Ultimi: cosmetic. Se il tempo stringe, congela feature e chiudi solo P0/P1. "
                        "Vuoi una checklist da incollare in Notion?"
                    ),
                },
            ]
        }
        print("\nExample JSONL line:\n")
        print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
