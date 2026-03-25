# Backlog — fasi dataset non ancora implementate nel repo

Riferimento: [PIANO_DATASET_EXPANSION_v1.md](../../../docs/PIANO_DATASET_EXPANSION_v1.md) (Fasi 4, 5, 9).

Queste fasi **non** hanno ancora pipeline complete in `tools/scraping/`. Da pianificare come step successivi (script dedicati, rate limit, formato raw compatibile con `merge_outputs_to_raw_names.py` o nuovi blob in `TRAIN_PARTS_OPTIONAL`).

| Fase | Contenuto | Note |
|------|-----------|------|
| **4** | Esoteric Archives, Hermetic Library | HTTP scraping / mirror; rispettare ToS e robots |
| **5** | Internet Archive — testi OCR | `internetarchive` già in `requirements-scraping.txt`; serve selezione titoli e normalizzazione testo |
| **9** | peS2o (STEM multidisciplinare) | Dataset voluminoso; subset + filtro dominio come per altri HF |

Dopo implementazione: estendere `merge_outputs_to_raw_names.py` e/o `build_dataset.py` se servono nuovi nomi file in `data/raw/`.
