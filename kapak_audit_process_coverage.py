#!/usr/bin/env python3
"""Audita cobertura de procesos seleccionados vs procesos con preguntas validas.

Genera una tabla por contract_id con:
- selected_in_210
- has_questions_for_embeddings
- n_preguntas_validas
- label_acusatorio
- nro_preguntas_acusatorias
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "Codes" / "output"

    labels_path = output_dir / "kapak_sie_process_accusatory_labels.csv"
    raw_path = output_dir / "kapak_sie_questions_raw.csv"
    out_path = output_dir / "kapak_sie_process_audit_coverage.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"No existe: {labels_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"No existe: {raw_path}")

    labels = pd.read_csv(labels_path)
    raw = pd.read_csv(raw_path)

    if "contract_id" not in labels.columns:
        raise ValueError("El archivo de labels no contiene 'contract_id'.")
    if "contract_id" not in raw.columns:
        raise ValueError("El archivo raw no contiene 'contract_id'.")

    # Normaliza tipo para evitar problemas en merge.
    labels["contract_id"] = labels["contract_id"].astype(str)
    raw["contract_id"] = raw["contract_id"].astype(str)

    # Conteo de preguntas validas por proceso (las que entraron al raw).
    counts = (
        raw.groupby("contract_id", as_index=False)
        .agg(n_preguntas_validas=("contract_id", "size"))
    )

    # Auditoria principal: base = procesos seleccionados por el script de labels.
    audit = labels.merge(counts, on="contract_id", how="left")
    audit["selected_in_210"] = 1
    audit["n_preguntas_validas"] = audit["n_preguntas_validas"].fillna(0).astype(int)
    audit["has_questions_for_embeddings"] = (audit["n_preguntas_validas"] > 0).astype(int)

    # Asegura columnas esperadas.
    if "label_acusatorio" not in audit.columns:
        audit["label_acusatorio"] = pd.NA
    if "nro_preguntas_acusatorias" not in audit.columns:
        audit["nro_preguntas_acusatorias"] = pd.NA

    audit = audit[
        [
            "contract_id",
            "selected_in_210",
            "has_questions_for_embeddings",
            "n_preguntas_validas",
            "label_acusatorio",
            "nro_preguntas_acusatorias",
        ]
    ].sort_values(["has_questions_for_embeddings", "n_preguntas_validas", "contract_id"], ascending=[True, True, True])

    audit.to_csv(out_path, index=False, encoding="utf-8")

    # Resumen rapido en consola.
    total_selected = int(audit["selected_in_210"].sum())
    total_with_q = int(audit["has_questions_for_embeddings"].sum())
    total_without_q = total_selected - total_with_q

    print(f"[INFO] Procesos seleccionados: {total_selected}")
    print(f"[INFO] Con preguntas validas: {total_with_q}")
    print(f"[INFO] Sin preguntas validas: {total_without_q}")
    print(f"[OK] Auditoria: {out_path}")


if __name__ == "__main__":
    main()

