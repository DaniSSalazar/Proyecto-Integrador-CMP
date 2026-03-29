#!/usr/bin/env python3
"""Genera etiquetas acusatorio/no acusatorio a nivel proceso (contract_id).

Salida:
  Codes/output/kapak_sie_process_accusatory_labels.csv

Usa el mismo criterio de seleccion que kapak_extract_embeddings.py:
- Subasta Inversa
- Fecha de publicacion del proceso (START_DATE/END_DATE)
- LIMIT por PROCESS_LIMIT (ordenado por contract_id)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


SQL_PROCESS_LABELS = """
WITH publication_dates AS (
    SELECT
        f.contract_id,
        (elem2 ->> 'fecha_value')::timestamp AS fecha_publicacion
    FROM fechas f
    CROSS JOIN LATERAL jsonb_array_elements(f.fechas) AS elem2
    WHERE elem2 ->> 'fecha_name' = 'fecha_de_publicacion'
),
selected_processes AS (
    SELECT DISTINCT pya.contract_id
    FROM preguntas_y_aclaraciones pya
    JOIN soce_universo su ON su.id = pya.contract_id
    JOIN publication_dates pd ON pd.contract_id = pya.contract_id
    WHERE su.contract_type ILIKE '%subasta inversa%'
      AND pd.fecha_publicacion BETWEEN :start_date AND :end_date
    ORDER BY pya.contract_id
    LIMIT :process_limit
)
SELECT
    sp.contract_id,
    pd.fecha_publicacion::date AS fecha_publicacion,
    COALESCE(i04.nro_preguntas_acusatorias, 0) AS nro_preguntas_acusatorias,
    CASE
        WHEN COALESCE(i04.nro_preguntas_acusatorias, 0) > 0 THEN 1
        ELSE 0
    END AS label_acusatorio
FROM selected_processes sp
LEFT JOIN publication_dates pd ON pd.contract_id = sp.contract_id
LEFT JOIN mv_sie_proceso_indicador_04 i04 ON i04.contract_id = sp.contract_id::text
ORDER BY sp.contract_id;
"""


@dataclass
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    start_date: str
    end_date: str
    process_limit: int
    output_dir: Path


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    load_dotenv(base_dir / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    missing = [
        key
        for key, value in {
            "DB_HOST": host,
            "DB_NAME": name,
            "DB_USER": user,
            "DB_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            f"Faltan variables para DB ({', '.join(missing)}). "
            "Define DB_HOST, DB_PORT, DB_NAME, DB_USER y DB_PASSWORD en .env"
        )

    output_dir_env = os.getenv("OUTPUT_DIR", "./Codes/output")
    output_dir = Path(output_dir_env)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        db_host=str(host),
        db_port=int(port),
        db_name=str(name),
        db_user=str(user),
        db_password=str(password),
        start_date=os.getenv("START_DATE", "2022-01-01"),
        end_date=os.getenv("END_DATE", "2022-12-31"),
        process_limit=int(os.getenv("PROCESS_LIMIT", "10")),
        output_dir=output_dir,
    )


def extract_process_labels(settings: Settings) -> pd.DataFrame:
    engine = create_engine(
        "postgresql+psycopg2://",
        pool_pre_ping=True,
        connect_args={
            "host": settings.db_host,
            "port": settings.db_port,
            "dbname": settings.db_name,
            "user": settings.db_user,
            "password": settings.db_password,
        },
    )

    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(SQL_PROCESS_LABELS),
            conn,
            params={
                "start_date": settings.start_date,
                "end_date": settings.end_date,
                "process_limit": settings.process_limit,
            },
        )

    # En caso de duplicados por joins, conservar una fila por contract_id.
    df = df.sort_values(["contract_id"]).drop_duplicates(subset=["contract_id"], keep="first")
    return df


def main() -> None:
    settings = load_settings()
    df = extract_process_labels(settings)

    out_path = settings.output_dir / "kapak_sie_process_accusatory_labels.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[INFO] Procesos considerados: {settings.process_limit}")
    print(f"[INFO] Filas generadas: {len(df):,}")
    print(f"[OK] CSV labels: {out_path}")


if __name__ == "__main__":
    main()
