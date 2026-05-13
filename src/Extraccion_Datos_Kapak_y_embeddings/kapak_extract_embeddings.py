#!/usr/bin/env python3
#Daniela Salazar, para Proyecto Integrador CMP

"""Extraccion de preguntas SIE desde Kapak + generacion de embeddings. 
    Año 2022

Uso:
  1) Copiar Codes/.env.example a .env (en la raiz del proyecto o en Codes/)
  2) Completar credenciales y OPENAI_API_KEY
  3) Ejecutar: python Codes/kapak_extract_embeddings.py


"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text


SQL_SIE_QA = """
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
    pya.contract_id,
    pd.fecha_publicacion,
    kv.value ->> 'fecha_pregunta'       AS fecha_pregunta,
    kv.value ->> 'pregunta_aclaracion'  AS pregunta,
    kv.value ->> 'respuesta_aclaracion' AS respuesta
FROM preguntas_y_aclaraciones pya
JOIN selected_processes sp ON sp.contract_id = pya.contract_id
JOIN soce_universo su ON su.id = pya.contract_id
JOIN publication_dates pd ON pd.contract_id = pya.contract_id
,   jsonb_array_elements(pya.preguntas_y_aclaraciones) AS elem
,   jsonb_each(elem) AS kv
WHERE su.contract_type ILIKE '%subasta inversa%'
  AND jsonb_typeof(pya.preguntas_y_aclaraciones) = 'array'
  AND jsonb_typeof(elem) = 'object'
  AND COALESCE(kv.value ->> 'pregunta_aclaracion', '') <> 'ACLARACION'
  AND pd.fecha_publicacion BETWEEN :start_date AND :end_date
ORDER BY pya.contract_id, fecha_pregunta;
"""


@dataclass
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    openai_api_key: str
    embed_model: str
    start_date: str
    end_date: str
    process_limit: int
    batch_size: int
    output_dir: Path
    embedding_cache_file: Path
    include_label_column: bool
    label_value: int


def load_settings() -> Settings:
    # Busca .env en raiz del proyecto y en Codes/
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

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise ValueError("Falta OPENAI_API_KEY en .env")

    output_dir_env = os.getenv("OUTPUT_DIR", "./Codes/output")
    output_dir = Path(output_dir_env)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_default = output_dir / "embedding_cache_text-embedding-3-large.jsonl"
    cache_env = os.getenv("EMBEDDING_CACHE_FILE", str(cache_default))
    embedding_cache_file = Path(cache_env)
    if not embedding_cache_file.is_absolute():
        embedding_cache_file = (base_dir / embedding_cache_file).resolve()
    else:
        embedding_cache_file = embedding_cache_file.resolve()

    return Settings(
        db_host=str(host),
        db_port=int(port),
        db_name=str(name),
        db_user=str(user),
        db_password=str(password),
        openai_api_key=openai_api_key,
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        start_date=os.getenv("START_DATE", "2022-01-01"),
        end_date=os.getenv("END_DATE", "2022-12-31"),
        process_limit=int(os.getenv("PROCESS_LIMIT", "10")),
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        output_dir=output_dir,
        embedding_cache_file=embedding_cache_file,
        include_label_column=os.getenv("INCLUDE_LABEL_COLUMN", "false").lower() == "true",
        label_value=int(os.getenv("LABEL_VALUE", "0")),
    )


def extract_questions(settings: Settings) -> pd.DataFrame:
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
    query = text(SQL_SIE_QA)

    with engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "start_date": settings.start_date,
                "end_date": settings.end_date,
                "process_limit": settings.process_limit,
            },
        )

    df["pregunta"] = df["pregunta"].fillna("").astype(str).str.strip()
    df["respuesta"] = df["respuesta"].fillna("").astype(str).str.strip()

    # Texto final para embedding: pregunta + respuesta para preservar contexto
    df["text_for_embedding"] = (
        "Pregunta: "
        + df["pregunta"]
        + "\nRespuesta: "
        + df["respuesta"]
    ).str.strip()

    # Evita filas vacias y duplicadas para mantener idempotencia.
    df = df[df["text_for_embedding"].str.len() > 0].copy()
    df = df.drop_duplicates(
        subset=["contract_id", "fecha_publicacion", "fecha_pregunta", "pregunta", "respuesta"]
    ).sort_values(["contract_id", "fecha_pregunta"], kind="mergesort")

    return df


def batch_iter(values: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def load_embedding_cache(cache_path: Path) -> Dict[str, List[float]]:
    cache: Dict[str, List[float]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cache[item["text"]] = item["embedding"]
    return cache


def save_embedding_cache(cache_path: Path, cache: Dict[str, List[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for text_key in sorted(cache.keys()):
            f.write(
                json.dumps(
                    {"text": text_key, "embedding": cache[text_key]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def embed_texts(settings: Settings, texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=settings.openai_api_key)
    cache = load_embedding_cache(settings.embedding_cache_file)
    unique_missing = [t for t in sorted(set(texts)) if t not in cache]
    print(f"[INFO] Textos unicos: {len(set(texts)):,} | Nuevos para embedding: {len(unique_missing):,}")

    for idx, chunk in enumerate(batch_iter(unique_missing, settings.batch_size), start=1):
        retries = 0
        while True:
            try:
                resp = client.embeddings.create(model=settings.embed_model, input=chunk)
                for input_text, item in zip(chunk, resp.data):
                    cache[input_text] = item.embedding
                break
            except Exception as exc:
                retries += 1
                if retries > 5:
                    raise RuntimeError(f"Fallo en lote {idx}: {exc}") from exc
                wait_s = min(2**retries, 30)
                time.sleep(wait_s)

    save_embedding_cache(settings.embedding_cache_file, cache)
    embeddings: List[List[float]] = [cache[t] for t in texts]
    return embeddings


def _json_default(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def save_outputs(settings: Settings, df: pd.DataFrame, embeddings: List[List[float]]) -> None:
    if len(df) != len(embeddings):
        raise RuntimeError("Cantidad de filas != cantidad de embeddings")

    raw_path = settings.output_dir / "kapak_sie_questions_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8")

    jsonl_path = settings.output_dir / "kapak_sie_questions_embeddings.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row, emb in zip(df.to_dict(orient="records"), embeddings):
            row["embedding_model"] = settings.embed_model
            row["embedding"] = emb
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    print(f"[OK] Raw CSV: {raw_path}")
    print(f"[OK] Embeddings JSONL: {jsonl_path}")

    # Formato "matriz" compatible con train_embeddings_*.csv (columnas 0..N-1).
    emb_matrix = pd.DataFrame(embeddings)
    emb_matrix.columns = [str(i) for i in range(emb_matrix.shape[1])]
    if settings.include_label_column:
        emb_matrix["label"] = settings.label_value
    matrix_path = (
        settings.output_dir
        / f"kapak_sie_embeddings_matrix_{settings.embed_model.replace('-', '_')}.csv"
    )
    emb_matrix.to_csv(matrix_path, index=False, encoding="utf-8")
    print(f"[OK] Embeddings matrix CSV: {matrix_path}")


def main() -> None:
    settings = load_settings()
    df = extract_questions(settings)
    print(f"[INFO] Filas extraidas: {len(df):,}")
    print(f"[INFO] Procesos maximos configurados: {settings.process_limit}")
    print(f"[INFO] Cache embeddings: {settings.embedding_cache_file}")

    texts = df["text_for_embedding"].tolist()
    embeddings = embed_texts(settings, texts)

    save_outputs(settings, df, embeddings)
    print("[OK] Proceso completado")


if __name__ == "__main__":
    main()
