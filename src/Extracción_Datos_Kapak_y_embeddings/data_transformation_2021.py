#Daniela Salazar, para Proyecto Integrador CMP

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


EMBED_MODEL = "text-embedding-3-large"
DEFAULT_API_BATCH_SIZE = 256
DEFAULT_OUTPUT_CHUNK_SIZE = 5000
DEFAULT_MAX_WORKERS = 8
MAX_RETRIES = 6


SCRIPT_DIR = Path(__file__).resolve().parent
CODES_DIR = SCRIPT_DIR
BASE_DIR = CODES_DIR.parent
DATA_2021_PATH = CODES_DIR / "preguntas_2021.csv"
OUTPUT_DIR = CODES_DIR / "output" / "2021"
PREPARED_DATA_PATH = OUTPUT_DIR / "preguntas_2021_prepared.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Carga y prepara preguntas_2021.csv, luego genera embeddings "
            "de forma concurrente y los guarda en chunks CSV."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DATA_2021_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prepared-output", type=Path, default=PREPARED_DATA_PATH)
    parser.add_argument("--model", default=EMBED_MODEL)
    parser.add_argument("--api-batch-size", type=int, default=DEFAULT_API_BATCH_SIZE)
    parser.add_argument("--output-chunk-size", type=int, default=DEFAULT_OUTPUT_CHUNK_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Indice final exclusivo. Si no se define, procesa hasta el final.",
    )
    parser.add_argument(
        "--skip-prepared-save",
        action="store_true",
        help="No guarda el CSV preparado.",
    )
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(CODES_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY no encontrada. Define OPENAI_API_KEY=sk-... "
            "en .env en la raiz del proyecto o en Codes/.env."
        )
    return api_key


def read_wrapped_csv(data_path: Path) -> pd.DataFrame:
    rows: list[str] = []
    with data_path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.reader(file_obj):
            if row:
                rows.append(row[0])
    return pd.read_csv(io.StringIO("\n".join(rows)))


def prepare_2021_data(data_path: Path) -> pd.DataFrame:
    raw_df = read_wrapped_csv(data_path)
    df = raw_df.copy()

    df["valor_preguntas"] = df["valor_preguntas"].fillna("").astype(str).str.strip()
    df["es_acusatoria"] = (
        df["es_acusatoria"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )
    df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"], errors="coerce")

    n_before = len(df)
    df = df[df["valor_preguntas"].str.len() > 0].copy()
    df = df[df["fecha_publicacion"].notna()].copy()
    df = df.reset_index(drop=True)

    print(f"Filas eliminadas (sin texto/fecha) : {n_before - len(df):,}")
    print(f"Filas validas                      : {len(df):,}")
    print(f"Procesos unicos                    : {df['contract_id'].nunique():,}")
    print(
        f"Preguntas acusatorias              : "
        f"{int(df['es_acusatoria'].sum()):,} ({df['es_acusatoria'].mean() * 100:.1f}%)"
    )
    print(
        f"Rango de fechas                    : "
        f"{df['fecha_publicacion'].min().date()} -> {df['fecha_publicacion'].max().date()}"
    )
    return df


def batched(values: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for index in range(0, len(values), batch_size):
        yield index // batch_size, values[index : index + batch_size]


def request_embeddings(
    batch_index: int,
    texts: list[str],
    api_key: str,
    model: str,
) -> tuple[int, list[list[float]]]:
    client = OpenAI(api_key=api_key)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(model=model, input=texts)
            ordered = sorted(response.data, key=lambda item: item.index)
            return batch_index, [item.embedding for item in ordered]
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Fallo definitivo en lote {batch_index + 1} tras {MAX_RETRIES} intentos: {exc}"
                ) from exc
            wait_seconds = min(2**attempt, 30) + random.random()
            print(
                f"[WARN] Lote {batch_index + 1} fallo en intento {attempt}/{MAX_RETRIES}. "
                f"Reintentando en {wait_seconds:.1f}s."
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Estado invalido en lote {batch_index + 1}.")


def embed_unique_texts(
    texts: list[str],
    api_key: str,
    model: str,
    batch_size: int,
    max_workers: int,
) -> dict[str, list[float]]:
    unique_texts = list(dict.fromkeys(texts))
    print(f"Textos totales                     : {len(texts):,}")
    print(f"Textos unicos a embedir            : {len(unique_texts):,}")

    batches = list(batched(unique_texts, batch_size))
    if not batches:
        return {}

    print(f"Lotes API                          : {len(batches):,}")
    print(f"Tamano lote API                    : {batch_size:,}")
    print(f"Workers concurrentes               : {max_workers:,}")

    embeddings_by_text: dict[str, list[float]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(request_embeddings, batch_index, chunk, api_key, model): (batch_index, chunk)
            for batch_index, chunk in batches
        }
        for future in as_completed(future_map):
            batch_index, chunk = future_map[future]
            _, batch_embeddings = future.result()
            for text_value, embedding in zip(chunk, batch_embeddings):
                embeddings_by_text[text_value] = embedding
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(f"Lotes completados                  : {completed:,}/{len(batches):,}")

    return embeddings_by_text


def write_embedding_chunks(
    texts: list[str],
    embeddings_by_text: dict[str, list[float]],
    output_dir: Path,
    start_idx: int,
    chunk_size: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    total_rows = len(texts)

    for offset in range(0, total_rows, chunk_size):
        chunk_texts = texts[offset : offset + chunk_size]
        chunk_embeddings = [embeddings_by_text[text_value] for text_value in chunk_texts]
        emb_df = pd.DataFrame(chunk_embeddings)
        emb_df.columns = [str(i) for i in range(emb_df.shape[1])]

        chunk_start = start_idx + offset
        chunk_end = chunk_start + len(chunk_texts)
        chunk_path = output_dir / f"embeddings_2021_{chunk_start}_{chunk_end}.csv"
        emb_df.to_csv(chunk_path, index=False)
        written_files.append(chunk_path)
        print(f"Chunk guardado                     : {chunk_path.name} ({len(emb_df):,} filas)")

    return written_files


def main() -> None:
    args = parse_args()
    if args.api_batch_size <= 0:
        raise ValueError("--api-batch-size debe ser mayor que 0.")
    if args.output_chunk_size <= 0:
        raise ValueError("--output-chunk-size debe ser mayor que 0.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers debe ser mayor que 0.")
    if args.start_idx < 0:
        raise ValueError("--start-idx no puede ser negativo.")

    api_key = load_api_key()
    df = prepare_2021_data(args.data_path)

    if not args.skip_prepared_save:
        args.prepared_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.prepared_output, index=False)
        print(f"CSV preparado guardado             : {args.prepared_output}")

    end_idx = len(df) if args.end_idx is None else min(args.end_idx, len(df))
    if end_idx <= args.start_idx:
        raise ValueError("El rango solicitado no contiene filas.")

    df_slice = df.iloc[args.start_idx:end_idx].copy().reset_index(drop=True)
    texts = df_slice["valor_preguntas"].tolist()

    print(f"Rango procesado                    : [{args.start_idx:,}, {end_idx:,})")
    embeddings_by_text = embed_unique_texts(
        texts=texts,
        api_key=api_key,
        model=args.model,
        batch_size=args.api_batch_size,
        max_workers=args.max_workers,
    )
    written_files = write_embedding_chunks(
        texts=texts,
        embeddings_by_text=embeddings_by_text,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        chunk_size=args.output_chunk_size,
    )

    print(f"Archivos generados                 : {len(written_files):,}")
    print(f"Directorio de salida               : {args.output_dir}")


if __name__ == "__main__":
    main()
