#!/usr/bin/env python3
"""Pipeline end-to-end para indices por proceso real (contract_id).

Entrena 3 modelos (NB, RF, LR) con tus datasets de entrenamiento, predice
probabilidades sobre embeddings reales extraidos de Kapak y calcula indices
por proceso sin fusion de modelos.

Salidas:
  - Codes/output/kapak_sie_question_probabilities_by_model.csv
  - Codes/output/kapak_sie_process_indices_by_model.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


@dataclass
class Paths:
    train_nb: Path
    train_rf: Path
    train_lr: Path
    target_embeddings: Path
    target_raw: Path
    output_dir: Path


def default_paths(base_dir: Path) -> Paths:
    return Paths(
        train_nb=base_dir
        / "Codes"
        / "Archivos_Naive_Bayes"
        / "train_embeddings_balanced_AUG_GPT4o_mini_total_augmented_text-embedding-3-large.csv",
        train_rf=base_dir
        / "Codes"
        / "Archivos_Random_Forest"
        / "train_embeddings_balanced_total_sentence_prompt_GPT4o_mini_text-embedding-3-large.csv",
        train_lr=base_dir
        / "Codes"
        / "Archivos_Logistic_Regression"
        / "train_embeddings_balanced_total_sinonimos_text-embedding-3-large.csv",
        target_embeddings=base_dir
        / "Codes"
        / "output"
        / "kapak_sie_embeddings_matrix_text_embedding_3_large.csv",
        target_raw=base_dir / "Codes" / "output" / "kapak_sie_questions_raw.csv",
        output_dir=base_dir / "Codes" / "output",
    )


def _validate_files(paths: Paths) -> None:
    for p in [
        paths.train_nb,
        paths.train_rf,
        paths.train_lr,
        paths.target_embeddings,
        paths.target_raw,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"No se encontro archivo requerido: {p}")


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "label"]


def _load_target_embeddings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    return df


def _load_target_contract_ids(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "contract_id" not in df.columns:
        raise ValueError("El archivo raw no contiene la columna 'contract_id'.")
    return df[["contract_id"]].copy()


def _prepare_model_inputs(
    train_path: Path, target_embeddings: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(train_path)
    if "label" not in train.columns:
        raise ValueError(f"El train no tiene columna label: {train_path}")

    features = _feature_cols(train)
    missing = [c for c in features if c not in target_embeddings.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas de embeddings en target para {train_path.name}. "
            f"Ejemplos faltantes: {missing[:10]}"
        )

    x_train = train[features].to_numpy()
    y_train = train["label"].to_numpy()
    x_target = target_embeddings[features].to_numpy()
    return x_train, y_train, x_target


def _ic_topk(arr: np.ndarray, k: int = 3) -> float:
    if arr.size == 0:
        return float("nan")
    k_eff = min(k, arr.size)
    topk = np.partition(arr, arr.size - k_eff)[-k_eff:]
    return float(topk.mean())


def _ic_percentile(arr: np.ndarray, alpha: float = 0.90) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, alpha))


def _aggregate_process_indices(
    df_q: pd.DataFrame, prob_col: str, prefix: str, k_top: int, alpha: float
) -> pd.DataFrame:
    def _agg(g: pd.Series) -> pd.Series:
        v = g.to_numpy(dtype=float)
        return pd.Series(
            {
                f"IC_media_{prefix}": float(v.mean()),
                f"IC_max_{prefix}": float(v.max()),
                f"IC_top{k_top}_{prefix}": _ic_topk(v, k=k_top),
                f"IC_pctl_{int(alpha*100)}_{prefix}": _ic_percentile(v, alpha=alpha),
            }
        )

    out = (
        df_q.groupby("contract_id", as_index=False)
        .agg(n_preguntas=("contract_id", "size"))
        .merge(
            df_q.groupby("contract_id")[prob_col].apply(_agg).reset_index(),
            on="contract_id",
            how="left",
        )
    )
    return out


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    paths = default_paths(base_dir)
    _validate_files(paths)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    target_embeddings = _load_target_embeddings(paths.target_embeddings)
    target_raw = _load_target_contract_ids(paths.target_raw)
    if len(target_embeddings) != len(target_raw):
        raise ValueError(
            "No coincide el numero de filas entre embeddings y raw. "
            f"embeddings={len(target_embeddings)}, raw={len(target_raw)}"
        )

    # Entrenar + predecir por modelo
    x_train_nb, y_train_nb, x_target_nb = _prepare_model_inputs(
        paths.train_nb, target_embeddings
    )
    model_nb = GaussianNB(var_smoothing=0.35111917342151305)
    model_nb.fit(x_train_nb, y_train_nb)
    prob_nb = model_nb.predict_proba(x_target_nb)[:, 1]

    x_train_rf, y_train_rf, x_target_rf = _prepare_model_inputs(
        paths.train_rf, target_embeddings
    )
    model_rf = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        n_estimators=200,
        random_state=42,
        n_jobs=1,
    )
    model_rf.fit(x_train_rf, y_train_rf)
    prob_rf = model_rf.predict_proba(x_target_rf)[:, 1]

    x_train_lr, y_train_lr, x_target_lr = _prepare_model_inputs(
        paths.train_lr, target_embeddings
    )
    model_lr = LogisticRegression(
        C=100,
        penalty="l2",
        solver="saga",
        max_iter=1000,
        random_state=42,
    )
    model_lr.fit(x_train_lr, y_train_lr)
    prob_lr = model_lr.predict_proba(x_target_lr)[:, 1]

    # Probabilidades por pregunta (fila)
    q_probs = target_raw.copy()
    q_probs.insert(0, "row_id", np.arange(len(q_probs), dtype=int))
    q_probs["prob_nb"] = prob_nb
    q_probs["prob_rf"] = prob_rf
    q_probs["prob_lr"] = prob_lr

    q_out = paths.output_dir / "kapak_sie_question_probabilities_by_model.csv"
    q_probs.to_csv(q_out, index=False, encoding="utf-8")

    # Indices por proceso real (contract_id), por modelo, sin fusion
    k_top = 3
    alpha = 0.90
    nb_idx = _aggregate_process_indices(q_probs, "prob_nb", "nb", k_top, alpha)
    rf_idx = _aggregate_process_indices(q_probs, "prob_rf", "rf", k_top, alpha)
    lr_idx = _aggregate_process_indices(q_probs, "prob_lr", "lr", k_top, alpha)

    process_idx = (
        nb_idx.merge(rf_idx.drop(columns=["n_preguntas"]), on="contract_id", how="outer")
        .merge(lr_idx.drop(columns=["n_preguntas"]), on="contract_id", how="outer")
        .sort_values("contract_id")
        .reset_index(drop=True)
    )

    p_out = paths.output_dir / "kapak_sie_process_indices_by_model.csv"
    process_idx.to_csv(p_out, index=False, encoding="utf-8")

    print(f"[INFO] Preguntas procesadas: {len(q_probs):,}")
    print(f"[INFO] Procesos unicos: {q_probs['contract_id'].nunique():,}")
    print(f"[OK] Probabilidades por pregunta: {q_out}")
    print(f"[OK] Indices por proceso: {p_out}")


if __name__ == "__main__":
    main()

