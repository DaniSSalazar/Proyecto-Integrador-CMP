#Daniela Salazar, para Proyecto Integrador CMP

"""
Extracción de preguntas SIE 2021 desde Kapak/SOCE.

Uso:
  1) Crear un archivo .env en la raíz del proyecto o en Codes/
  2) Completar credenciales de la base de datos
  3) Ejecutar: python Codes/extraer_preguntas_2021.py

Salida:
  - preguntas_2021.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


SQL_PREGUNTAS_2021 = """
SELECT 
    z.contract_id,
    z.codigo,
    z.fecha_publicacion,
    z.estado_del_proceso,
    p.presupuesto_referencial_total_sin_iva,
    p.monto_contrato,
    p.monto_adjudicacion,
    z.valor_preguntas,
    z.flag AS es_acusatoria,
    CASE 
        WHEN z.flag THEN 'Acusatoria'
        ELSE 'No Acusatoria'
    END AS clasificacion
FROM (
    SELECT 
        x.contract_id,
        x.codigo,
        x.estado_del_proceso,
        x.fecha_publicacion,
        (SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)) AS valor_preguntas,
        to_tsvector('spanish'::regconfig, ((SELECT jsonb_each.value -> 'pregunta_aclaracion'::text
           FROM jsonb_each(x.preguntas) jsonb_each(key, value)))::text) 
            @@ to_tsquery('spanish'::regconfig, 'corrupción|direccionado|limitante|vulneración|ocultamiento|violación|incompleto|trato<->justo'::text) AS flag
    FROM (
        SELECT 
            b.contract_id,
            b.codigo,
            sf.fecha_value::date AS fecha_publicacion,
            CASE substring(b.estado_del_proceso::text, 1, 1)
                WHEN '{'::text THEN b.estado_del_proceso::text
                ELSE ('{"estado_del_proceso":"'::text || b.estado_del_proceso::text) || '"}'::text
            END::jsonb ->> 'estado_del_proceso'::text AS estado_del_proceso,
            jsonb_array_elements(a.preguntas_y_aclaraciones) AS preguntas,
            is_valid_json(jsonb_array_elements_text(a.preguntas_y_aclaraciones)) AS condicion
        FROM soce_descripcion b
            JOIN soce_fechas sf ON b.contract_id::text = sf.contract_id::text
            JOIN preguntas_y_aclaraciones a ON b.contract_id::integer = a.contract_id
        WHERE jsonb_typeof(a.preguntas_y_aclaraciones) = 'array'::text 
            AND b.tipo_de_contratacion::text = 'Subasta Inversa Electrónica'::text 
            AND sf.fecha_name::text = 'fecha_de_publicacion'::text
    ) x
    WHERE x.condicion IS TRUE 
        AND (x.estado_del_proceso = ANY (ARRAY[
            'Preguntas, Respuestas y Aclaraciones'::text, 'Entrega de Propuesta'::text,
            'Convalidacion de Errores'::text, 'Calificación de Participantes'::text,
            'Oferta Inicial'::text, 'Puja'::text, 'Negociación'::text,
            'Reprogramación Puja'::text, 'Suspendido'::text, 'Por Adjudicar'::text,
            'Adjudicado - Registro de Contratos'::text, 'Adjudicada'::text,
            'Ejecución de Contrato'::text, 'En Recepción'::text, 'Finalizada'::text,
            'Terminado Unilateralmente'::text, 'Finalizado por mutuo acuerdo'::text,
            'Finalizado por disolución de la persona Jurídica'::text,
            'Finalizado a solicitud del contratista'::text, 'Cancelado'::text, 'Desierta'::text
        ]))
) z
LEFT JOIN mv_sie_procesos_datos p ON z.contract_id::text = p.contract_id::text 
    AND z.codigo::text = p.codigo::text
WHERE EXTRACT(YEAR FROM z.fecha_publicacion) = 2021
    AND z.valor_preguntas::text <> '"ACLARACION"'
ORDER BY z.fecha_publicacion, z.contract_id;
"""


@dataclass
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    output_dir: Path
    output_file: Path


def load_settings() -> Settings:
    """
    Carga variables desde .env.

    El archivo .env puede estar:
      - en la raíz del proyecto
      - dentro de la carpeta Codes/
    """

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
            f"Faltan variables de base de datos: {', '.join(missing)}. "
            "Define DB_HOST, DB_PORT, DB_NAME, DB_USER y DB_PASSWORD en .env"
        )

    output_dir_env = os.getenv("OUTPUT_DIR", "./Codes/output")
    output_dir = Path(output_dir_env)

    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "preguntas_2021.csv"

    return Settings(
        db_host=str(host),
        db_port=int(port),
        db_name=str(name),
        db_user=str(user),
        db_password=str(password),
        output_dir=output_dir,
        output_file=output_file,
    )


def create_db_engine(settings: Settings):
    """
    Crea la conexión con PostgreSQL.
    """

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

    return engine


def extract_preguntas_2021(settings: Settings) -> pd.DataFrame:
    """
    Ejecuta el query y devuelve un DataFrame con las preguntas de 2021.
    """

    engine = create_db_engine(settings)

    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(SQL_PREGUNTAS_2021),
            conn,
        )

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica para que el CSV quede más usable.
    """

    if "valor_preguntas" in df.columns:
        df["valor_preguntas"] = (
            df["valor_preguntas"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.strip('"')
        )

    if "clasificacion" in df.columns:
        df["clasificacion"] = df["clasificacion"].fillna("").astype(str).str.strip()

    if "estado_del_proceso" in df.columns:
        df["estado_del_proceso"] = df["estado_del_proceso"].fillna("").astype(str).str.strip()

    return df


def save_output(settings: Settings, df: pd.DataFrame) -> None:
    """
    Guarda el resultado en preguntas_2021.csv.
    """

    df.to_csv(settings.output_file, index=False, encoding="utf-8-sig")
    print(f"[OK] Archivo guardado en: {settings.output_file}")


def main() -> None:
    settings = load_settings()

    print("[INFO] Cargando preguntas de procesos SIE del año 2021...")
    df = extract_preguntas_2021(settings)

    print(f"[INFO] Filas extraídas: {len(df):,}")

    df = clean_dataframe(df)

    save_output(settings, df)

    print("[OK] Proceso completado.")


if __name__ == "__main__":
    main()