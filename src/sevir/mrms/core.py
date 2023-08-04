from __future__ import annotations

import datetime
import json
import multiprocessing.pool

import polars as pl
import tqdm

from .._typing import StrPath
from .constants import (
    PROPERTIES,
    FEATURES,
    VALID_TIME,
    COORDINATES,
    GEOMETRY,
    PROBSEVERE_DATETIME_FORMAT,
    PROBSEVERE_SCHEMA,
    MAXRC_EMISS,
    MAXRC_ICECF,
    PROBSEVERE_SCHEMA,
    ID,
    VALID_TIME,
    PROBSEVERE_COLUMNS,
)


def read_probsevere(file: StrPath) -> pl.DataFrame:
    with open(file) as f:
        data = json.load(f)
    valid_time = datetime.datetime.strptime(data[VALID_TIME], PROBSEVERE_DATETIME_FORMAT)
    records = [f[PROPERTIES] | {VALID_TIME: valid_time, COORDINATES: f[GEOMETRY][COORDINATES]} for f in data[FEATURES]]

    df = pl.DataFrame(records, schema_overrides=PROBSEVERE_SCHEMA)
    return df.with_columns(
        **{
            MAXRC_EMISS: pl.when(df[MAXRC_EMISS] == "N/A").then(None).otherwise(df[MAXRC_EMISS]),
            MAXRC_ICECF: pl.when(df[MAXRC_ICECF] == "N/A").then(None).otherwise(df[MAXRC_ICECF]),
        }
    )


def read_many_probsevere(files: list[StrPath]) -> pl.DataFrame:
    df = pl.concat([read_probsevere(f) for f in files])
    return df[PROBSEVERE_COLUMNS].sort(pl.col(ID), pl.col(VALID_TIME))
