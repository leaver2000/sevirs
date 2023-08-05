"""
The goal here is to add some mrms data to the sevir. SEVIR data goes back to 2018.
and IAState maintains the a large archive of MRMS data but you have to download the
entire zip file.
"""

from __future__ import annotations

import datetime
import functools
import json
import logging
import os
import shutil
import tempfile
import zipfile
from typing import Any

import polars as pl
import requests
import tqdm

from .._typing import StrPath
from .constants import (
    COORDINATES,
    FEATURES,
    GEOMETRY,
    IASTATE_FILE_FORMAT,
    IASTATE_PATH_FORMAT,
    IASTATE_URL,
    ID,
    MAXRC_EMISS,
    MAXRC_ICECF,
    PROBSEVERE_COLUMNS,
    PROBSEVERE_DATETIME_FORMAT,
    PROBSEVERE_SCHEMA,
    PROPERTIES,
    VALID_TIME,
)


def download(save_to: str = ".", *, date: datetime.datetime) -> str | None:
    # - source
    if not os.path.exists(save_to):
        logging.info(f"🌩️ Creating {save_to} 🌩️")
        os.makedirs(save_to)
    path = date.strftime(IASTATE_PATH_FORMAT)
    name = date.strftime(IASTATE_FILE_FORMAT)
    url = f"{IASTATE_URL}/{path}/{name}"

    # - destination
    dest = os.path.join(save_to, name)
    if os.path.exists(dest):
        logging.info(f"🌩️ {dest} already exists 🌩️")
        return dest

    # - download
    logging.info(f"🌩️ Downloading {url} 🌩️")
    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            logging.error(f"🌩️ {url} returned {r.status_code} 🌩️")
            return None
        total = int(r.headers.get("Content-Length", 0))
        desc = "(Unknown total file size)" if total == 0 else ""
        r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed

        # - write
        with tqdm.tqdm.wrapattr(r.raw, "read", total=total, desc=desc) as raw:
            with open(dest, "wb") as f:
                shutil.copyfileobj(raw, f)

    logging.info(f"🌩️ Downloaded {dest} 🌩️")
    return dest


def read_probsevere(file: StrPath) -> pl.DataFrame:
    with open(file) as f:
        data = json.load(f)
    valid_time = {VALID_TIME: datetime.datetime.strptime(data[VALID_TIME], PROBSEVERE_DATETIME_FORMAT)}
    records = [f[PROPERTIES] | valid_time | {COORDINATES: f[GEOMETRY][COORDINATES]} for f in data[FEATURES]]

    df = pl.DataFrame(records, schema_overrides=PROBSEVERE_SCHEMA)
    return df.with_columns(
        **{
            MAXRC_EMISS: pl.when(df[MAXRC_EMISS] == "N/A").then(None).otherwise(df[MAXRC_EMISS]),
            MAXRC_ICECF: pl.when(df[MAXRC_ICECF] == "N/A").then(None).otherwise(df[MAXRC_ICECF]),
        }
    )


def read_many_probsevere(files: list[StrPath] | list[str] | list[os.PathLike[str]]) -> pl.DataFrame:
    df = pl.concat([read_probsevere(f) for f in files])
    return df[PROBSEVERE_COLUMNS].sort(pl.col(ID), pl.col(VALID_TIME))


class IAStateZipFile(zipfile.ZipFile):
    def contains(self, contains: str = "CONUS") -> pl.Series:
        s = pl.Series(self.namelist())
        suffix = "gz" if contains == "CONUS" else "json"
        return s.filter(s.str.contains(contains) & s.str.ends_with(suffix))

    @property
    def types(self) -> pl.Series:
        df = pl.DataFrame(self.contains("CONUS").str.split("/").to_list()).transpose()
        return df[df.columns[2]]

    def extract_conus(self, list_of_types: list[str], path: StrPath | None = None) -> list[str]:
        s = self.contains("CONUS")
        return [self.extract(f, path) for f in s.filter(self.types.is_in(list_of_types))]

    def extract_probsevere(self, path: StrPath | None = None) -> list[str]:
        s = self.contains("ProbSevere")
        return [self.extract(f, path) for f in s.filter(s.str.contains("ProbSevere"))]


def read(obj: Any) -> pl.DataFrame:
    if isinstance(obj, str):
        if os.path.isdir(obj):
            files = [os.path.join(obj, f) for f in os.listdir(obj)]
            return read_many_probsevere(files)
        elif obj.endswith(".zip"):
            with IAStateZipFile(obj) as f:
                with tempfile.TemporaryDirectory() as d:
                    files = f.extract_probsevere(d)
                    df = read_many_probsevere(files)
                    return df
        else:
            return read_probsevere(obj)
    elif isinstance(obj, list):
        return read_many_probsevere(obj)
    else:
        raise ValueError(f"Cannot read {obj}")
