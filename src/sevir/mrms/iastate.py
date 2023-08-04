import zipfile
import polars as pl
from .._typing import StrPath


class IAStateZipFile(zipfile.ZipFile):
    def __init__(self, file: str):
        super().__init__(file)

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
