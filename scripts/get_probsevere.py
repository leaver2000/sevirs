from __future__ import annotations
import sys
import datetime
import subprocess
import logging
import requests
import zipfile
import io
import pandas as pd

logging.getLogger().setLevel(logging.INFO)


def get_all(date: datetime.datetime):
    r = requests.get(f"https://mrms.agron.iastate.edu/{date:%Y/%m/%d/%Y%m%d%H}.zip")
    if r.status_code != 200:
        logging.warning(f"Could not get {date:%Y%m%d%H}.zip")
        return
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


def get_some(date: datetime.datetime):
    url = f"https://mtarchive.geol.iastate.edu/{date:%Y/%m/%d}/mrms/ncep/ProbSevere/"
    r = requests.get(url)
    if r.status_code != 200:
        print(pd.read_html(url))


def main(now: datetime.datetime, utc_end: datetime.datetime) -> int:
    while now < utc_end:
        cmd = (
            "wget -L -O "
            + now.strftime("%Y%m%d%H")
            + ".zip "
            + f"https://mrms.agron.iastate.edu/{now:%Y/%m/%d/%Y%m%d%H}.zip"
        )
        logging.info(cmd)
        subprocess.run(cmd, shell=True)
        now += datetime.timedelta(hours=1)
    return 0


import polars as pl

(
    BREF_1HR_MAX,
    BRIGHTBANDBOTTOMHEIGHT,
    BRIGHTBANDTOPHEIGHT,
    CREF_1HR_MAX,
    ECHOTOP_18,
    ECHOTOP_30,
    ECHOTOP_50,
    ECHOTOP_60,
    FLASH_CREST_MAXSOILSAT,
    FLASH_CREST_MAXSTREAMFLOW,
    FLASH_CREST_MAXUNITSTREAMFLOW,
    FLASH_HP_MAXSTREAMFLOW,
    FLASH_HP_MAXUNITSTREAMFLOW,
    FLASH_QPE_ARI01H,
    FLASH_QPE_ARI03H,
    FLASH_QPE_ARI06H,
    FLASH_QPE_ARI12H,
    FLASH_QPE_ARI24H,
    FLASH_QPE_ARI30M,
    FLASH_QPE_ARIMAX,
    FLASH_QPE_FFG01H,
    FLASH_QPE_FFG03H,
    FLASH_QPE_FFG06H,
    FLASH_QPE_FFGMAX,
    FLASH_SAC_MAXSOILSAT,
    FLASH_SAC_MAXSTREAMFLOW,
    FLASH_SAC_MAXUNITSTREAMFLOW,
    GAUGEINFLINDEX_01H_PASS1,
    GAUGEINFLINDEX_01H_PASS2,
    GAUGEINFLINDEX_03H_PASS1,
    GAUGEINFLINDEX_03H_PASS2,
    GAUGEINFLINDEX_06H_PASS1,
    GAUGEINFLINDEX_06H_PASS2,
    GAUGEINFLINDEX_12H_PASS1,
    GAUGEINFLINDEX_12H_PASS2,
    GAUGEINFLINDEX_24H_PASS1,
    GAUGEINFLINDEX_24H_PASS2,
    GAUGEINFLINDEX_48H_PASS1,
    GAUGEINFLINDEX_48H_PASS2,
    GAUGEINFLINDEX_72H_PASS1,
    GAUGEINFLINDEX_72H_PASS2,
    H50_ABOVE_M20C,
    H50_ABOVE_0C,
    H60_ABOVE_M20C,
    H60_ABOVE_0C,
    HEIGHTCOMPOSITEREFLECTIVITY,
    HEIGHTLOWLEVELCOMPOSITEREFLECTIVITY,
    LVL3_HREET,
    LVL3_HIGHRESVIL,
    LAYERCOMPOSITEREFLECTIVITY_ANC,
    LAYERCOMPOSITEREFLECTIVITY_HIGH,
    LAYERCOMPOSITEREFLECTIVITY_LOW,
    LAYERCOMPOSITEREFLECTIVITY_SUPER,
    LOWLEVELCOMPOSITEREFLECTIVITY,
    MESH,
    MESH_MAX_120MIN,
    MESH_MAX_1440MIN,
    MESH_MAX_240MIN,
    MESH_MAX_30MIN,
    MESH_MAX_360MIN,
    MESH_MAX_60MIN,
    MERGEDAZSHEAR_0_2KMAGL,
    MERGEDAZSHEAR_3_6KMAGL,
    MERGEDBASEREFLECTIVITY,
    MERGEDBASEREFLECTIVITYQC,
    MERGEDREFLECTIVITYATLOWESTALTITUDE,
    MERGEDREFLECTIVITYCOMPOSITE,
    MERGEDREFLECTIVITYQC,
    MERGEDREFLECTIVITYQCCOMPOSITE,
    MERGEDREFLECTIVITYQCOMPOSITE,
    MERGEDRHOHV,
    MERGEDZDR,
    MODEL_0DEGC_HEIGHT,
    MODEL_SURFACETEMP,
    MODEL_WETBULBTEMP,
    MULTISENSOR_QPE_01H_PASS1,
    MULTISENSOR_QPE_01H_PASS2,
    MULTISENSOR_QPE_03H_PASS1,
    MULTISENSOR_QPE_03H_PASS2,
    MULTISENSOR_QPE_06H_PASS1,
    MULTISENSOR_QPE_06H_PASS2,
    MULTISENSOR_QPE_12H_PASS1,
    MULTISENSOR_QPE_12H_PASS2,
    MULTISENSOR_QPE_24H_PASS1,
    MULTISENSOR_QPE_24H_PASS2,
    MULTISENSOR_QPE_48H_PASS1,
    MULTISENSOR_QPE_48H_PASS2,
    MULTISENSOR_QPE_72H_PASS1,
    MULTISENSOR_QPE_72H_PASS2,
    NLDN_CG_001MIN_AVGDENSITY,
    NLDN_CG_005MIN_AVGDENSITY,
    NLDN_CG_015MIN_AVGDENSITY,
    NLDN_CG_030MIN_AVGDENSITY,
    POSH,
    PRECIPFLAG,
    PRECIPRATE,
    RADARACCUMULATIONQUALITYINDEX_01H,
    RADARACCUMULATIONQUALITYINDEX_03H,
    RADARACCUMULATIONQUALITYINDEX_06H,
    RADARACCUMULATIONQUALITYINDEX_12H,
    RADARACCUMULATIONQUALITYINDEX_24H,
    RADARACCUMULATIONQUALITYINDEX_48H,
    RADARACCUMULATIONQUALITYINDEX_72H,
    RADAR_ONLY_QPE_01H,
    RADAR_ONLY_QPE_03H,
    RADAR_ONLY_QPE_06H,
    RADAR_ONLY_QPE_12H,
    RADAR_ONLY_QPE_15M,
    RADAR_ONLY_QPE_24H,
    RADAR_ONLY_QPE_48H,
    RADAR_ONLY_QPE_72H,
    RADAR_ONLY_QPE_SINCE12Z,
    RADAR_QUALITYINDEX,
    REFLECTIVITY_AT_LOWESTALTITUDE,
    REFLECTIVITY_M10C,
    REFLECTIVITY_M15C,
    REFLECTIVITY_M20C,
    REFLECTIVITY_M5C,
    REFLECTIVITY_0C,
    ROTATION_TRACK120MIN,
    ROTATION_TRACK1440MIN,
    ROTATION_TRACK240MIN,
    ROTATION_TRACK30MIN,
    ROTATION_TRACK360MIN,
    ROTATION_TRACK60MIN,
    ROTATION_TRACKML120MIN,
    ROTATION_TRACKML1440MIN,
    ROTATION_TRACKML240MIN,
    ROTATION_TRACKML30MIN,
    ROTATION_TRACKML360MIN,
    ROTATION_TRACKML60MIN,
    SHI,
    SEAMLESS_HSR,
    SEAMLESS_HSR_HEIGHT,
    SYNTHETIC_PRECIPRATEID,
    VII,
    VIL,
    VIL_DENSITY,
    VIL_MAX_120MIN,
    VIL_MAX_1440MIN,
    WARM_RAIN_PROBABILITY,
) = (
    "BREF_1HR_MAX",
    "BrightBandBottomHeight",
    "BrightBandTopHeight",
    "CREF_1HR_MAX",
    "EchoTop_18",
    "EchoTop_30",
    "EchoTop_50",
    "EchoTop_60",
    "FLASH_CREST_MAXSOILSAT",
    "FLASH_CREST_MAXSTREAMFLOW",
    "FLASH_CREST_MAXUNITSTREAMFLOW",
    "FLASH_HP_MAXSTREAMFLOW",
    "FLASH_HP_MAXUNITSTREAMFLOW",
    "FLASH_QPE_ARI01H",
    "FLASH_QPE_ARI03H",
    "FLASH_QPE_ARI06H",
    "FLASH_QPE_ARI12H",
    "FLASH_QPE_ARI24H",
    "FLASH_QPE_ARI30M",
    "FLASH_QPE_ARIMAX",
    "FLASH_QPE_FFG01H",
    "FLASH_QPE_FFG03H",
    "FLASH_QPE_FFG06H",
    "FLASH_QPE_FFGMAX",
    "FLASH_SAC_MAXSOILSAT",
    "FLASH_SAC_MAXSTREAMFLOW",
    "FLASH_SAC_MAXUNITSTREAMFLOW",
    "GaugeInflIndex_01H_Pass1",
    "GaugeInflIndex_01H_Pass2",
    "GaugeInflIndex_03H_Pass1",
    "GaugeInflIndex_03H_Pass2",
    "GaugeInflIndex_06H_Pass1",
    "GaugeInflIndex_06H_Pass2",
    "GaugeInflIndex_12H_Pass1",
    "GaugeInflIndex_12H_Pass2",
    "GaugeInflIndex_24H_Pass1",
    "GaugeInflIndex_24H_Pass2",
    "GaugeInflIndex_48H_Pass1",
    "GaugeInflIndex_48H_Pass2",
    "GaugeInflIndex_72H_Pass1",
    "GaugeInflIndex_72H_Pass2",
    "H50_Above_-20C",
    "H50_Above_0C",
    "H60_Above_-20C",
    "H60_Above_0C",
    "HeightCompositeReflectivity",
    "HeightLowLevelCompositeReflectivity",
    "LVL3_HREET",
    "LVL3_HighResVIL",
    "LayerCompositeReflectivity_ANC",
    "LayerCompositeReflectivity_High",
    "LayerCompositeReflectivity_Low",
    "LayerCompositeReflectivity_Super",
    "LowLevelCompositeReflectivity",
    "MESH",
    "MESH_Max_120min",
    "MESH_Max_1440min",
    "MESH_Max_240min",
    "MESH_Max_30min",
    "MESH_Max_360min",
    "MESH_Max_60min",
    "MergedAzShear_0-2kmAGL",
    "MergedAzShear_3-6kmAGL",
    "MergedBaseReflectivity",
    "MergedBaseReflectivityQC",
    "MergedReflectivityAtLowestAltitude",
    "MergedReflectivityComposite",
    "MergedReflectivityQC",
    "MergedReflectivityQCComposite",
    "MergedReflectivityQComposite",
    "MergedRhoHV",
    "MergedZdr",
    "Model_0degC_Height",
    "Model_SurfaceTemp",
    "Model_WetBulbTemp",
    "MultiSensor_QPE_01H_Pass1",
    "MultiSensor_QPE_01H_Pass2",
    "MultiSensor_QPE_03H_Pass1",
    "MultiSensor_QPE_03H_Pass2",
    "MultiSensor_QPE_06H_Pass1",
    "MultiSensor_QPE_06H_Pass2",
    "MultiSensor_QPE_12H_Pass1",
    "MultiSensor_QPE_12H_Pass2",
    "MultiSensor_QPE_24H_Pass1",
    "MultiSensor_QPE_24H_Pass2",
    "MultiSensor_QPE_48H_Pass1",
    "MultiSensor_QPE_48H_Pass2",
    "MultiSensor_QPE_72H_Pass1",
    "MultiSensor_QPE_72H_Pass2",
    "NLDN_CG_001min_AvgDensity",
    "NLDN_CG_005min_AvgDensity",
    "NLDN_CG_015min_AvgDensity",
    "NLDN_CG_030min_AvgDensity",
    "POSH",
    "PrecipFlag",
    "PrecipRate",
    "RadarAccumulationQualityIndex_01H",
    "RadarAccumulationQualityIndex_03H",
    "RadarAccumulationQualityIndex_06H",
    "RadarAccumulationQualityIndex_12H",
    "RadarAccumulationQualityIndex_24H",
    "RadarAccumulationQualityIndex_48H",
    "RadarAccumulationQualityIndex_72H",
    "RadarOnly_QPE_01H",
    "RadarOnly_QPE_03H",
    "RadarOnly_QPE_06H",
    "RadarOnly_QPE_12H",
    "RadarOnly_QPE_15M",
    "RadarOnly_QPE_24H",
    "RadarOnly_QPE_48H",
    "RadarOnly_QPE_72H",
    "RadarOnly_QPE_Since12Z",
    "RadarQualityIndex",
    "ReflectivityAtLowestAltitude",
    "Reflectivity_-10C",
    "Reflectivity_-15C",
    "Reflectivity_-20C",
    "Reflectivity_-5C",
    "Reflectivity_0C",
    "RotationTrack120min",
    "RotationTrack1440min",
    "RotationTrack240min",
    "RotationTrack30min",
    "RotationTrack360min",
    "RotationTrack60min",
    "RotationTrackML120min",
    "RotationTrackML1440min",
    "RotationTrackML240min",
    "RotationTrackML30min",
    "RotationTrackML360min",
    "RotationTrackML60min",
    "SHI",
    "SeamlessHSR",
    "SeamlessHSRHeight",
    "SyntheticPrecipRateID",
    "VII",
    "VIL",
    "VIL_Density",
    "VIL_Max_120min",
    "VIL_Max_1440min",
    "WarmRainProbability",
)
import numpy as np
import os

# StrPath: TypeAlias = str | os.PathLike[str]  # stable


class IAStateZipFile(zipfile.ZipFile):
    def __init__(self, file: str):
        super().__init__(file)

    def __repr__(self):
        return repr(self.to_frame())

    def to_series(self):
        s = pl.Series(self.namelist())
        return s.filter(s.str.contains("CONUS") & (s.str.ends_with("gz") | s.str.contains("json")))

    def to_frame(self):
        df = pl.DataFrame(self.to_series().str.split("/").to_list()).transpose()
        df.columns = ["date", "region", "type", "file"]
        return df

    @property
    def types(self) -> pl.Series:
        return self.to_frame()["type"]

    def select(self, list_of_types: list[str], path: StrPath | None = None) -> list[str]:
        s = self.to_series()
        return [self.extract(f, path) for f in s.filter(self.types.is_in(list_of_types))]

    def select_probsevere(self, path: StrPath | None = None) -> list[str]:
        s = pl.Series(self.namelist())
        return [self.extract(f, path) for f in s.filter(s.str.contains("ProbSevere"))]


def unzip(file: str, key_words: list[str] = ["CONUS", "ProbSevere"]) -> int:
    with IAStateZipFile(file) as z:
        print(z.select_probsevere())
        # img_types = sorted(z.to_frame()["type"].unique())
        # print(", ".join([t.upper() for t in img_types]), "=", img_types)

        # print(", ".join(z.to_frame()["type"].unique().str.to_uppercase().to_list()))

    return 0


if __name__ == "__main__":
    sys.exit(
        #
        unzip("2021060100.zip")
    )
    # Get hourly files from June 1 and June 2 2021
    # sys.exit(main(
    #     now=datetime.datetime(2021, 6, 1, 0),
    #     utc_end=datetime.datetime(2021, 6, 3, 0),
    # ))
