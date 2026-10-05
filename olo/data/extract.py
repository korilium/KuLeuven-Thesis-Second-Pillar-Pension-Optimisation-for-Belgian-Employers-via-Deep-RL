

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from io import BytesIO
from datetime import date



def extractDataYieldNBB(startPeriod: str = "1993-03", endPeriod: str = date.today().strftime("%Y-%m")) -> pd.DataFrame:
    # ── Namespaces ────────────────────────────────────────────────────────────
    NS = {
        "message": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
        "generic":  "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
    }

    # ── Fetch ─────────────────────────────────────────────────────────────────
    url = "https://nsidisseminate-stat.nbb.be/rest/data/BE2,DF_IROLOBE2,1.0/M..F"
    response = requests.get(url, params={"endPeriod": endPeriod, "startPeriod": startPeriod}, timeout=60)
    response.raise_for_status()

    # ── Parse XML ─────────────────────────────────────────────────────────────
    root = ET.parse(BytesIO(response.content)).getroot()

    records = []
    for series in root.findall(".//generic:Series", NS):

        # Dimension labels (FREQ, MATURITY, REF_AREA, …)
        meta = {
            v.attrib["id"]: v.attrib["value"]
            for v in series.findall("generic:SeriesKey/generic:Value", NS)
        }

        # One row per observation
        for obs in series.findall("generic:Obs", NS):
            records.append({
                **meta,
                "DATE":  obs.find("generic:ObsDimension", NS).attrib["value"], # type: ignore
                "YIELD": obs.find("generic:ObsValue",     NS).attrib["value"], # type: ignore
            })

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df["DATE"]  = pd.to_datetime(df["DATE"])
    df["YIELD"] = pd.to_numeric(df["YIELD"])
    
    return df



def load_olo(startPeriod: str = "2000-01", cache: str = None, refresh: bool = False):
    """
    OLO data the economy needs, fetched from the NBB once and then read from a CSV
    cache, so environments build offline and every run calibrates on the SAME data.
    Delete the cache (or pass refresh=True) to pull a newer vintage.

    Returns
    -------
    df10Y      : monthly 10Y OLO history (DATE, YIELD in %)
    maturities : maturities (years) of the most recent cross-section
    yields     : yields (%) of the most recent cross-section
    """
    import os
    if cache is None:
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "olo_yields.csv")
    if refresh or not os.path.exists(cache):
        extractDataYieldNBB(startPeriod=startPeriod).to_csv(cache, index=False)
    dfYield = pd.read_csv(cache, parse_dates=["DATE"])

    df10Y = (dfYield[dfYield["IROLOBE2_MATUR"] == "10Y"]
             .sort_values("DATE").reset_index(drop=True))

    lastDate = dfYield["DATE"].max()
    dfCurrentYield = (
        dfYield[dfYield["DATE"] == lastDate]
        .copy()
        .assign(MAT_NUM=lambda df: df["IROLOBE2_MATUR"].str.replace("Y", "").astype(int))
        .sort_values("MAT_NUM")
        .reset_index(drop=True)
    )
    return df10Y, dfCurrentYield["MAT_NUM"].values, dfCurrentYield["YIELD"].values


# Example usage
if __name__ == "__main__":
    df = extractDataYieldNBB(startPeriod="2000-01")
