

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



# Example usage

df = extractDataYieldNBB(startPeriod="2000-01")





