"""Header-preserving prediction truncation (MEDSDataset._truncate_with_header)."""

from __future__ import annotations

import pandas as pd

from data.meds_dataset import MEDSDataset
from data.meds_parser import Event


def test_truncate_with_header_keeps_length_and_header():
    ds = object.__new__(MEDSDataset)
    ds.max_seq_len = 8

    header = [
        Event(pd.Timestamp("1980-01-01"), "AGE", 40.0),
        Event(pd.NaT, "GENDER//M", None),
        Event(pd.NaT, "RACE//WHITE", None),
        Event(pd.NaT, "BMI", 22.0),
    ]
    clinical = [
        Event(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), f"C{i}", float(i))
        for i in range(30)
    ]
    events = header + clinical
    out = MEDSDataset._truncate_with_header(ds, events)
    assert len(out) == 8
    assert out[0].code == "AGE"
    assert out[-1].code == "C29"
