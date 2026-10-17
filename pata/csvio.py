"""CSV in, CSV out. Shared by the CLI and the API upload endpoint."""

from __future__ import annotations

import csv
import io

from pata.schema import Market, TriageInput
from pata.service import TriageResult

TRUE = {"1", "true", "yes", "y", "t"}
NORMALISED = ["admin1", "city", "area", "street", "building", "unit", "landmark", "postcode",
              "makani", "po_box"]


def read_inputs(text: str, default_market: Market) -> tuple[list[dict], list[TriageInput]]:
    """Parse an orders CSV. Needs a raw_address (or address) column; the rest is optional."""
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    if rows and "raw_address" not in rows[0] and "address" not in rows[0]:
        raise ValueError("CSV needs a raw_address column")
    inputs = []
    for r in rows:
        inputs.append(
            TriageInput(
                market=(r.get("market") or default_market.value).strip().upper(),
                raw_address=(r.get("raw_address") or r.get("address") or "").strip() or "-",
                phone_present=(r.get("phone_present") or "true").strip().lower() in TRUE,
                payment=(r.get("payment") or "PREPAID").strip().upper(),
                declared_pincode=r.get("declared_pincode") or None,
                city_hint=r.get("city_hint") or None,
            )
        )
    return rows, inputs


def write_results(rows: list[dict], results: list[TriageResult]) -> str:
    extra = ["decision", "confidence", "reason_codes", "driver_note", "call_script", "source"]
    base = list(rows[0].keys()) if rows else ["raw_address"]
    fields = base + [c for c in extra + NORMALISED if c not in base]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row, res in zip(rows, results, strict=True):
        o = res.output
        n = o.normalised.model_dump(mode="json")
        w.writerow({
            **row,
            "decision": o.decision.value,
            "confidence": o.confidence,
            "reason_codes": " ".join(c.value for c in o.reason_codes),
            "driver_note": o.driver_note,
            "call_script": o.call_script or "",
            "source": res.source,
            **{k: n[k] or "" for k in NORMALISED},
        })
    return buf.getvalue()
