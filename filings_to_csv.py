"""
filings.xbrl.org -> Excel-ready CSV
===================================

Pick one or more companies/filings from the filings.xbrl.org index and download
a single flat CSV containing every tagged fact, one row per fact.

Run:
    pip install streamlit pandas requests
    streamlit run filings_to_csv.py

Data source: https://filings.xbrl.org (JSON:API). xBRL-JSON outputs are produced
by the Arelle XBRL processor. Access is free but XBRL International reserve the
right to rate-limit; keep MAX_WORKERS low and be polite.
"""

from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import requests
import streamlit as st

BASE = "https://filings.xbrl.org"
API = f"{BASE}/api"
HEADERS = {"User-Agent": "filings-to-csv/1.0 (bulk fact extract for analysis)"}
MAX_WORKERS = 4
PAGE_SIZE = 200
EXCEL_CELL_LIMIT = 32_000  # Excel hard limit is 32,767


# ----------------------------------------------------------------------------
# Index: fetch the filing list
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index(country: str | None, max_pages: int = 25) -> pd.DataFrame:
    """Fetch filings (with entity names) from the JSON:API index."""
    rows: list[dict[str, Any]] = []
    params = {
        "include": "entity",
        "page[size]": PAGE_SIZE,
        "sort": "-period_end",
    }
    if country and country != "All":
        params["filter[country]"] = country

    url = f"{API}/filings"
    for _ in range(max_pages):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        entities = {
            e["id"]: e["attributes"]
            for e in payload.get("included", [])
            if e.get("type") == "entity"
        }

        for f in payload.get("data", []):
            attrs = f.get("attributes", {})
            ent_ref = (
                f.get("relationships", {})
                .get("entity", {})
                .get("data") or {}
            )
            ent = entities.get(ent_ref.get("id"), {})
            rows.append(
                {
                    "company": ent.get("name") or "(unknown)",
                    "identifier": ent.get("identifier"),
                    "period_end": attrs.get("period_end"),
                    "country": attrs.get("country"),
                    "errors": attrs.get("error_count"),
                    "inconsistencies": attrs.get("inconsistency_count"),
                    "fxo_id": attrs.get("fxo_id"),
                    "json_url": attrs.get("json_url"),
                    "viewer_url": attrs.get("viewer_url"),
                }
            )

        nxt = payload.get("links", {}).get("next")
        if not nxt:
            break
        url, params = nxt, None  # next link already carries the query string

    df = pd.DataFrame(rows)
    if not df.empty:
        # xBRL-JSON is not generated for every filing (technical errors, etc.)
        df["has_json"] = df["json_url"].notna()
        df = df.sort_values(["company", "period_end"], ascending=[True, False])
    return df


# ----------------------------------------------------------------------------
# Flattening xBRL-JSON -> long table
# ----------------------------------------------------------------------------

def _split_period(period: str | None) -> tuple[str | None, str | None, str | None]:
    """Return (instant, start, end). Durations are 'start/end'."""
    if not period:
        return None, None, None
    if "/" in period:
        start, end = period.split("/", 1)
        return None, start, end
    return period, None, None


def _to_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten_report(doc: dict, meta: dict | None = None) -> pd.DataFrame:
    """Flatten one xBRL-JSON report into one row per fact."""
    meta = meta or {}
    facts = doc.get("facts", {}) or {}
    doc_info = doc.get("documentInfo", {}) or {}
    taxonomy = "; ".join(doc_info.get("taxonomy", []) or [])

    rows = []
    for fact_id, fact in facts.items():
        dims = dict(fact.get("dimensions", {}) or {})
        concept = dims.pop("concept", None)
        entity = dims.pop("entity", None)
        period = dims.pop("period", None)
        unit = dims.pop("unit", None)
        language = dims.pop("language", None)
        noteid = dims.pop("noteId", None)

        instant, start, end = _split_period(period)
        prefix, _, local = (concept or "").partition(":")
        value = fact.get("value")
        numeric = _to_number(value)
        text = "" if value is None else str(value)

        row = {
            "company": meta.get("company"),
            "identifier": meta.get("identifier"),
            "filing_period_end": meta.get("period_end"),
            "country": meta.get("country"),
            "fxo_id": meta.get("fxo_id"),
            "fact_id": fact_id,
            "concept": concept,
            "concept_prefix": prefix or None,
            "concept_name": local or None,
            "is_extension": bool(prefix) and prefix not in {"ifrs-full", "ifrs", "uk-bus", "uk-core"},
            "value_text": text,
            "value_numeric": numeric,
            "decimals": fact.get("decimals"),
            "unit": unit,
            "entity": entity,
            "period_instant": instant,
            "period_start": start,
            "period_end": end,
            "language": language,
            "note_id": noteid,
            "is_text_block": bool(local and local.endswith("TextBlock")) or len(text) > 500,
            "value_length": len(text),
            "n_dimensions": len(dims),
            "taxonomy": taxonomy,
            "json_url": meta.get("json_url"),
        }
        for key, val in dims.items():
            row[f"dim:{key}"] = val
        row["dimensions_json"] = json.dumps(dims, ensure_ascii=False) if dims else None
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_flatten(meta_json: str) -> pd.DataFrame:
    meta = json.loads(meta_json)
    url = meta["json_url"]
    if not url.startswith("http"):
        url = BASE + url
    resp = requests.get(url, headers=HEADERS, timeout=180)
    resp.raise_for_status()
    return flatten_report(resp.json(), meta)


# ----------------------------------------------------------------------------
# Excel hygiene
# ----------------------------------------------------------------------------

_FORMULA_START = re.compile(r"^[=+\-@\t\r]")


def make_excel_safe(df: pd.DataFrame, truncate: bool = True) -> pd.DataFrame:
    """Neutralise CSV-injection and over-long cells in text columns."""
    out = df.copy()
    # pandas 3 uses a dedicated 'str' dtype; pandas 2 uses object
    text_cols = [
        c
        for c in out.columns
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c])
    ]
    for col in text_cols:
        s = out[col].astype("string")
        if truncate:
            too_long = (s.str.len() > EXCEL_CELL_LIMIT).fillna(False)
            s = s.mask(too_long, s.str.slice(0, EXCEL_CELL_LIMIT) + " …[truncated]")
        if col == "value_text":
            # numbers are fine; only guard genuine text that opens like a formula
            risky = (
                s.notna()
                & s.str.match(_FORMULA_START).fillna(False)
                & out["value_numeric"].isna()
            )
            s = s.mask(risky, "'" + s)
        out[col] = s
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    # BOM so Excel opens UTF-8 correctly on Windows
    return buf.getvalue().encode("utf-8-sig")


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="filings.xbrl.org → CSV", layout="wide")
    st.title("filings.xbrl.org → Excel-ready CSV")
    st.caption(
        "Select companies from the ESEF / UKSEF filing index and export every "
        "tagged fact as a single flat CSV. Data: filings.xbrl.org; xBRL-JSON "
        "generated by Arelle."
    )

    with st.sidebar:
        st.header("1. Load the index")
        country = st.selectbox(
            "Country",
            ["GB", "All", "IE", "FR", "DE", "NL", "SE", "NO", "DK", "FI",
             "IT", "ES", "PL", "BE", "AT", "PT", "GR", "CY", "UA"],
        )
        max_pages = st.slider("Max index pages (200 filings each)", 1, 40, 10)
        load = st.button("Fetch filing index", type="primary")

    if load or "index" in st.session_state:
        if load:
            with st.spinner("Fetching index…"):
                st.session_state["index"] = fetch_index(country, max_pages)
        idx = st.session_state["index"]

        if idx.empty:
            st.warning("No filings returned for that filter.")
            return

        st.subheader("2. Choose filings")
        name_filter = st.text_input("Filter by company name contains")
        view = idx[idx["has_json"]]
        if name_filter:
            view = view[view["company"].str.contains(name_filter, case=False, na=False)]

        companies = sorted(view["company"].unique())
        chosen = st.multiselect(
            f"Companies ({len(companies)} available)", companies, max_selections=50
        )
        selected = view[view["company"].isin(chosen)]

        if not selected.empty:
            st.write(f"{len(selected)} filing(s) selected")
            st.dataframe(
                selected[["company", "period_end", "country", "errors", "fxo_id"]],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("3. Options")
            col1, col2 = st.columns(2)
            with col1:
                drop_text = st.checkbox(
                    "Exclude text blocks (recommended for Excel)", value=True
                )
                numeric_only = st.checkbox("Numeric facts only", value=False)
            with col2:
                drop_empty_dims = st.checkbox("Drop all-empty dimension columns", value=True)
                keep_dim_json = st.checkbox("Keep dimensions_json column", value=True)

            if st.button("Build CSV", type="primary"):
                metas = [
                    json.dumps(r, default=str)
                    for r in selected.to_dict("records")
                ]
                frames, failures = [], []
                progress = st.progress(0.0)
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                    for i, (m, result) in enumerate(
                        zip(metas, pool.map(_safe_fetch, metas)), start=1
                    ):
                        if isinstance(result, Exception):
                            failures.append((json.loads(m)["fxo_id"], str(result)))
                        else:
                            frames.append(result)
                        progress.progress(i / len(metas))

                if not frames:
                    st.error("No facts retrieved.")
                    for fid, err in failures:
                        st.write(f"- {fid}: {err}")
                    return

                data = pd.concat(frames, ignore_index=True)

                if drop_text:
                    data = data[~data["is_text_block"]]
                if numeric_only:
                    data = data[data["value_numeric"].notna()]
                if drop_empty_dims:
                    dim_cols = [c for c in data.columns if c.startswith("dim:")]
                    empty = [c for c in dim_cols if data[c].isna().all()]
                    data = data.drop(columns=empty)
                if not keep_dim_json:
                    data = data.drop(columns=["dimensions_json"], errors="ignore")

                data = make_excel_safe(data)

                st.success(
                    f"{len(data):,} facts from {len(frames)} filing(s), "
                    f"{data.shape[1]} columns."
                )
                if failures:
                    st.warning(f"{len(failures)} filing(s) failed — see below.")
                    for fid, err in failures:
                        st.write(f"- {fid}: {err}")

                st.dataframe(data.head(200), use_container_width=True)
                st.download_button(
                    "Download CSV",
                    data=to_csv_bytes(data),
                    file_name="xbrl_facts.csv",
                    mime="text/csv",
                    type="primary",
                )


def _safe_fetch(meta_json: str):
    try:
        return fetch_and_flatten(meta_json)
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        return exc


if __name__ == "__main__":
    main()
