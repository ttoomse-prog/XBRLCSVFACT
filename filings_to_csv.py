"""
filings.xbrl.org -> Excel-ready CSV  (v2, teaching edition)
===========================================================

Pick companies from the filings.xbrl.org index — individually or from a curated
set — and download every tagged fact as one flat, spreadsheet-ready CSV.

Run:
    pip install streamlit pandas requests
    streamlit run filings_to_csv.py

Data source: https://filings.xbrl.org (JSON:API). xBRL-JSON outputs are produced
by the Arelle XBRL processor. The API is free but XBRL International reserve the
right to rate-limit, so keep MAX_WORKERS low.

CLASSROOM NOTE
--------------
If a whole cohort runs this at once you are 30 machines hitting one API. Build
the dataset once yourself, save it as Parquet (much smaller than CSV, and it
keeps date and number types intact), commit it to the repo, and set
CLASSROOM_CACHE below to its filename. The app then offers the cached copy
first and only calls the API when someone asks for something new. Update
CLASSROOM_VINTAGE whenever you rebuild it so students know how current it is.
"""

from __future__ import annotations

import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import requests
import streamlit as st

BASE = "https://filings.xbrl.org"
API = f"{BASE}/api"
HEADERS = {"User-Agent": "filings-to-csv/2.0 (teaching tool; bulk fact extract)"}
MAX_WORKERS = 4
PAGE_SIZE = 200
EXCEL_CELL_LIMIT = 32_000  # Excel's hard limit is 32,767
CLASSROOM_CACHE = "classroom_dataset.parquet"  # optional pre-built file in repo root
CLASSROOM_VINTAGE = "September 2026"  # update whenever you rebuild the file above
USER_SETS_FILE = "sets.json"  # optional, extends CURATED_SETS


# ----------------------------------------------------------------------------
# Curated sets
# ----------------------------------------------------------------------------
# Sets come from sets.json in the repo root. Each entry can match on any of:
#   countries   list of country codes, exact
#   leis        list of LEIs — the durable option, since names change
#   name_any    case-insensitive substrings of the company name
#   latest_only keep one filing per company (default true)
#   limit       cap the number of filings
# Build one by hand with "Save this selection as a set" at the bottom of the
# page. If sets.json is absent, the set picker is hidden entirely.
#
# Note there is no size or market-cap field in the filings index, so a set like
# "largest 100" has to be built from a selection rather than derived.

CURATED_SETS: dict[str, dict[str, Any]] = {}


def load_user_sets() -> dict[str, dict[str, Any]]:
    """Merge in sets.json from the repo root if present."""
    if not os.path.exists(USER_SETS_FILE):
        return {}
    try:
        with open(USER_SETS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        st.sidebar.warning(f"Could not read {USER_SETS_FILE} — ignoring it.")
        return {}


def apply_set(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Filter the index down to the filings a set describes."""
    out = df.copy()
    if spec.get("countries"):
        out = out[out["country"].isin(spec["countries"])]
    if spec.get("leis"):
        wanted = {str(x).upper() for x in spec["leis"]}
        out = out[out["identifier"].str.upper().isin(wanted)]
    if spec.get("name_any"):
        pattern = "|".join(re.escape(frag) for frag in spec["name_any"])
        out = out[out["company"].str.contains(pattern, case=False, na=False)]
    if spec.get("latest_only", True):
        out = out.sort_values("period_end", ascending=False).drop_duplicates("company")
    if spec.get("limit"):
        out = out.head(int(spec["limit"]))
    return out


# ----------------------------------------------------------------------------
# Glossary and column dictionary — the teaching layer
# ----------------------------------------------------------------------------

GLOSSARY = [
    ("Fact", "One tagged number or piece of text in a report — a single cell of "
             "meaning. '£4,231m of revenue for the year to 31 December 2024' is one fact."),
    ("Concept", "What the fact is, drawn from a taxonomy. ifrs-full:Revenue is a "
                "concept. The prefix says which taxonomy it came from."),
    ("Extension", "A concept the company invented because nothing in the standard "
                  "taxonomy fitted. Useful to study: lots of extensions can mean the "
                  "taxonomy is a poor fit, or that comparability is being lost."),
    ("Dimension", "A qualifier that narrows a fact — by segment, by geography, by "
                  "class of share. Revenue for the UK segment is the same concept as "
                  "total revenue, with a dimension attached."),
    ("Period", "Instant facts are balances at a date (assets). Duration facts cover "
               "a span (revenue). This is why balance sheet and income statement "
               "items look different in the data."),
    ("Unit", "Currency or measure, as an identifier such as iso4217:GBP. Facts "
             "without a unit are text or counts."),
    ("Decimals", "How precisely the value was reported. -3 means the figure was "
                 "given in thousands, not that it is wrong."),
    ("Text block", "A whole note or policy section tagged as one enormous string. "
                   "Excluded by default here because a single one can be longer than "
                   "this entire page."),
    ("ESEF / UKSEF", "The EU and UK rules requiring listed companies to file annual "
                     "reports in inline XBRL. This index only holds those filings — "
                     "smaller companies file with Companies House instead."),
]

COLUMN_DICTIONARY = {
    "company": "Company name as recorded in the filings index.",
    "identifier": "Legal Entity Identifier (LEI) — stable across years, unlike names.",
    "filing_period_end": "Period end of the filing the fact came from.",
    "country": "Country of the filing regime.",
    "fxo_id": "The filings.xbrl.org identifier for this filing.",
    "fact_id": "Identifier for the fact within the report.",
    "concept": "Full concept name including taxonomy prefix.",
    "concept_prefix": "Taxonomy the concept came from.",
    "concept_name": "Concept without the prefix — usually what you filter on.",
    "is_extension": "TRUE if the company defined this concept itself.",
    "value_text": "The reported value as text (always populated).",
    "value_numeric": "The value as a number, blank for text facts. Use this for sums.",
    "decimals": "Reported precision. -3 = thousands, -6 = millions.",
    "unit": "Currency or measure identifier.",
    "entity": "Reporting entity the fact belongs to.",
    "period_instant": "Date, for balance-type facts.",
    "period_start": "Start date, for period-type facts.",
    "period_end": "End date, for period-type facts.",
    "language": "Language of a text fact, where given.",
    "note_id": "Footnote link, where present.",
    "is_text_block": "TRUE for narrative blocks and very long values.",
    "value_length": "Character length of the value — useful for spotting narrative.",
    "n_dimensions": "How many dimensions qualify this fact. 0 = a headline total.",
    "taxonomy": "Taxonomy entry points used by the filing.",
    "dim:*": "One column per dimension found, e.g. dim:ifrs-full:SegmentsAxis.",
    "dimensions_json": "All dimensions for the fact as JSON, for when you need them together.",
    "json_url": "Source xBRL-JSON file, so any row can be traced back.",
    "viewer_url": "Opens the filing in the inline viewer with tags highlighted.",
}

# Concepts students most often want. Local names, no prefix.
COMMON_CONCEPTS = [
    "Revenue", "ProfitLoss", "ProfitLossBeforeTax", "OperatingProfitLoss",
    "Assets", "Liabilities", "Equity", "CashAndCashEquivalents",
    "PropertyPlantAndEquipment", "Goodwill", "IntangibleAssetsOtherThanGoodwill",
    "EmployeeBenefitsExpense", "NumberOfEmployees", "BasicEarningsLossPerShare",
    "CashFlowsFromUsedInOperatingActivities", "IncomeTaxExpenseContinuingOperations",
]


# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def fetch_index(country: str | None, max_pages: int = 25) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    params = {"include": "entity", "page[size]": PAGE_SIZE, "sort": "-period_end"}
    if country and country != "All":
        params["filter[country]"] = country

    url = f"{API}/filings"
    status = st.empty()
    for page in range(1, max_pages + 1):
        status.caption(f"Reading the filings index — page {page}…")
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
            ent_ref = (f.get("relationships", {}).get("entity", {}).get("data")) or {}
            ent = entities.get(ent_ref.get("id"), {})
            rows.append({
                "company": ent.get("name") or "(unknown)",
                "identifier": ent.get("identifier") or "",
                "period_end": attrs.get("period_end"),
                "country": attrs.get("country"),
                "errors": attrs.get("error_count"),
                "inconsistencies": attrs.get("inconsistency_count"),
                "fxo_id": attrs.get("fxo_id"),
                "json_url": attrs.get("json_url"),
                "viewer_url": attrs.get("viewer_url"),
            })

        nxt = payload.get("links", {}).get("next")
        if not nxt:
            break
        url, params = nxt, None
    status.empty()

    df = pd.DataFrame(rows)
    if not df.empty:
        df["has_json"] = df["json_url"].notna()
        df = df.sort_values(["company", "period_end"], ascending=[True, False])
    return df


# ----------------------------------------------------------------------------
# Flattening
# ----------------------------------------------------------------------------

def _split_period(period: str | None) -> tuple[str | None, str | None, str | None]:
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


# filings.xbrl.org lays out a filing as
#   /{lei}/{date}/{scheme}/{country}/{n}/{lei}-{date}/reports/ixbrlviewer.html
# and fxo_id holds every part of that: {lei}-{date}-{scheme}-{country}-{n}
_FXO = re.compile(
    r"^(?P<lei>[A-Z0-9]{20})-(?P<date>\d{4}-\d{2}-\d{2})"
    r"-(?P<scheme>[A-Z]+)-(?P<cc>[A-Z]{2})-(?P<n>\d+)$"
)


def viewer_link(fxo_id: Any, known: Any = None) -> str | None:
    """Link to the filing in the inline viewer, with tags highlighted."""
    if known and isinstance(known, str):
        return known if known.startswith("http") else BASE + known
    m = _FXO.match(str(fxo_id or ""))
    if not m:
        return None
    d = m.groupdict()
    return (
        f"{BASE}/{d['lei']}/{d['date']}/{d['scheme']}/{d['cc']}/{d['n']}/"
        f"{d['lei']}-{d['date']}/reports/ixbrlviewer.html"
    )


STANDARD_PREFIXES = {"ifrs-full", "ifrs", "uk-bus", "uk-core", "uk-gaap", "esef_cor"}


def flatten_report(doc: dict, meta: dict | None = None) -> pd.DataFrame:
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
            "is_extension": bool(prefix) and prefix not in STANDARD_PREFIXES,
            "value_text": text,
            "value_numeric": _to_number(value),
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
            "viewer_url": viewer_link(meta.get("fxo_id"), meta.get("viewer_url")),
        }
        for key, val in dims.items():
            row[f"dim:{key}"] = val
        row["dimensions_json"] = json.dumps(dims, ensure_ascii=False) if dims else None
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=40)
def fetch_and_flatten(meta_json: str) -> pd.DataFrame:
    meta = json.loads(meta_json)
    url = meta["json_url"]
    if not url.startswith("http"):
        url = BASE + url
    resp = requests.get(url, headers=HEADERS, timeout=180)
    resp.raise_for_status()
    return flatten_report(resp.json(), meta)


# ----------------------------------------------------------------------------
# Excel hygiene and output
# ----------------------------------------------------------------------------

_FORMULA_START = re.compile(r"^[=+\-@\t\r]")


def make_excel_safe(df: pd.DataFrame, truncate: bool = True) -> pd.DataFrame:
    out = df.copy()
    text_cols = [
        c for c in out.columns
        if out[c].dtype == object or pd.api.types.is_string_dtype(out[c])
    ]
    for col in text_cols:
        s = out[col].astype("string")
        if truncate:
            too_long = (s.str.len() > EXCEL_CELL_LIMIT).fillna(False)
            s = s.mask(too_long, s.str.slice(0, EXCEL_CELL_LIMIT) + " …[truncated]")
        if col == "value_text":
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
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel reads UTF-8


def dictionary_bytes(columns: list[str]) -> bytes:
    lines = [
        "COLUMN GUIDE",
        "Source: filings.xbrl.org, xBRL-JSON generated by Arelle.",
        "",
    ]
    for col in columns:
        key = "dim:*" if col.startswith("dim:") else col
        desc = COLUMN_DICTIONARY.get(key, "")
        lines.append(f"{col}\n    {desc}\n")
    lines.append("\nGLOSSARY\n")
    for term, meaning in GLOSSARY:
        lines.append(f"{term}\n    {meaning}\n")
    return "\n".join(lines).encode("utf-8")


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Concepts down the side, one column per company and period."""
    base = df[(df["value_numeric"].notna()) & (df["n_dimensions"] == 0)]
    if base.empty:
        return pd.DataFrame()
    base = base.assign(
        column=base["company"].astype(str) + " " + base["filing_period_end"].astype(str)
    )
    return base.pivot_table(
        index="concept_name",
        columns="column",
        values="value_numeric",
        aggfunc="first",
    ).reset_index()


def _safe_fetch(meta_json: str):
    try:
        return fetch_and_flatten(meta_json)
    except Exception as exc:  # noqa: BLE001 — surfaced in the UI
        return exc


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def render_intro() -> None:
    st.info(
        "**New here?** Every listed company in the EU and UK now files its annual "
        "report with machine-readable tags attached to the numbers. This tool "
        "collects those tagged numbers for the companies you choose and hands them "
        "back as a spreadsheet — one row per tagged number.",
        icon="📘",
    )
    with st.expander("What the words mean"):
        for term, meaning in GLOSSARY:
            st.markdown(f"**{term}** — {meaning}")
    with st.expander("Things to try once you have the data"):
        st.markdown(
            "- Filter to `concept_name = Revenue` and `n_dimensions = 0` to get each "
            "company's headline top line, then compare.\n"
            "- Count rows where `is_extension` is TRUE. Which companies invent the "
            "most concepts of their own, and why might that be?\n"
            "- Compare `n_dimensions` across a bank and a retailer — who discloses "
            "more segment detail?\n"
            "- Look at `decimals`. Do companies report to the nearest thousand or "
            "million, and does that change between statements?\n"
            "- Open a row's `viewer_url` and find the same number in the published "
            "report. Does the tag match what a human reader would say it means?"
        )


@st.cache_data(show_spinner=False)
def read_cache(path: str) -> pd.DataFrame:
    """Read the pre-built dataset. Parquet keeps dtypes; CSV is the fallback."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    # A file that has been through Excel comes back with day-first dates.
    col = "filing_period_end"
    if col in df.columns and df[col].astype(str).str.match(r"\d{2}/\d{2}/\d{4}").any():
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce") \
                    .dt.strftime("%Y-%m-%d")
    # datasets built before report links existed can have them derived
    if "viewer_url" not in df.columns and "fxo_id" in df.columns:
        df["viewer_url"] = df["fxo_id"].map(viewer_link)
    return df


def offer_classroom_cache() -> pd.DataFrame | None:
    if not os.path.exists(CLASSROOM_CACHE):
        return None
    st.success(
        f"A ready-made dataset is available, built in {CLASSROOM_VINTAGE}. Load it "
        "to start straight away, or build your own selection below. Companies that "
        "have reported since then will show newer figures if you fetch them live.",
        icon="📦",
    )
    if st.button("Load the ready-made dataset"):
        try:
            return read_cache(CLASSROOM_CACHE)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read {CLASSROOM_CACHE}: {exc}")
    return None


def main() -> None:
    st.set_page_config(page_title="Filings index → CSV", page_icon="📥", layout="wide")
    st.title("Filings index → CSV")
    st.caption(
        "Tagged facts from ESEF and UKSEF annual reports, as a spreadsheet. "
        "Data: filings.xbrl.org · xBRL-JSON generated by Arelle."
    )

    render_intro()

    all_sets = {**CURATED_SETS, **load_user_sets()}

    with st.sidebar:
        st.header("1 · Load the index")
        country = st.selectbox(
            "Country",
            ["GB", "All", "IE", "FR", "DE", "NL", "SE", "NO", "DK", "FI",
             "IT", "ES", "PL", "BE", "AT", "PT", "GR", "CY", "UA"],
            help="The index covers ESEF, UKSEF and Ukrainian filings only.",
        )
        max_pages = st.slider(
            "How much of the index to read", 1, 40, 10,
            help="Each page is 200 filings. Start small.",
        )
        load = st.button("Fetch filing index", type="primary")
        st.divider()
        st.caption(
            "Built on the free filings.xbrl.org API. Please don't hammer it — "
            "results are cached for an hour."
        )

    cached = offer_classroom_cache()
    if cached is not None:
        st.session_state["data"] = cached

    if load:
        with st.spinner("Fetching…"):
            st.session_state["index"] = fetch_index(country, max_pages)

    if "index" in st.session_state:
        idx = st.session_state["index"]
        if idx.empty:
            st.warning("No filings came back for that filter. Try another country.")
            return

        st.subheader("2 · Choose companies")
        available = idx[idx["has_json"]]

        preselected: list[str] = []
        if all_sets:
            set_name = st.selectbox(
                "Start from a set (optional)", ["— none —"] + sorted(all_sets),
            )
            if set_name != "— none —":
                spec = all_sets[set_name]
                if spec.get("note"):
                    st.caption(spec["note"])
                matched = apply_set(available, spec)
                preselected = sorted(matched["company"].unique())
                if not preselected:
                    st.warning(
                        "Nothing in this set matched the index you've loaded. Try a "
                        "different country, or read more pages."
                    )
                else:
                    st.caption(f"{len(preselected)} companies matched.")

        name_filter = st.text_input("Search by name")
        view = available
        if name_filter:
            view = view[view["company"].str.contains(name_filter, case=False, na=False)]

        companies = sorted(set(view["company"].unique()) | set(preselected))
        chosen = st.multiselect(
            f"Companies ({len(companies)} available)",
            companies,
            default=preselected[:25],
            max_selections=40,
            help="Each company can have several years of filings.",
        )
        selected = available[available["company"].isin(chosen)]

        latest_only = st.checkbox("Most recent filing per company only", value=True)
        if latest_only and not selected.empty:
            selected = selected.sort_values("period_end", ascending=False) \
                               .drop_duplicates("company")

        if not selected.empty:
            st.dataframe(
                selected[["company", "period_end", "country", "errors", "fxo_id"]],
                use_container_width=True, hide_index=True,
            )

            st.subheader("3 · Shape the output")
            col1, col2 = st.columns(2)
            with col1:
                drop_text = st.checkbox(
                    "Leave out narrative text blocks", value=True,
                    help="These are whole notes tagged as one value. Excel struggles.",
                )
                numeric_only = st.checkbox("Numbers only", value=False)
                totals_only = st.checkbox(
                    "Headline figures only (no segment breakdowns)", value=False,
                    help="Keeps facts with no dimensions attached.",
                )
            with col2:
                concept_filter = st.multiselect(
                    "Keep only these concepts (optional)", COMMON_CONCEPTS,
                    help="Leave empty to keep everything.",
                )
                drop_empty_dims = st.checkbox("Tidy away empty columns", value=True)

            if st.button("Build the spreadsheet", type="primary"):
                metas = [json.dumps(r, default=str) for r in selected.to_dict("records")]
                frames, failures = [], []

                with st.status("Collecting filings…", expanded=True) as status:
                    progress = st.progress(0.0)
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                        for i, (meta_json, result) in enumerate(
                            zip(metas, pool.map(_safe_fetch, metas)), start=1
                        ):
                            meta = json.loads(meta_json)
                            if isinstance(result, Exception):
                                failures.append((meta["company"], str(result)))
                                st.write(f"✗ {meta['company']} — could not be read")
                            else:
                                st.write(f"✓ {meta['company']} — {len(result):,} facts")
                            progress.progress(i / len(metas), text=f"{i} of {len(metas)}")
                            if not isinstance(result, Exception):
                                frames.append(result)
                    status.update(
                        label=f"Collected {len(frames)} of {len(metas)} filings",
                        state="complete", expanded=False,
                    )

                if not frames:
                    st.error(
                        "Nothing could be retrieved. The API may be busy — wait a "
                        "moment and try again with fewer companies."
                    )
                    return

                data = pd.concat(frames, ignore_index=True)
                raw_count = len(data)

                if drop_text:
                    data = data[~data["is_text_block"]]
                if numeric_only:
                    data = data[data["value_numeric"].notna()]
                if totals_only:
                    data = data[data["n_dimensions"] == 0]
                if concept_filter:
                    data = data[data["concept_name"].isin(concept_filter)]
                if drop_empty_dims:
                    dim_cols = [c for c in data.columns if c.startswith("dim:")]
                    data = data.drop(columns=[c for c in dim_cols if data[c].isna().all()])

                if data.empty:
                    st.warning(
                        f"All {raw_count:,} facts were filtered out. Loosen the "
                        "options above — the concept filter is the usual culprit."
                    )
                    return

                data = make_excel_safe(data)
                st.session_state["data"] = data

    # ---- results ----
    if "data" in st.session_state:
        data = st.session_state["data"]
        st.subheader("4 · Your data")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Facts", f"{len(data):,}")
        c2.metric("Companies", data["company"].nunique())
        c3.metric("Concepts", data["concept_name"].nunique())
        if "is_extension" in data:
            share = 100 * data["is_extension"].mean()
            c4.metric("Company-defined", f"{share:.0f}%")

        tab_long, tab_pivot, tab_guide = st.tabs(
            ["One row per fact", "Comparison table", "Column guide"]
        )

        with tab_long:
            st.dataframe(
                data.head(300), use_container_width=True,
                column_config={
                    "viewer_url": st.column_config.LinkColumn(
                        "Annual report", display_text="Open ↗",
                        help="Opens the filing with its tags highlighted.",
                    )
                },
            )
            st.caption(
                "Every row links back to the published report — useful for "
                "checking what a tagged number actually refers to."
            )
            st.download_button(
                "Download CSV", to_csv_bytes(data),
                file_name="xbrl_facts.csv", mime="text/csv", type="primary",
            )

        with tab_pivot:
            pivot = build_pivot(data)
            if pivot.empty:
                st.info(
                    "A comparison table needs headline numeric facts. Rebuild with "
                    "'Numbers only' ticked."
                )
            else:
                st.caption("Concepts down the side, one column per company and year.")
                st.dataframe(pivot.head(300), use_container_width=True)
                st.download_button(
                    "Download comparison CSV", to_csv_bytes(pivot),
                    file_name="xbrl_comparison.csv", mime="text/csv",
                )

        with tab_guide:
            for col in data.columns:
                key = "dim:*" if col.startswith("dim:") else col
                st.markdown(f"**{col}** — {COLUMN_DICTIONARY.get(key, '')}")
            st.download_button(
                "Download the column guide", dictionary_bytes(list(data.columns)),
                file_name="column_guide.txt", mime="text/plain",
            )

        with st.expander("Save this selection as a reusable set"):
            st.caption(
                "Sets are keyed on LEI, so they keep working as company names change. "
                "Download this, save it as sets.json in the repo, and it appears in "
                "the set list for everyone."
            )
            label = st.text_input("Set name", value="My set")
            spec = {
                label: {
                    "note": f"{data['company'].nunique()} companies.",
                    "leis": sorted(x for x in data["identifier"].dropna().unique() if x),
                }
            }
            st.download_button(
                "Download sets.json",
                json.dumps(spec, indent=2).encode("utf-8"),
                file_name="sets.json", mime="application/json",
            )


if __name__ == "__main__":
    main()
