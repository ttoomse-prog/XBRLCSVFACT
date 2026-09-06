"""
Company dashboard
=================

Joins company-level context (industry, employees, market cap) to the tagged
facts extracted from ESEF/UKSEF filings, so you can look at one company in the
round — what it reports, and how it tags.

Drop this file in a `pages/` folder next to filings_to_csv.py and Streamlit
picks it up as a second page automatically. Needs two files in the repo root:

    classroom_dataset.parquet   facts, from the main page
    companies.csv               company metadata, keyed on LEI

Data: filings.xbrl.org (xBRL-JSON via Arelle) joined to company reference data.
"""

from __future__ import annotations

import os
import re

import pandas as pd
import streamlit as st

BASE = "https://filings.xbrl.org"

# fxo_id holds every part of the filing's path: {lei}-{date}-{scheme}-{cc}-{n}
_FXO = re.compile(
    r"^(?P<lei>[A-Z0-9]{20})-(?P<date>\d{4}-\d{2}-\d{2})"
    r"-(?P<scheme>[A-Z]+)-(?P<cc>[A-Z]{2})-(?P<n>\d+)$"
)


def viewer_link(fxo_id: object) -> str | None:
    """Link to the filing in the inline viewer, with tags highlighted."""
    m = _FXO.match(str(fxo_id or ""))
    if not m:
        return None
    d = m.groupdict()
    return (
        f"{BASE}/{d['lei']}/{d['date']}/{d['scheme']}/{d['cc']}/{d['n']}/"
        f"{d['lei']}-{d['date']}/reports/ixbrlviewer.html"
    )


FACTS_FILE = "classroom_dataset.parquet"
COMPANIES_FILE = "companies.csv"

HEADLINE_CONCEPTS = [
    "Revenue", "ProfitLossBeforeTax", "ProfitLoss", "Assets", "Liabilities",
    "Equity", "CashAndCashEquivalents", "IncomeTaxExpenseContinuingOperations",
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    facts = pd.read_parquet(FACTS_FILE)
    companies = pd.read_csv(COMPANIES_FILE)
    return facts, companies


def reporting_currency(facts: pd.DataFrame) -> pd.Series:
    """Most-used unit per company, tidied to a plain currency code."""
    used = facts[facts["unit"].notna()]
    if used.empty:
        return pd.Series(dtype="object")
    return (
        used.groupby("company")["unit"]
        .agg(lambda s: s.value_counts().index[0])
        .str.replace("iso4217:", "", regex=False)
    )


# ----------------------------------------------------------------------------
# Per-company views
# ----------------------------------------------------------------------------

def latest_period(facts: pd.DataFrame, company: str) -> str | None:
    periods = facts.loc[facts["company"] == company, "filing_period_end"].dropna()
    return periods.max() if not periods.empty else None


def headline_figures(facts: pd.DataFrame, company: str, period: str) -> pd.DataFrame:
    """Undimensioned numeric facts for the concepts people actually look up."""
    rows = facts[
        (facts["company"] == company)
        & (facts["filing_period_end"] == period)
        & (facts["n_dimensions"] == 0)
        & (facts["value_numeric"].notna())
        & (facts["concept_name"].isin(HEADLINE_CONCEPTS))
    ]
    if rows.empty:
        return pd.DataFrame()
    # xBRL-JSON writes period ends as the following midnight, so a 31 Dec 2025
    # balance carries 2026-01-01. Step midnight values back a day so the dates
    # read the way the report states them.
    when = pd.to_datetime(
        rows["period_end"].fillna(rows["period_instant"]), errors="coerce"
    )
    at_midnight = (when.dt.hour == 0) & (when.dt.minute == 0) & (when.dt.second == 0)
    when = when.mask(at_midnight, when - pd.Timedelta(days=1))
    rows = rows.assign(_when=when.dt.strftime("%Y-%m-%d"))
    rows = rows.sort_values("_when", ascending=False).drop_duplicates("concept_name")
    return (
        rows[["concept_name", "value_numeric", "unit", "_when", "decimals"]]
        .rename(columns={"concept_name": "Concept", "value_numeric": "Value",
                         "unit": "Unit", "_when": "As at / to",
                         "decimals": "Precision"})
        .sort_values("Concept")
        .reset_index(drop=True)
    )


def company_reports(facts: pd.DataFrame, company: str) -> pd.DataFrame:
    """Every filing held for this company, with a link to the published report."""
    rows = (
        facts[facts["company"] == company][["filing_period_end", "fxo_id"]]
        .drop_duplicates()
        .sort_values("filing_period_end", ascending=False)
    )
    if rows.empty:
        return pd.DataFrame()
    rows = rows.assign(report=rows["fxo_id"].map(viewer_link))
    return rows.rename(columns={"filing_period_end": "Period end"})[
        ["Period end", "report"]
    ].reset_index(drop=True)


def tagging_profile(facts: pd.DataFrame, company: str) -> dict[str, float]:
    rows = facts[facts["company"] == company]
    if rows.empty:
        return {}
    return {
        "facts": len(rows),
        "concepts": rows["concept_name"].nunique(),
        "extension_pct": 100 * rows["is_extension"].mean(),
        "dimensioned_pct": 100 * (rows["n_dimensions"] > 0).mean(),
        "numeric_pct": 100 * rows["value_numeric"].notna().mean(),
    }


def top_extensions(facts: pd.DataFrame, company: str, n: int = 10) -> pd.DataFrame:
    rows = facts[(facts["company"] == company) & (facts["is_extension"])]
    if rows.empty:
        return pd.DataFrame()
    return (
        rows["concept_name"].value_counts().head(n)
        .rename_axis("Company-defined concept").reset_index(name="Times used")
    )


# ----------------------------------------------------------------------------
# Cross-company view
# ----------------------------------------------------------------------------

def comparison_table(facts: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    agg = facts.groupby("company").agg(
        facts=("fact_id", "size"),
        concepts=("concept_name", "nunique"),
        extension_pct=("is_extension", "mean"),
        dimensioned_pct=("n_dimensions", lambda s: (s > 0).mean()),
        latest_filing=("filing_period_end", "max"),
    )
    # the fxo_id of the most recent filing, not merely the last row in the group
    latest = (facts.sort_values("filing_period_end")
                   .drop_duplicates("company", keep="last")
                   .set_index("company")["fxo_id"])
    agg = agg.join(latest.rename("latest_fxo"))
    agg["extension_pct"] = (100 * agg["extension_pct"]).round(1)
    agg["dimensioned_pct"] = (100 * agg["dimensioned_pct"]).round(1)
    agg = agg.join(reporting_currency(facts).rename("currency"))

    out = companies.merge(agg, left_on="company", right_index=True, how="left")
    out["report"] = out["latest_fxo"].map(viewer_link)
    return out[[
        "company", "industry", "employees", "market_cap_m", "share_price",
        "currency", "facts", "concepts", "extension_pct", "dimensioned_pct",
        "latest_filing", "report",
    ]].sort_values("market_cap_m", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Company dashboard", page_icon="🏢", layout="wide")
    st.title("Company dashboard")

    missing = [f for f in (FACTS_FILE, COMPANIES_FILE) if not os.path.exists(f)]
    if missing:
        st.error(
            f"Missing {', '.join(missing)}. Both files need to sit in the repo "
            "root — build the facts file on the main page, and companies.csv "
            "holds the company reference data."
        )
        return

    facts, companies = load_data()
    st.caption(
        f"{len(facts):,} tagged facts from {facts['company'].nunique()} companies, "
        "joined to company reference data. Facts from filings.xbrl.org."
    )

    tab_company, tab_compare, tab_notes = st.tabs(
        ["One company", "All companies", "Reading this well"]
    )

    # ---- single company ----
    with tab_company:
        name = st.selectbox("Company", sorted(companies["company"]))
        row = companies[companies["company"] == name].iloc[0]
        currency = reporting_currency(facts).get(name, "—")

        st.subheader(row["parsed_name"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Industry", row["industry"])
        c2.metric("Employees", f"{int(row['employees']):,}")
        c3.metric("Market cap (m)", f"{row['market_cap_m']:,.0f}")
        c4.metric("Reports in", currency)

        with st.expander("What the company does"):
            st.write(row["description"])

        reports = company_reports(facts, name)
        if not reports.empty:
            st.markdown("**Published reports**")
            st.dataframe(
                reports, use_container_width=True, hide_index=True,
                column_config={
                    "report": st.column_config.LinkColumn(
                        "Annual report", display_text="Open ↗",
                        help="Opens the report with its tagged numbers highlighted.",
                    )
                },
            )
            st.caption(
                "Reading a figure below, then finding it highlighted in the report, "
                "is the quickest way to see what a tag actually means."
            )

        period = latest_period(facts, name)
        st.markdown(f"**Headline figures** — filing to {period}, in {currency}")
        figures = headline_figures(facts, name, period)
        if figures.empty:
            st.info(
                "No undimensioned headline figures found. Banks and insurers often "
                "tag their primary statements differently — look in the full fact "
                "list instead."
            )
        else:
            st.dataframe(figures, use_container_width=True, hide_index=True)
            st.caption(
                "Precision is the reported `decimals` value: -3 means the figure "
                "was given to the nearest thousand, -6 to the nearest million."
            )

        st.markdown("**How this company tags**")
        prof = tagging_profile(facts, name)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Facts tagged", f"{prof['facts']:,}")
        p2.metric("Distinct concepts", f"{prof['concepts']:,}")
        p3.metric("Company-defined", f"{prof['extension_pct']:.1f}%")
        p4.metric("With dimensions", f"{prof['dimensioned_pct']:.1f}%")

        ext = top_extensions(facts, name)
        if not ext.empty:
            st.caption("The concepts this company had to invent for itself:")
            st.dataframe(ext, use_container_width=True, hide_index=True)

    # ---- all companies ----
    with tab_compare:
        table = comparison_table(facts, companies)
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "report": st.column_config.LinkColumn(
                    "Annual report", display_text="Open ↗",
                    help="Most recent filing held, with tags highlighted.",
                )
            },
        )

        st.markdown("**Company-defined concepts, by company**")
        st.bar_chart(
            table.set_index("company")["extension_pct"].sort_values(ascending=False),
            height=320, y_label="% of facts using a company-defined concept",
        )

        st.markdown("**Does size explain how much gets tagged?**")
        st.scatter_chart(
            table, x="market_cap_m", y="facts", color="industry",
            height=380, x_label="Market cap (m)", y_label="Facts tagged",
        )

        st.markdown("**By industry**")
        by_industry = (
            table.groupby("industry")
            .agg(companies=("company", "size"),
                 median_extension_pct=("extension_pct", "median"),
                 median_facts=("facts", "median"))
            .sort_values("median_extension_pct", ascending=False)
            .reset_index()
        )
        st.dataframe(by_industry, use_container_width=True, hide_index=True)

    # ---- guidance ----
    with tab_notes:
        st.markdown(
            """
**Currencies do not match.** These companies report in three different
currencies, so adding revenue across them, or ranking them on it, gives a
meaningless answer. Check the `currency` column before any comparison, or
restrict yourself to one currency at a time. Percentages and counts are safe.

**Market cap and share price come from a different source** to the filings, and
were captured on one day. They are not part of the filing and were not audited.
Treat them as context, not as data to analyse alongside the reported figures.

**A high share of company-defined concepts is not a fault.** It can mean the
business genuinely does something the taxonomy did not anticipate — or that a
company is reporting in its own idiom, which makes it harder to compare with
peers. Reading a few of the invented concept names usually tells you which.

**Fact counts measure disclosure, not quality.** A company can tag thousands of
facts and still tag the important ones badly. If you want to test that, open a
number in the viewer and ask whether the concept chosen is what a reader would
understand the number to mean.

**Employee counts come from the reference data**, not from the filings. Where a
company also tags a number of employees, the two can differ — different dates,
different definitions of who counts. That gap is worth a discussion in itself.
            """
        )


if __name__ == "__main__":
    main()
