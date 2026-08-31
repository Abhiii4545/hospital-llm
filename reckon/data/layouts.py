"""Layout specifications for the synthetic corpus.

Twenty-one layouts across the seven archetypes the brief names. Diversity is
STRUCTURAL, not cosmetic: what varies is column order, where the header block
sits, where the totals block sits, whether rows are grouped under category
subtotals, how many rows fit before a page break, and whether the table is
bordered, ruled or plain.

Cosmetic-only variation (a different font on the same DOM) would teach a model
nothing, because the thing that broke RECKON v1 was not typography. Column order
in particular is varied deliberately: it is exactly what a fixed-order regex
cannot survive.

Hospital names and marks are invented. No real hospital's logo, trademark or
letterhead is reproduced anywhere in this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LayoutSpec", "LAYOUTS", "by_id", "ARCHETYPES"]

ARCHETYPES = (
    "corporate", "grouped", "nursing_home", "government",
    "diagnostic", "pharmacy", "discharge",
)

#: Every column key a template may place. Order within a LayoutSpec is the
#: printed order, and it genuinely differs between layouts.
ALL_COLUMNS = (
    "serial_no", "description", "service_date", "category",
    "quantity", "unit_rate", "amount", "hsn_code", "batch", "expiry",
)


@dataclass(frozen=True)
class LayoutSpec:
    id: str
    archetype: str
    template: str
    columns: tuple[str, ...]
    header_position: str      # banner | split | left | minimal | letterhead
    totals_position: str      # right | left | full | inline
    table_style: str          # bordered | striped | ruled | plain
    rows_per_page: int
    font: str                 # serif | sans | mono | condensed
    hospital_type: str
    bilingual: bool = False
    grouped_subtotals: bool = False
    repeat_header_on_continuation: bool = True


def _spec(**kwargs) -> LayoutSpec:
    return LayoutSpec(**kwargs)


LAYOUTS: tuple[LayoutSpec, ...] = (
    # -- corporate: dense itemised, multi-page ---------------------------
    _spec(id="L01_corporate_dense", archetype="corporate", template="corporate.html.j2",
          columns=("serial_no", "description", "service_date", "quantity", "unit_rate", "amount"),
          header_position="banner", totals_position="right", table_style="bordered",
          rows_per_page=24, font="sans", hospital_type="multi_speciality"),
    _spec(id="L02_corporate_hsn", archetype="corporate", template="corporate.html.j2",
          columns=("serial_no", "description", "hsn_code", "quantity", "unit_rate", "amount"),
          header_position="split", totals_position="right", table_style="striped",
          rows_per_page=20, font="serif", hospital_type="multi_speciality"),
    _spec(id="L03_corporate_rate_first", archetype="corporate", template="corporate.html.j2",
          # rate BEFORE quantity - the permutation a fixed-order regex cannot survive
          columns=("serial_no", "description", "unit_rate", "quantity", "amount", "service_date"),
          header_position="banner", totals_position="left", table_style="bordered",
          rows_per_page=28, font="condensed", hospital_type="multi_speciality",
          repeat_header_on_continuation=False),

    # -- grouped by category with subtotals --------------------------------
    _spec(id="L04_grouped_basic", archetype="grouped", template="grouped.html.j2",
          columns=("serial_no", "description", "quantity", "unit_rate", "amount"),
          header_position="split", totals_position="right", table_style="ruled",
          rows_per_page=18, font="sans", hospital_type="general",
          grouped_subtotals=True),
    _spec(id="L05_grouped_dated", archetype="grouped", template="grouped.html.j2",
          columns=("description", "service_date", "quantity", "amount"),
          header_position="left", totals_position="full", table_style="plain",
          rows_per_page=16, font="serif", hospital_type="general",
          grouped_subtotals=True),
    _spec(id="L06_grouped_amount_first", archetype="grouped", template="grouped.html.j2",
          columns=("amount", "description", "quantity", "unit_rate"),
          header_position="banner", totals_position="left", table_style="striped",
          rows_per_page=22, font="sans", hospital_type="general",
          grouped_subtotals=True),

    # -- small nursing home: sparse, stamped -------------------------------
    _spec(id="L07_nursing_sparse", archetype="nursing_home", template="nursing_home.html.j2",
          columns=("serial_no", "description", "amount"),
          header_position="left", totals_position="inline", table_style="plain",
          rows_per_page=12, font="serif", hospital_type="nursing_home"),
    _spec(id="L08_nursing_qty", archetype="nursing_home", template="nursing_home.html.j2",
          columns=("description", "quantity", "unit_rate", "amount"),
          header_position="minimal", totals_position="inline", table_style="ruled",
          rows_per_page=10, font="mono", hospital_type="nursing_home"),
    _spec(id="L09_nursing_dated", archetype="nursing_home", template="nursing_home.html.j2",
          columns=("service_date", "description", "amount"),
          header_position="left", totals_position="left", table_style="plain",
          rows_per_page=14, font="serif", hospital_type="nursing_home"),

    # -- government: minimal, bilingual ------------------------------------
    _spec(id="L10_govt_telugu", archetype="government", template="government.html.j2",
          columns=("serial_no", "description", "quantity", "unit_rate", "amount"),
          header_position="minimal", totals_position="full", table_style="plain",
          rows_per_page=20, font="sans", hospital_type="government", bilingual=True),
    _spec(id="L11_govt_hindi", archetype="government", template="government.html.j2",
          columns=("serial_no", "description", "amount"),
          header_position="banner", totals_position="full", table_style="ruled",
          rows_per_page=26, font="serif", hospital_type="government", bilingual=True),
    _spec(id="L12_govt_plain", archetype="government", template="government.html.j2",
          columns=("description", "quantity", "amount"),
          header_position="minimal", totals_position="left", table_style="plain",
          rows_per_page=30, font="mono", hospital_type="government",
          repeat_header_on_continuation=False),

    # -- diagnostic centre: panel format -----------------------------------
    _spec(id="L13_diag_panel", archetype="diagnostic", template="diagnostic.html.j2",
          columns=("serial_no", "description", "quantity", "unit_rate", "amount"),
          header_position="letterhead", totals_position="right", table_style="striped",
          rows_per_page=18, font="sans", hospital_type="diagnostic_centre"),
    _spec(id="L14_diag_two_col", archetype="diagnostic", template="diagnostic.html.j2",
          columns=("description", "amount"),
          header_position="letterhead", totals_position="inline", table_style="ruled",
          rows_per_page=24, font="serif", hospital_type="diagnostic_centre"),
    _spec(id="L15_diag_hsn", archetype="diagnostic", template="diagnostic.html.j2",
          columns=("description", "hsn_code", "unit_rate", "quantity", "amount"),
          header_position="split", totals_position="right", table_style="bordered",
          rows_per_page=16, font="condensed", hospital_type="diagnostic_centre"),

    # -- pharmacy sub-bill: batch and expiry -------------------------------
    _spec(id="L16_pharmacy_batch", archetype="pharmacy", template="pharmacy.html.j2",
          columns=("serial_no", "description", "batch", "expiry", "quantity", "unit_rate", "amount"),
          header_position="banner", totals_position="right", table_style="bordered",
          rows_per_page=22, font="condensed", hospital_type="general"),
    _spec(id="L17_pharmacy_hsn", archetype="pharmacy", template="pharmacy.html.j2",
          columns=("description", "hsn_code", "batch", "quantity", "amount"),
          header_position="split", totals_position="full", table_style="striped",
          rows_per_page=26, font="mono", hospital_type="general"),
    _spec(id="L18_pharmacy_expiry_first", archetype="pharmacy", template="pharmacy.html.j2",
          columns=("expiry", "batch", "description", "quantity", "unit_rate", "amount"),
          header_position="minimal", totals_position="right", table_style="ruled",
          rows_per_page=20, font="sans", hospital_type="general"),

    # -- discharge summary with billing annexure ---------------------------
    _spec(id="L19_discharge_annexure", archetype="discharge", template="discharge.html.j2",
          columns=("serial_no", "description", "amount"),
          header_position="letterhead", totals_position="full", table_style="ruled",
          rows_per_page=14, font="serif", hospital_type="multi_speciality"),
    _spec(id="L20_discharge_dense", archetype="discharge", template="discharge.html.j2",
          columns=("serial_no", "description", "quantity", "unit_rate", "amount"),
          header_position="banner", totals_position="right", table_style="bordered",
          rows_per_page=18, font="sans", hospital_type="multi_speciality"),
    _spec(id="L21_discharge_bilingual", archetype="discharge", template="discharge.html.j2",
          columns=("description", "quantity", "amount"),
          header_position="letterhead", totals_position="full", table_style="plain",
          rows_per_page=16, font="serif", hospital_type="general", bilingual=True),
)

_BY_ID = {spec.id: spec for spec in LAYOUTS}


def by_id(layout_id: str) -> LayoutSpec:
    return _BY_ID[layout_id]
