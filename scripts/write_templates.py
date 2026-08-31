"""One-shot generator for the Jinja layout templates.

Kept in the repo so the templates are regenerable and reviewable as a set rather
than as eight files edited independently. Run: python scripts/write_templates.py
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES = Path("reckon/data/templates")

FILES: dict[str, str] = {}

FILES["_base.html.j2"] = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  @page { size: A4; margin: 0; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 34px 40px 40px 40px; background: #fff; color: #111;
    font-family: {{ fontstack }}; font-size: {{ base_size }}px; line-height: 1.35;
    width: 794px; min-height: 1123px; position: relative;
  }
  .rule { border-bottom: 1.5px solid #000; margin: 6px 0 10px; }
  .thin { border-bottom: 1px solid #666; margin: 5px 0 8px; }
  .banner { text-align: center; }
  .banner .org { font-size: {{ base_size + 7 }}px; font-weight: 700; letter-spacing: .5px; }
  .org-sub { font-size: {{ base_size - 1 }}px; color: #333; }
  .split { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; }
  .split .org { font-size: {{ base_size + 5 }}px; font-weight: 700; }
  .doctitle { text-align: center; font-weight: 700; letter-spacing: 2px;
              margin: 10px 0 8px; font-size: {{ base_size + 1 }}px; }
  .meta { width: 100%; border-collapse: collapse; margin-bottom: 10px; }
  .meta td { padding: 1.5px 6px 1.5px 0; vertical-align: top; font-size: {{ base_size - 1 }}px; }
  .meta .k { color: #222; white-space: nowrap; }
  .meta .v { font-weight: 600; padding-right: 22px; }
  table.items { width: 100%; border-collapse: collapse; margin-top: 4px;
                font-size: {{ base_size - 1 }}px; }
  table.items th { text-align: left; font-weight: 700; padding: 3px 5px; }
  table.items td { padding: 2.5px 5px; vertical-align: top; }
  table.items td.num, table.items th.num { text-align: right; white-space: nowrap; }
  table.items td.ctr, table.items th.ctr { text-align: center; }
  {% if table_style == 'bordered' %}
    table.items, table.items th, table.items td { border: 1px solid #333; }
    table.items th { background: #ececec; }
  {% elif table_style == 'striped' %}
    table.items th { border-bottom: 2px solid #333; background: #f2f2f2; }
    table.items tbody tr:nth-child(even) { background: #f7f7f7; }
    table.items td { border-bottom: 1px solid #ddd; }
  {% elif table_style == 'ruled' %}
    table.items th { border-bottom: 1.5px solid #000; }
    table.items td { border-bottom: 1px dotted #999; }
  {% else %}
    table.items th { border-bottom: 1px solid #000; }
  {% endif %}
  .grp { font-weight: 700; padding-top: 7px; }
  .sub { font-weight: 600; border-top: 1px solid #999; }
  .totals { margin-top: 12px; font-size: {{ base_size - 1 }}px; }
  .totals table { border-collapse: collapse; }
  .totals td { padding: 2px 8px; }
  .totals td.k { text-align: left; }
  .totals td.v { text-align: right; font-weight: 600; white-space: nowrap;
                 min-width: 108px; }
  .totals .net td { border-top: 1.5px solid #000; font-weight: 700; }
  .t-right { display: flex; justify-content: flex-end; }
  .t-left  { display: flex; justify-content: flex-start; }
  .t-full table { width: 100%; }
  .t-inline td { padding: 1px 6px; }
  .foot { position: absolute; bottom: 26px; left: 40px; right: 40px;
          font-size: {{ base_size - 3 }}px; color: #444;
          display: flex; justify-content: space-between; }
  .cont { font-style: italic; font-size: {{ base_size - 2 }}px; color: #444;
          margin-bottom: 6px; }
  .hand { position: absolute; font-family: 'Segoe Script','Bradley Hand',cursive;
          color: #123a8f; transform: rotate(-6deg); font-size: {{ base_size + 3 }}px; }
  .stamp { position: absolute; right: 54px; bottom: 96px; border: 2.5px solid #1b5e20;
           color: #1b5e20; padding: 7px 13px; font-size: {{ base_size - 1 }}px;
           font-weight: 700; transform: rotate(-11deg); opacity: .75;
           border-radius: 4px; letter-spacing: 1px; }
  .bi { font-size: {{ base_size - 1 }}px; }
  {% block extra_css %}{% endblock %}
</style></head>
<body>
{% block content %}{% endblock %}
{% if page.continued %}{% endif %}
<div class="foot">
  <div>{{ ctx.invoice_no }}</div>
  <div>Page {{ page.index + 1 }} of {{ page.total_pages }}</div>
</div>
{% if ctx.handwritten and page.show_totals %}
  <div class="hand" style="left: 58%; bottom: 150px;">corrected {{ ctx.totals.net_amount }}</div>
{% endif %}
{% if stamped and page.show_totals %}<div class="stamp">PAID</div>{% endif %}
</body></html>
"""

FILES["_macros.html.j2"] = """{% macro items_table(spec, rows, labels) %}
<table class="items">
  <thead><tr>
    {% for col in spec.columns %}
      <th class="{{ 'num' if col in ('amount','unit_rate') else ('ctr' if col in ('quantity','serial_no') else '') }}">{{ labels[col] }}</th>
    {% endfor %}
  </tr></thead>
  <tbody>
  {% for row in rows %}
    {% if row.get('__group__') %}
      <tr><td class="grp" colspan="{{ spec.columns|length }}">{{ row['__group__'] }}</td></tr>
    {% elif row.get('__subtotal__') %}
      <tr class="sub">
        <td colspan="{{ spec.columns|length - 1 }}">{{ row['__subtotal__'] }}</td>
        <td class="num">{{ row['amount'] }}</td>
      </tr>
    {% else %}
      <tr>
      {% for col in spec.columns %}
        <td class="{{ 'num' if col in ('amount','unit_rate') else ('ctr' if col in ('quantity','serial_no') else '') }}">{{ row[col] }}</td>
      {% endfor %}
      </tr>
    {% endif %}
  {% endfor %}
  </tbody>
</table>
{% endmacro %}

{% macro totals_block(spec, ctx, page) %}
{% if page.show_totals %}
<div class="totals {{ 't-right' if spec.totals_position == 'right' else ('t-left' if spec.totals_position == 'left' else ('t-full' if spec.totals_position == 'full' else 't-inline')) }}">
  <table>
    <tr><td class="k">Gross Amount</td><td class="v">{{ ctx.totals.gross_amount }}</td></tr>
    <tr><td class="k">Discount</td><td class="v">{{ ctx.totals.discount }}</td></tr>
    <tr><td class="k">CGST</td><td class="v">{{ ctx.totals.cgst }}</td></tr>
    <tr><td class="k">SGST</td><td class="v">{{ ctx.totals.sgst }}</td></tr>
    <tr class="net"><td class="k">Net Amount</td><td class="v">{{ ctx.totals.net_amount }}</td></tr>
    <tr><td class="k">Advance Paid</td><td class="v">{{ ctx.totals.advance_paid }}</td></tr>
    <tr><td class="k">Balance Due</td><td class="v">{{ ctx.totals.balance_due }}</td></tr>
  </table>
</div>
{% endif %}
{% endmacro %}

{% macro meta_pairs(pairs, per_row) %}
<table class="meta"><tbody>
{% for chunk in pairs|batch(per_row) %}
  <tr>{% for k, v in chunk %}<td class="k">{{ k }}</td><td class="v">{{ v }}</td>{% endfor %}</tr>
{% endfor %}
</tbody></table>
{% endmacro %}
"""

_CONT = """{% if page.continued %}<div class="cont">Continued from page {{ page.index }} &mdash; {{ ctx.hospital.name }}</div>{% endif %}"""

FILES["corporate.html.j2"] = """{% extends "_base.html.j2" %}
{% import "_macros.html.j2" as m %}
{% block content %}
{% if page.show_header or spec.repeat_header_on_continuation %}
  {% if spec.header_position == 'split' %}
    <div class="split">
      <div><div class="org">""" + "{{ ctx.hospital.name }}" + """</div>
        <div class="org-sub">{{ ctx.hospital.address }}, {{ ctx.hospital.city }}</div>
        <div class="org-sub">{{ ctx.hospital.state }}</div></div>
      <div style="text-align:right" class="org-sub">
        GSTIN: {{ ctx.hospital.gstin }}<br>Invoice: {{ ctx.invoice_no }}</div>
    </div>
  {% else %}
    <div class="banner">
      <div class="org">{{ ctx.hospital.name }}</div>
      <div class="org-sub">{{ ctx.hospital.address }}, {{ ctx.hospital.city }}, {{ ctx.hospital.state }}</div>
      <div class="org-sub">GSTIN: {{ ctx.hospital.gstin }}</div>
    </div>
  {% endif %}
  <div class="rule"></div>
{% endif %}
""" + _CONT + """
{% if page.show_header %}
  <div class="doctitle">FINAL BILL OF SUPPLY</div>
  {{ m.meta_pairs([
      ('Patient Name', ctx.patient.name), ('UHID', ctx.patient.uhid or '-'),
      ('Age', ctx.patient.age), ('Sex', ctx.patient.sex),
      ('IP No', ctx.patient.ip_number), ('Ward Type', ctx.patient.ward_type),
      ('Admission Date', ctx.patient.admission_date), ('Discharge Date', ctx.patient.discharge_date),
      ('Insurer', ctx.insurance.insurer_name), ('TPA', ctx.insurance.tpa_name),
      ('Policy No', ctx.insurance.policy_number), ('Claim No', ctx.insurance.claim_number),
    ], 2) }}
  <div class="thin"></div>
{% endif %}
{{ m.items_table(spec, rows, labels) }}
{{ m.totals_block(spec, ctx, page) }}
{% endblock %}
"""

FILES["grouped.html.j2"] = """{% extends "_base.html.j2" %}
{% import "_macros.html.j2" as m %}
{% block content %}
{% if spec.header_position == 'left' %}
  <div class="org" style="font-size:{{ base_size + 5 }}px;font-weight:700">{{ ctx.hospital.name }}</div>
  <div class="org-sub">{{ ctx.hospital.address }} &middot; {{ ctx.hospital.city }} &middot; {{ ctx.hospital.state }}</div>
  <div class="org-sub">GST No {{ ctx.hospital.gstin }}</div>
{% elif spec.header_position == 'split' %}
  <div class="split">
    <div><div class="org">{{ ctx.hospital.name }}</div>
      <div class="org-sub">{{ ctx.hospital.city }}, {{ ctx.hospital.state }}</div></div>
    <div class="org-sub" style="text-align:right">GST No {{ ctx.hospital.gstin }}</div>
  </div>
{% else %}
  <div class="banner"><div class="org">{{ ctx.hospital.name }}</div>
    <div class="org-sub">{{ ctx.hospital.city }} &middot; GST No {{ ctx.hospital.gstin }}</div></div>
{% endif %}
<div class="rule"></div>
""" + _CONT + """
{% if page.show_header %}
  {{ m.meta_pairs([
      ('Name', ctx.patient.name), ('UHID', ctx.patient.uhid or 'NOT ISSUED'),
      ('IP No', ctx.patient.ip_number), ('Ward', ctx.patient.ward_type),
      ('DOA', ctx.patient.admission_date), ('DOD', ctx.patient.discharge_date),
      ('Insurance', ctx.insurance.insurer_name), ('TPA', ctx.insurance.tpa_name),
      ('Policy', ctx.insurance.policy_number), ('Claim', ctx.insurance.claim_number),
    ], 2) }}
  <div class="thin"></div>
  <div class="doctitle">STATEMENT OF CHARGES BY CATEGORY</div>
{% endif %}
{{ m.items_table(spec, rows, labels) }}
{{ m.totals_block(spec, ctx, page) }}
{% endblock %}
"""

FILES["nursing_home.html.j2"] = """{% extends "_base.html.j2" %}
{% import "_macros.html.j2" as m %}
{% block extra_css %}
  body { padding: 46px 52px; }
  .meta td { padding: 2.5px 6px 2.5px 0; }
{% endblock %}
{% block content %}
{% if spec.header_position == 'minimal' %}
  <div style="font-weight:700">{{ ctx.hospital.name }}</div>
  <div class="org-sub">{{ ctx.hospital.city }} | GSTIN {{ ctx.hospital.gstin }}</div>
{% else %}
  <div class="org" style="font-size:{{ base_size + 4 }}px;font-weight:700">{{ ctx.hospital.name }}</div>
  <div class="org-sub">{{ ctx.hospital.address }}</div>
  <div class="org-sub">{{ ctx.hospital.city }} - {{ ctx.hospital.state }}</div>
  <div class="org-sub">GSTIN {{ ctx.hospital.gstin }}</div>
{% endif %}
<div class="thin"></div>
""" + _CONT + """
{% if page.show_header %}
  <div class="doctitle">BILL</div>
  <table class="meta"><tbody>
    <tr><td class="k">Patient</td><td class="v">{{ ctx.patient.name }}</td></tr>
    <tr><td class="k">Age / Sex</td><td class="v">{{ ctx.patient.age }} / {{ ctx.patient.sex }}</td></tr>
    <tr><td class="k">UHID</td><td class="v">{{ ctx.patient.uhid or '' }}</td></tr>
    <tr><td class="k">IP No</td><td class="v">{{ ctx.patient.ip_number }}</td></tr>
    <tr><td class="k">Ward</td><td class="v">{{ ctx.patient.ward_type }}</td></tr>
    <tr><td class="k">Admitted</td><td class="v">{{ ctx.patient.admission_date }}</td></tr>
    <tr><td class="k">Discharged</td><td class="v">{{ ctx.patient.discharge_date }}</td></tr>
    <tr><td class="k">Insurer</td><td class="v">{{ ctx.insurance.insurer_name }}</td></tr>
    <tr><td class="k">Policy</td><td class="v">{{ ctx.insurance.policy_number }}</td></tr>
  </tbody></table>
{% endif %}
{{ m.items_table(spec, rows, labels) }}
{{ m.totals_block(spec, ctx, page) }}
{% endblock %}
"""

FILES["government.html.j2"] = """{% extends "_base.html.j2" %}
{% import "_macros.html.j2" as m %}
{% block extra_css %} body { padding: 30px 34px; } {% endblock %}
{% block content %}
<div class="{{ 'banner' if spec.header_position == 'banner' else '' }}">
  <div class="bi">{{ bi.govt }}</div>
  <div class="org" style="font-weight:700;font-size:{{ base_size + 3 }}px">{{ ctx.hospital.name }}</div>
  <div class="org-sub">{{ ctx.hospital.address }}, {{ ctx.hospital.city }}</div>
  <div class="org-sub">GSTIN {{ ctx.hospital.gstin }}</div>
</div>
<div class="thin"></div>
""" + _CONT + """
{% if page.show_header %}
  <div class="bi" style="font-weight:700">{{ bi.patient_details }}</div>
  <table class="meta"><tbody>
    <tr><td class="k">{{ bi.name }}</td><td class="v">{{ ctx.patient.name }}</td>
        <td class="k">{{ bi.age }}</td><td class="v">{{ ctx.patient.age }}</td></tr>
    <tr><td class="k">{{ bi.sex }}</td><td class="v">{{ ctx.patient.sex }}</td>
        <td class="k">UHID</td><td class="v">{{ ctx.patient.uhid or '--' }}</td></tr>
    <tr><td class="k">IP No</td><td class="v">{{ ctx.patient.ip_number }}</td>
        <td class="k">{{ bi.ward }}</td><td class="v">{{ ctx.patient.ward_type }}</td></tr>
    <tr><td class="k">{{ bi.admission }}</td><td class="v">{{ ctx.patient.admission_date }}</td>
        <td class="k">{{ bi.discharge }}</td><td class="v">{{ ctx.patient.discharge_date }}</td></tr>
    <tr><td class="k">Insurer</td><td class="v">{{ ctx.insurance.insurer_name }}</td>
        <td class="k">Claim</td><td class="v">{{ ctx.insurance.claim_number }}</td></tr>
  </tbody></table>
  <div class="bi" style="font-weight:700;margin-top:6px">{{ bi.details }}</div>
{% endif %}
{{ m.items_table(spec, rows, labels) }}
{{ m.totals_block(spec, ctx, page) }}
{% endblock %}
"""

FILES["diagnostic.html.j2"] = """{% extends "_base.html.j2" %}
{% import "_macros.html.j2" as m %}
{% block extra_css %}
  .lh { text-align:center; border-bottom: 3px double #222; padding-bottom: 8px; }
  .lh .org { font-size: {{ base_size + 8 }}px; font-weight: 700; letter-spacing: 1px; }
{% endblock %}
{% block content %}
{% if spec.header_position == 'split' %}
  <div class="split"><div><div class="org" style="font-size:{{ base_size+5 }}px;font-weight:700">{{ ctx.hospital.name }}</div>
    <div class="org-sub">{{ ctx.hospital.city }}, {{ ctx.hospital.state }}</div></div>
    <div class="org-sub" style="text-align:right">GSTIN: {{ ctx.hospital.gstin }}</div></div>
  <div class="rule"></div>
{% else %}
  <div class="lh">
    <div class="org">{{ ctx.hospital.name }}</div>
    <div class="org-sub">{{ ctx.hospital.address }}, {{ ctx.hospital.city }}, {{ ctx.hospital.state }}</div>
    <div class="org-sub">GSTIN: {{ ctx.hospital.gstin }}</div>
  </div>
{% endif %}
""" + _CONT + """
{% if page.show_header %}
  <div class="doctitle">TEST PANEL INVOICE</div>
  {{ m.meta_pairs([
      ('Patient Name', ctx.patient.name), ('UHID', ctx.patient.uhid or 'N/A'),
      ('Age', ctx.patient.age), ('Sex', ctx.patient.sex),
      ('Referred IP', ctx.patient.ip_number), ('Ward Type', ctx.patient.ward_type),
      ('Collected On', ctx.patient.admission_date), ('Reported On', ctx.patient.discharge_date),
      ('Insurer', ctx.insurance.insurer_name), ('TPA', ctx.insurance.tpa_name),
      ('Policy No', ctx.insurance.policy_number), ('Claim No', ctx.insurance.claim_number),
    ], 2) }}
  <div class="thin"></div>
{% endif %}
{{ m.items_table(spec, rows, labels) }}
{{ m.totals_block(spec, ctx, page) }}
{% endblock %}
"""

FILES["pharmacy.html.j2"] = """{% extends "_base.html.j2" %}
{% import "_macros.html.j2" as m %}
{% block extra_css %} table.items { font-size: {{ base_size - 2 }}px; } {% endblock %}
{% block content %}
{% if spec.header_position == 'split' %}
  <div class="split"><div><div class="org" style="font-size:{{ base_size+4 }}px;font-weight:700">{{ ctx.hospital.name }} &mdash; Pharmacy</div>
    <div class="org-sub">{{ ctx.hospital.city }}</div></div>
    <div class="org-sub" style="text-align:right">DL No 21B/{{ ctx.invoice_no[-4:] }}<br>GSTIN {{ ctx.hospital.gstin }}</div></div>
{% elif spec.header_position == 'minimal' %}
  <div style="font-weight:700">{{ ctx.hospital.name }} &mdash; Pharmacy Sub-Bill</div>
  <div class="org-sub">GSTIN {{ ctx.hospital.gstin }}</div>
{% else %}
  <div class="banner"><div class="org">{{ ctx.hospital.name }}</div>
    <div class="org-sub">PHARMACY SUB-BILL &middot; {{ ctx.hospital.city }} &middot; GSTIN {{ ctx.hospital.gstin }}</div></div>
{% endif %}
<div class="rule"></div>
""" + _CONT + """
{% if page.show_header %}
  {{ m.meta_pairs([
      ('Patient', ctx.patient.name), ('UHID', ctx.patient.uhid or '-'),
      ('IP No', ctx.patient.ip_number), ('Ward', ctx.patient.ward_type),
      ('Issued From', ctx.patient.admission_date), ('Issued To', ctx.patient.discharge_date),
      ('Insurer', ctx.insurance.insurer_name), ('Claim No', ctx.insurance.claim_number),
    ], 2) }}
  <div class="thin"></div>
{% endif %}
{{ m.items_table(spec, rows, labels) }}
{{ m.totals_block(spec, ctx, page) }}
{% endblock %}
"""

FILES["discharge.html.j2"] = """{% extends "_base.html.j2" %}
{% import "_macros.html.j2" as m %}
{% block extra_css %}
  .lh { text-align:center; border-bottom: 2px solid #222; padding-bottom: 7px; }
  .clin { font-size: {{ base_size - 1 }}px; margin: 8px 0 4px; }
  .clin b { display:inline-block; min-width: 122px; }
{% endblock %}
{% block content %}
{% if spec.header_position == 'banner' %}
  <div class="banner"><div class="org" style="font-size:{{ base_size+6 }}px;font-weight:700">{{ ctx.hospital.name }}</div>
    <div class="org-sub">{{ ctx.hospital.address }}, {{ ctx.hospital.city }} &middot; GSTIN {{ ctx.hospital.gstin }}</div></div>
{% else %}
  <div class="lh"><div class="org" style="font-size:{{ base_size+6 }}px;font-weight:700">{{ ctx.hospital.name }}</div>
    <div class="org-sub">{{ ctx.hospital.address }}, {{ ctx.hospital.city }}, {{ ctx.hospital.state }}</div>
    <div class="org-sub">GSTIN {{ ctx.hospital.gstin }}</div></div>
{% endif %}
""" + _CONT + """
{% if page.show_header %}
  <div class="doctitle">{% if ctx.bilingual %}{{ bi.discharge_summary }}{% else %}DISCHARGE SUMMARY{% endif %}</div>
  <div class="clin"><b>Patient</b> {{ ctx.patient.name }} &nbsp; <b>Age/Sex</b> {{ ctx.patient.age }}/{{ ctx.patient.sex }}</div>
  <div class="clin"><b>UHID</b> {{ ctx.patient.uhid or '--' }} &nbsp; <b>IP No</b> {{ ctx.patient.ip_number }}</div>
  <div class="clin"><b>Ward</b> {{ ctx.patient.ward_type }}</div>
  <div class="clin"><b>Admitted</b> {{ ctx.patient.admission_date }} &nbsp; <b>Discharged</b> {{ ctx.patient.discharge_date }}</div>
  <div class="clin"><b>Insurer / TPA</b> {{ ctx.insurance.insurer_name }} / {{ ctx.insurance.tpa_name }}</div>
  <div class="clin"><b>Policy / Claim</b> {{ ctx.insurance.policy_number }} / {{ ctx.insurance.claim_number }}</div>
  <div class="clin"><b>Diagnosis</b> {{ diagnosis }}</div>
  <div class="clin"><b>Condition</b> Stable at discharge. Advised review after 7 days.</div>
  <div class="thin"></div>
  <div style="font-weight:700;margin-bottom:3px">ANNEXURE &mdash; BILLING DETAILS</div>
{% endif %}
{{ m.items_table(spec, rows, labels) }}
{{ m.totals_block(spec, ctx, page) }}
{% endblock %}
"""


def main() -> None:
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    for name, body in FILES.items():
        (TEMPLATES / name).write_text(body, encoding="utf-8")
    print(f"wrote {len(FILES)} templates to {TEMPLATES}")


if __name__ == "__main__":
    main()
