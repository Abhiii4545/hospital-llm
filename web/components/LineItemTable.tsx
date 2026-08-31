"use client";

import type { AdjudicationResult, LineItem } from "@/lib/api";

const COLUMNS: Array<{ key: keyof LineItem; label: string; numeric?: boolean }> = [
  { key: "serial_no", label: "#" },
  { key: "description", label: "Description" },
  { key: "category", label: "Category" },
  { key: "quantity", label: "Qty", numeric: true },
  { key: "unit_rate", label: "Rate", numeric: true },
  { key: "amount", label: "Amount", numeric: true },
];

/**
 * Line items, with each row showing the deduction that landed on it.
 *
 * Rows are struck through when adjudication removed them entirely, and every
 * deduction carries the rule id and clause that produced it. A reviewer should
 * never have to ask why a line was refused.
 */
export default function LineItemTable({
  items,
  adjudication,
  onEdit,
}: {
  items: LineItem[];
  adjudication: AdjudicationResult | null;
  onEdit: (index: number, key: string, value: string) => void;
}) {
  const byIndex = new Map<number, typeof adjudication extends null ? never : any[]>();
  adjudication?.deductions.forEach((deduction) => {
    if (deduction.line_index === null) return;
    const list = byIndex.get(deduction.line_index) ?? [];
    list.push(deduction);
    byIndex.set(deduction.line_index, list);
  });

  if (!items.length) {
    return <p className="muted">No line items extracted.</p>;
  }

  return (
    <table>
      <thead>
        <tr>
          {COLUMNS.map((column) => (
            <th key={String(column.key)} className={column.numeric ? "num" : ""}>
              {column.label}
            </th>
          ))}
          <th>Adjudication</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item, index) => {
          const deductions = byIndex.get(index) ?? [];
          const removed = deductions.some((d: any) => d.rule_id.startsWith("LIST_I"));
          return (
            <tr key={index} className={removed ? "dropped" : ""}>
              {COLUMNS.map((column) => (
                <td key={String(column.key)} className={column.numeric ? "num" : ""}>
                  <input
                    value={(item[column.key] as string) ?? ""}
                    style={{
                      border: "none",
                      background: "transparent",
                      padding: "2px 3px",
                      textAlign: column.numeric ? "right" : "left",
                    }}
                    onChange={(event) =>
                      onEdit(index, String(column.key), event.target.value)
                    }
                  />
                </td>
              ))}
              <td>
                {deductions.length === 0 ? (
                  <span className="muted">—</span>
                ) : (
                  deductions.map((d: any, i: number) => (
                    <span
                      key={i}
                      className="pill ded"
                      title={`${d.clause}\n${d.reason}`}
                      style={{ marginRight: 4 }}
                    >
                      {d.rule_id} −{d.amount}
                    </span>
                  ))
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
