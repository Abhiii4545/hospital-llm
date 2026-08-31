"use client";

import type { ReckonDocument } from "@/lib/api";

const BLOCKS: Array<{ key: keyof ReckonDocument; title: string }> = [
  { key: "patient", title: "Patient" },
  { key: "hospital", title: "Hospital" },
  { key: "insurance", title: "Insurance" },
  { key: "totals", title: "Totals" },
];

/**
 * Editable extracted fields.
 *
 * Every edit is a labelled example from the real distribution - the one the
 * synthetic corpus only approximates - so corrections are logged rather than
 * merely applied. An edited field is marked so the reviewer can see at a glance
 * what they have already touched.
 */
export default function FieldEditor({
  document,
  edited,
  onChange,
  onFocusField,
  activeField,
}: {
  document: ReckonDocument;
  edited: Set<string>;
  onChange: (path: string, value: string) => void;
  onFocusField: (path: string, value: string | null) => void;
  activeField: string | null;
}) {
  return (
    <>
      {BLOCKS.map(({ key, title }) => {
        const block = document[key] as Record<string, string | null>;
        if (!block) return null;
        return (
          <div key={key} style={{ marginBottom: 16 }}>
            <h2>{title}</h2>
            {Object.entries(block).map(([name, value]) => {
              const path = `${key}.${name}`;
              const isEdited = edited.has(path);
              return (
                <div
                  className={`field ${isEdited ? "edited" : ""}`}
                  key={path}
                  style={
                    activeField === path
                      ? { background: "#1d2734", borderRadius: 6 }
                      : undefined
                  }
                >
                  <label htmlFor={path}>{name}</label>
                  <input
                    id={path}
                    value={value ?? ""}
                    placeholder="— not extracted —"
                    onFocus={() => onFocusField(path, value)}
                    onChange={(event) => onChange(path, event.target.value)}
                  />
                  <span className="pill">
                    {isEdited ? "edited" : value ? "" : "missing"}
                  </span>
                </div>
              );
            })}
          </div>
        );
      })}
    </>
  );
}
