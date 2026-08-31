import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RECKON v2 — claim review",
  description:
    "Human-in-the-loop review and IRDAI adjudication for Indian hospital bills.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
