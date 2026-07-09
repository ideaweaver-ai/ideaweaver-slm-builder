import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "IdeaWeaver SLM Builder | Configure a Small Language Model",
  description:
    "Configure a from-scratch, Gemma-4-Nano-style small language model — attention, RoPE, KV-cache sharing, training hyperparameters — and actually train it on TinyStories.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      {/* suppressHydrationWarning: browser extensions (Grammarly, etc.) inject
          attributes into <body> before React hydrates — that's a benign
          mismatch, not an app bug. */}
      <body className="bg-[#09090b] font-sans text-white antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
