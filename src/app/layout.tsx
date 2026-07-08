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
    "Configure a from-scratch, Gemma-4-Nano-style small language model — attention, RoPE, KV-cache sharing, training hyperparameters — and export a ready-to-train Python config.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-[#09090b] font-sans text-white antialiased">
        {children}
      </body>
    </html>
  );
}
