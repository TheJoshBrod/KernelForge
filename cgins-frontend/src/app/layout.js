import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata = {
  title: "CGinS",
  description: "CGinS Project Interface",
};

import { ConfigProvider } from "../context/ConfigContext";
import { Toaster } from "sonner";

// ... existing code ...

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased bg-zinc-950 text-zinc-100`}>
        <ConfigProvider>
          {children}
          <Toaster richColors theme="dark" />
        </ConfigProvider>
      </body>
    </html>
  );
}
