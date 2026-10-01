import type { Metadata } from "next";
import "./globals.css";
import { Shell } from "@/components/shell";
import { getMonitoringSnapshot } from "@/lib/monitoring";

export const metadata: Metadata = {
  title: "Centro de control MLOps Pulso TransMi",
  description: "Dashboard operativo de MLOps para Pulso TransMi.",
};

export const revalidate = 60;

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const snapshot = await getMonitoringSnapshot();
  return (
    <html lang="es">
      <body>
        <Shell snapshot={snapshot}>{children}</Shell>
      </body>
    </html>
  );
}
