import type { Metadata } from "next";
import "./globals.css";
import { Shell } from "@/components/shell";
import { getMonitoringSnapshot } from "@/lib/monitoring";
import { snapshotIsStale } from "@/lib/snapshot";

export const metadata: Metadata = {
  title: "Centro de control MLOps Pulso TransMi",
  description: "Dashboard operativo de MLOps para Pulso TransMi.",
};

export const dynamic = "force-dynamic";

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const snapshot = await getMonitoringSnapshot();
  return (
    <html lang="es">
      <body>
        <Shell monitoring={snapshot} stale={snapshot.snapshot ? snapshotIsStale(snapshot.snapshot.generated_at) : true}>{children}</Shell>
      </body>
    </html>
  );
}
