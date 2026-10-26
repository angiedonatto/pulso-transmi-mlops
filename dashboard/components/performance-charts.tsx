"use client";
import { TrendChart, BreakdownCharts } from "@/components/analytics";
import type { GroupMetric, PerformancePoint } from "@/lib/performance";
export function PerformanceCharts({history,stations,horizons}:{history:PerformancePoint[];stations:GroupMetric[];horizons:GroupMetric[]}) { return <div className="space-y-5"><TrendChart history={history}/><BreakdownCharts stations={stations} horizons={horizons}/></div>; }
