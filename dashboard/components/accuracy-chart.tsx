"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function AccuracyChart({ data }: { data: Array<Record<string, number | string | null>> }) {
  return (
    <div className="h-[300px] rounded-lg border border-white/10 bg-[#10151d]/88 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
          <defs>
            <linearGradient id="accuracy" x1="0" x2="0" y1="0" y2="1">
              <stop offset="5%" stopColor="#41d19b" stopOpacity={0.42} />
              <stop offset="95%" stopColor="#41d19b" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
          <XAxis dataKey="cycle_id" tick={{ fill: "#90a0ae", fontSize: 12 }} tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis domain={[70, 100]} tick={{ fill: "#90a0ae", fontSize: 12 }} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ background: "#10151d", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, color: "#edf4f2" }}
            formatter={(value) => [`${Number(value).toFixed(1)}%`, "accuracy"]}
          />
          <Area type="monotone" dataKey="cycle_accuracy" stroke="#41d19b" strokeWidth={2.5} fill="url(#accuracy)" />
          <Area type="monotone" dataKey="rolling6_accuracy" stroke="#4dbed5" strokeWidth={2} fill="transparent" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
