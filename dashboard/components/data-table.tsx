import { text } from "@/lib/format";

export function DataTable({ columns, rows }: { columns: string[]; rows: Array<Record<string, unknown>> }) {
  return (
    <div className="overflow-hidden rounded-lg border border-white/10 bg-[#10151d]/88">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] border-collapse text-left text-sm">
          <thead className="bg-white/[0.04] text-[11px] uppercase tracking-[0.14em] text-slate-400">
            <tr>
              {columns.map((column) => (
                <th key={column} className="border-b border-white/10 px-4 py-3 font-semibold">
                  {column.replaceAll("_", " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/8">
            {rows.length ? (
              rows.map((row, index) => (
                <tr key={index} className="hover:bg-white/[0.03]">
                  {columns.map((column) => (
                    <td key={column} className="max-w-[320px] truncate px-4 py-3 text-slate-200">
                      {text(row[column])}
                    </td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-4 py-6 text-slate-400" colSpan={columns.length}>
                  No hay registros disponibles.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
