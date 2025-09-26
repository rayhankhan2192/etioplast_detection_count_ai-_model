export default function AIGeneratedReport({ result }) {
  return (
    <div className="max-w-4xl mx-auto mt-8">
      {result?.explanation && (
        <h2 className="text-lg font-semibold text-slate-900 mb-2">
          AI Generated Report
        </h2>
      )}

      {result?.explanation && (
        <div className="bg-slate-50 border border-slate-200 text-slate-700 text-sm leading-relaxed p-4 rounded-xl shadow-sm">
          {result?.explanation}
        </div>
      )}
    </div>
  );
}
