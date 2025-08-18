// QuantificationPanel.jsx
import { useMemo, useState } from "react";
import AIGeneratedReport from "./AIGeneratedReport";

export default function QuantificationPanel({ result, activeIndex = 0 }) {
  const [downloading, setDownloading] = useState(false);

  const isFolder = Array.isArray(result?.results) && result.results.length > 0;
  const masterZipUrl = isFolder ? result?.master_zip_url : null; // <-- use server-provided URL

  // Normalize to an array; we can index everything with activeIndex
  const items = useMemo(() => {
    if (isFolder) {
      return result.results.map((r, i) => ({
        label: r?.file || `Image ${i + 1}`,
        analysis: r?.analysis || {},
        csvUrl: r?.output_urls?.measurements_csv || null,
        imageUrl:
          r?.output_urls?.contours || r?.output_image_url || r?.output_urls?.blended || null,
        explanation: r?.explanation || null,
        output_urls: r?.output_urls || {},
        save_dir_url: r?.save_dir_url || null,
      }));
    }
    if (result && (result.analysis || result.output_urls)) {
      return [
        {
          label: "Image",
          analysis: result.analysis || {},
          csvUrl: result?.output_urls?.measurements_csv || null,
          imageUrl:
            result?.output_urls?.contours ||
            result?.output_image_url ||
            result?.output_urls?.blended ||
            null,
          explanation: result?.explanation || null,
          output_urls: result?.output_urls || {},
          save_dir_url: result?.save_dir_url || null,
        },
      ];
    }
    return [];
  }, [isFolder, result]);

  const active =
    items[Math.min(activeIndex, Math.max(items.length - 1, 0))] || {
      analysis: {},
      csvUrl: null,
      imageUrl: null,
      explanation: null,
    };
  const A = active.analysis || {};

  // ---- ZIP download: use master_zip_url from analyze-folder JSON ----
  const handleDownloadZip = async () => {
    if (!isFolder || !masterZipUrl) return;
    try {
      setDownloading(true);
      // Trigger browser download directly from the URL returned by the backend
      const a = document.createElement("a");
      a.href = masterZipUrl;
      a.download = ""; // let server/content-disposition decide
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error("ZIP open failed:", e);
      alert("Could not open the ZIP link. See console for details.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <section className="py-12 pt-2">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-slate-200/50 w-full">
        <div className="flex items-center gap-3 mb-6">
          <div className="bg-emerald-100 p-2 rounded-lg">
            <svg className="h-5 w-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-slate-900">📊 Quantified Results</h2>

          {/* CSV for the active image + ZIP for all (from master_zip_url) */}
          <div className="ml-auto flex items-center gap-2">
            {active.csvUrl ? (
              <a
                href={active.csvUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold py-2 px-3 rounded-lg"
              >
                Download CSV (Current)
              </a>
            ) : (
              <button
                disabled
                className="inline-flex items-center gap-2 bg-slate-200 text-slate-400 text-sm font-semibold py-2 px-3 rounded-lg cursor-not-allowed"
                title="No CSV available"
              >
                Download CSV (Current)
              </button>
            )}

            <button
              onClick={handleDownloadZip}
              disabled={!isFolder || !masterZipUrl || downloading}
              className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white text-sm font-semibold py-2 px-3 rounded-lg"
              title={isFolder ? (masterZipUrl ? "Download all results as ZIP" : "ZIP not available") : "ZIP available for folder mode"}
            >
              {downloading ? "Opening ZIP…" : "Download ZIP (All)"}
            </button>
          </div>
        </div>

        {/* Metrics grid (active image) */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <QuantCard color="yellow" label="Etioplast Area" value={`${A?.Etioplast?.total_area_um2 ?? 0} µm²`} />
          <QuantCard color="purple" label="PLB Area" value={`${A?.PLB?.total_area_um2 ?? 0} µm²`} />
          <QuantCard color="red" label="Prothylakoid Count" value={`${A?.Prothylakoid?.count ?? 0}`} />
          <QuantCard color="red" label="Total Prothylakoid Length" value={`${A?.Prothylakoid?.total_length_um ?? 0} µm`} />
          <QuantCard color="blue" label="Plastoglobule Count" value={`${A?.Plastoglobule?.count ?? 0}`} />
          <QuantCard color="blue" label="Avg. Plastoglobule Diameter" value={`${A?.Plastoglobule?.diameter_um ?? 0} µm`} />
          <QuantCard color="blue" label="Starch Grain Count" value={`${A?.StarchGrain?.count ?? A?.StarchGain?.count ?? 0}`} />
          <QuantCard color="blue" label="Starch Grain Area" value={`${A?.StarchGrain?.total_area_um2 ?? A?.StarchGain?.total_area_um2 ?? 0} µm²`} />
        </div>

        {/* AI result UNDER the quantification */}
        {active.explanation && (
          <div className="mt-6">
            <AIGeneratedReport result={{ analysis: A, explanation: active.explanation }} />
          </div>
        )}
      </div>
    </section>
  );
}

function QuantCard({ color, label, value }) {
  return (
    <div className={`bg-gradient-to-br from-${color}-50 to-${color}-100 p-4 rounded-xl border border-${color}-200`}>
      <div className="flex items-center gap-2 mb-2">
        <div className={`color-dot bg-${color}-500`}></div>
        <p className={`text-sm font-medium text-${color}-800`}>{label}</p>
      </div>
      <p className={`text-2xl font-bold text-${color}-900`}>{value}</p>
    </div>
  );
}
