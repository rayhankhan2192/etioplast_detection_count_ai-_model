// App.jsx
import { useEffect, useState } from "react";
import "./App.css";
import About from "./components/About";
import Footer from "./components/Footer";
import Header from "./components/Header";
import Hero from "./components/Hero";
import { Legend } from "./components/Legend";
import QuantificationPanel from "./components/QuantificationPanel";
import ResultsDisplay from "./components/ResultsDisplay";
import UploadSection from "./components/UploadSection";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [result, setResult] = useState({});
  const [loading, setLoading] = useState(false);

  // Shared index for slideshow + metrics + CSV + AI
  const [activeIndex, setActiveIndex] = useState(0);

  // Pixels-per-micron, provided by UploadSection
  const [pxPerUm, setPxPerUm] = useState("");

  // Progress HUD state
  const [showHUD, setShowHUD] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [phase, setPhase] = useState("idle"); // 'idle' | 'upload' | 'process' | 'done'
  const [imagesTotal, setImagesTotal] = useState(0);
  const [imagesDone, setImagesDone] = useState(0);

  const handleDetection = async () => {
    const files =
      selectedFiles?.length ? selectedFiles : selectedFile ? [selectedFile] : [];
    if (!files.length) {
      alert("Please upload at least one image first.");
      return;
    }
    if (!pxPerUm) {
      alert("Please enter pixels-per-micron (px_per_um).");
      return;
    }

    const isFolder = files.length > 1;
    const url = isFolder
      ? "http://127.0.0.1:8000/api/analyze-folder/"
      : "http://127.0.0.1:8000/api/analyze-file/";

    const formData = new FormData();
    if (isFolder) {
      files.forEach((f) => formData.append("files", f));
    } else {
      formData.append("file", files[0]);
    }
    formData.append("px_per_um", pxPerUm);

    // HUD: start upload
    setImagesTotal(files.length);
    setImagesDone(0);
    setUploadPct(0);
    setPhase("upload");
    setShowHUD(true);
    setLoading(true);

    try {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url, true);

      xhr.upload.onprogress = (evt) => {
        if (evt.lengthComputable) {
          setUploadPct(Math.round((evt.loaded / evt.total) * 100));
        }
      };

      xhr.upload.onload = () => {
        setPhase("process");
        setUploadPct(100);
      };

      xhr.onreadystatechange = () => {
        if (xhr.readyState === 4) {
          if (xhr.status >= 200 && xhr.status < 300) {
            let data = {};
            try {
              data = JSON.parse(xhr.responseText || "{}");
            } catch (_) {}

            setResult(data);
            setActiveIndex(0);

            // Prefer server count if present; otherwise fallback to files.length
            const serverCount =
              (Array.isArray(data?.results) ? data.results.length : undefined) ??
              data?.processed ??
              data?.images_processed ??
              data?.num_processed ??
              data?.count;
            const finalDone = Number.isFinite(Number(serverCount))
              ? Number(serverCount)
              : files.length;

            // 🔒 Force a non-zero completed state on success
            setImagesDone(Math.max(finalDone, files.length));

            // Brief DONE phase so users see N/N before hiding
            setPhase("done");
            setUploadPct(100);
            setTimeout(() => {
              setShowHUD(false);
              setPhase("idle");
              setUploadPct(0);
            }, 700);
          } else {
            console.error("Detection failed:", xhr.status, xhr.responseText);
            alert("Detection failed. See console for details.");
            setShowHUD(false);
            setPhase("idle");
            setUploadPct(0);
          }
          setLoading(false);
        }
      };

      xhr.send(formData);
    } catch (e) {
      console.error("Detection failed:", e);
      alert("Detection failed. See console for details.");
      setLoading(false);
      setShowHUD(false);
      setPhase("idle");
      setUploadPct(0);
    }
  };

  // Keep activeIndex in-bounds if results length changes
  useEffect(() => {
    const len = Array.isArray(result?.results)
      ? result.results.length
      : result?.output_image_url
      ? 1
      : 0;
    if (activeIndex >= len) setActiveIndex(0);
  }, [result, activeIndex]);

  return (
    <div className="bg-slate-50 text-slate-800 min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow">
        <Hero />

        <div className="max-w-7xl mx-auto px-4 grid grid-cols-1 lg:grid-cols-3 gap-6 mt-12">
          <div className="lg:col-span-1 space-y-6">
            <UploadSection
              handleDetection={handleDetection}
              setSelectedFiles={setSelectedFiles}
              setSelectedFile={setSelectedFile}
              setPxPerUm={setPxPerUm}
            />
            <Legend />
          </div>

          <div className="lg:col-span-2 space-y-2">
            {showHUD && (
              <ProgressHUD
                phase={phase}
                uploadPct={uploadPct}
                total={imagesTotal}
                done={imagesDone}
              />
            )}

            <ResultsDisplay
              result={result}
              activeIndex={activeIndex}
              setActiveIndex={setActiveIndex}
            />
            <QuantificationPanel result={result} activeIndex={activeIndex} />
          </div>
        </div>
      </main>
      <About />
      <Footer />
    </div>
  );
}

function ProgressHUD({ phase, uploadPct, total, done }) {
  const isUploading = phase === "upload";
  const isProcessing = phase === "process";
  const isDone = phase === "done";

  return (
    <div className="p-5 rounded-xl border bg-white shadow flex flex-col gap-3">
      <div className="flex items-center gap-3">
        {isUploading && (
          <span className="inline-flex h-5 w-5 rounded-full bg-emerald-500" />
        )}
        {isProcessing && (
          <span className="inline-flex h-5 w-5 rounded-full border-2 border-slate-300 border-t-2 border-t-emerald-500 animate-spin" />
        )}
        {isDone && (
          <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-emerald-500 text-white text-xs">
            ✓
          </span>
        )}

        <p className="font-semibold text-slate-800">
          {isUploading
            ? "Uploading…"
            : isProcessing
            ? "Processing on server…"
            : "Completed"}
        </p>

        <span className="ml-auto text-sm text-slate-600">
          {isUploading
            ? `Images: ${total}`
            : `Processed: ${Math.min(done || 0, total)} / ${total}`}
        </span>
      </div>

      {/* Progress bar */}
      <div className="w-full h-3 rounded-full bg-slate-100 overflow-hidden">
        {isUploading ? (
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 transition-[width] duration-200"
            style={{ width: `${uploadPct}%` }}
          />
        ) : isProcessing ? (
          <div className="relative w-full h-full overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/30 via-emerald-500 to-emerald-500/30 animate-[slide_1.2s_linear_infinite]" />
          </div>
        ) : (
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-teal-500"
            style={{ width: "100%" }}
          />
        )}
      </div>

      <div className="flex justify-between text-xs text-slate-600">
        <span>
          {isUploading
            ? `${uploadPct}%`
            : isProcessing
            ? "Working… this may take a moment"
            : "Ready"}
        </span>
        <span>Images: {total}</span>
      </div>

      {/* Inline keyframes for the indeterminate shimmer */}
      <style>{`
        @keyframes slide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-[slide_1.2s_linear_infinite] {
          animation: slide 1.2s linear infinite;
        }
      `}</style>
    </div>
  );
}

export default App;
