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

  // 🔹 Shared index for slideshow + metrics + CSV + AI
  const [activeIndex, setActiveIndex] = useState(0);

  const handleDetection = async () => {
    const files = selectedFiles?.length ? selectedFiles : (selectedFile ? [selectedFile] : []);
    if (!files.length) {
      alert("Please upload at least one image first.");
      return;
    }

    setLoading(true);
    try {
      let response;
      if (files.length === 1) {
        const formData = new FormData();
        formData.append("file", files[0]);
        response = await fetch("http://127.0.0.1:8000/api/analyze-file/", { method: "POST", body: formData });
      } else {
        const formData = new FormData();
        files.forEach((f) => formData.append("files", f));
        response = await fetch("http://127.0.0.1:8000/api/analyze-folder/", { method: "POST", body: formData });
      }

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult(data);
      setActiveIndex(0); // reset to first image on new results
    } catch (e) {
      console.error("Detection failed:", e);
      alert("Detection failed. See console for details.");
    } finally {
      setLoading(false);
    }
  };

  // Keep activeIndex in-bounds if results length changes
  useEffect(() => {
    const len = Array.isArray(result?.results) ? result.results.length : (result?.output_image_url ? 1 : 0);
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
            />
            <Legend />
          </div>

          <div className="lg:col-span-2 space-y-2">
            {loading && (
              <div className="p-6 rounded-xl border bg-white shadow text-slate-600">Processing… please wait.</div>
            )}

            {/* 🔻 pass shared index both ways */}
            <ResultsDisplay
              result={result}
              activeIndex={activeIndex}
              setActiveIndex={setActiveIndex}
            />
            <QuantificationPanel
              result={result}
              activeIndex={activeIndex}
            />
          </div>
        </div>
      </main>
      <About />
      <Footer />
    </div>
  );
}

export default App;
