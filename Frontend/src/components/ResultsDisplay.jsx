// ResultsDisplay.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import AIGeneratedReport from "./AIGeneratedReport";

export default function ResultsDisplay({ result, activeIndex, setActiveIndex }) {
  // Build an array of image URLs for display
  const { images, isFolderMode, activeResult } = useMemo(() => {
    const folder = Array.isArray(result?.results) && result.results.length > 0;
    if (folder) {
      const imgs = result.results
        .map((r) => r?.output_urls?.contours || r?.output_image_url || r?.output_urls?.blended)
        .filter(Boolean);
      const current = result.results[Math.min(activeIndex, Math.max(imgs.length - 1, 0))] || null;
      return { images: imgs, isFolderMode: true, activeResult: current };
    }
    const single =
      result?.output_urls?.contours ||
      result?.output_image_url ||
      result?.output_urls?.blended;
    return {
      images: single ? [single] : [],
      isFolderMode: false,
      activeResult: single ? result : null,
    };
  }, [result, activeIndex]);

  const hasImages = images.length > 0;

  // slideshow autoplay is local UI state
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);

  // Stop autoplay if images set changes
  useEffect(() => {
    setPlaying(false);
    clearInterval(timer.current);
  }, [images]);

  // Autoplay
  useEffect(() => {
    clearInterval(timer.current);
    if (playing && images.length > 1) {
      timer.current = setInterval(() => {
        setActiveIndex((i) => (i + 1) % images.length);
      }, 2000);
    }
    return () => clearInterval(timer.current);
  }, [playing, images.length, setActiveIndex]);

  const prev = () => setActiveIndex((i) => (i - 1 + images.length) % images.length);
  const next = () => setActiveIndex((i) => (i + 1) % images.length);
  const togglePlay = () => setPlaying((p) => !p);

  return (
    <section id="results" className="py-12 pb-2">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-md border border-slate-200/50 max-w-4xl mx-auto">
        <h2 className="text-2xl font-bold text-slate-900 mb-6 flex items-center gap-3">
          <div className="bg-emerald-100 p-2 rounded-lg">
            <svg className="h-5 w-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2 2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          Detection Results
          {isFolderMode && hasImages && (
            <span className="ml-auto text-xs font-medium text-slate-500">
              {activeIndex + 1} / {images.length}
            </span>
          )}
        </h2>

        <div className="min-h-[20rem] flex flex-col items-center">
          {!hasImages && (
            <div className="text-center space-y-4 text-slate-500 p-8">
              <div className="bg-slate-100 p-6 rounded-full mx-auto w-fit">
                <svg className="h-16 w-16 text-slate-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Z" />
                </svg>
              </div>
              <p className="font-semibold text-lg text-slate-700">Image analysis will appear here</p>
              <p className="text-sm text-slate-500">Upload a microscopy image or a folder and run detection to see annotated results</p>
            </div>
          )}

          {hasImages && (
            <div className="w-full">
              <div className="w-full flex items-center justify-center">
                <img
                  src={images[activeIndex]}
                  alt={`Detection ${activeIndex + 1}`}
                  className="max-h-[28rem] w-auto rounded-xl object-contain border border-slate-200 bg-white shadow"
                  draggable={false}
                />
              </div>

              {isFolderMode && images.length > 1 && (
                <div className="mt-4 flex items-center justify-center gap-3">
                  <button onClick={prev} className="inline-flex items-center gap-2 bg-slate-800 text-white font-semibold py-2 px-4 rounded-xl shadow hover:bg-slate-900">◀ Prev</button>
                  <button onClick={togglePlay} className="inline-flex items-center gap-2 bg-emerald-600 text-white font-semibold py-2 px-4 rounded-xl shadow hover:bg-emerald-700">
                    {playing ? "Pause" : "Play"} Slideshow
                  </button>
                  <button onClick={next} className="inline-flex items-center gap-2 bg-slate-800 text-white font-semibold py-2 px-4 rounded-xl shadow hover:bg-slate-900">Next ▶</button>
                </div>
              )}

              {isFolderMode && images.length > 1 && (
                <div className="mt-4 grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                  {images.map((url, i) => (
                    <button
                      key={url + i}
                      onClick={() => setActiveIndex(i)}
                      className={`border rounded-lg overflow-hidden ${i === activeIndex ? "ring-2 ring-emerald-500" : "border-slate-200"}`}
                      title={`Go to ${i + 1}`}
                    >
                      <img src={url} alt={`thumb ${i + 1}`} className="aspect-square w-full object-cover" draggable={false} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 🔻 AI text for the current image */}
          {/* {activeResult && <AIGeneratedReport result={activeResult} />} */}
        </div>
      </div>
    </section>
  );
}
