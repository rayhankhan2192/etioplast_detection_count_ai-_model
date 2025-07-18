import { useState } from "react";

export default function UploadSection({ handleDetection, setSelectedFile }) {
  const [previewURL, setPreviewURL] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreviewURL(null);
  };

  return (
    <section id="upload" className="py-12 pb-2">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-md border border-slate-200/50 max-w-2xl mx-auto">
        <h2 className="text-2xl font-bold text-slate-900 mb-6 flex items-center gap-3">
          <div className="bg-emerald-100 p-2 rounded-lg">
            <svg
              className="h-5 w-5 text-emerald-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>
          Upload Image
        </h2>

        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="relative w-full h-80 bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl border-2 border-dashed border-slate-300 flex flex-col justify-center items-center text-center p-6 hover:border-emerald-400 hover:bg-emerald-50/50 cursor-pointer shadow-inner transition-all duration-300"
        >
          {!previewURL ? (
            <>
              <label htmlFor="fileInput" className="cursor-pointer space-y-4">
                <div className="bg-emerald-100 p-4 rounded-full mx-auto w-fit">
                  <svg
                    className="h-12 w-12 text-emerald-600"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth="1.5"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Z"
                    />
                  </svg>
                </div>
                <p className="text-slate-700 font-semibold text-lg">
                  Drop your microscopy image here
                </p>
                <p className="text-slate-500 text-sm">
                  or{" "}
                  <span className="text-emerald-600 font-medium">
                    browse files
                  </span>
                </p>
                <p className="text-xs text-slate-400">
                  Supports: JPG, PNG, TIFF, WEBP (Max: 10MB)
                </p>
              </label>
              <input
                type="file"
                accept="image/*"
                id="fileInput"
                className="hidden"
                onChange={handleFileChange}
              />
            </>
          ) : (
            <div className="h-full w-full flex items-center justify-center">
              <img
                src={previewURL}
                alt="Uploaded Preview"
                className="max-h-full max-w-full rounded-xl object-contain"
              />
            </div>
          )}
        </div>

        {/* Buttons Row */}
        <div className="mt-6 flex items-center justify-between gap-4">
          <button
            onClick={handleDetection}
            className="flex-1 bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-bold py-3 px-6 rounded-xl shadow-lg hover:from-emerald-700 hover:to-teal-700 transition-all duration-200"
          >
            Run Detection
          </button>

          {previewURL && (
            <button
              onClick={handleRemoveImage}
              className="bg-red-500 hover:bg-red-600 text-white rounded-xl w-12 h-12 flex items-center justify-center shadow-md"
              title="Remove Image"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
