// // import { useEffect, useMemo, useState } from "react";

// // export default function UploadSection({ handleDetection, setSelectedFiles, setSelectedFile }) {
// //   const [files, setFiles] = useState([]); // File[]
// //   const [errors, setErrors] = useState([]); // string[]

// //   const previews = useMemo(
// //     () => files.map((f) => ({ url: URL.createObjectURL(f), name: f.name, size: f.size })),
// //     [files]
// //   );

// //   useEffect(() => () => previews.forEach((p) => URL.revokeObjectURL(p.url)), [previews]);

// //   // Propagate up
// //   useEffect(() => {
// //     setSelectedFiles?.(files);
// //     setSelectedFile?.(files[0] ?? null);
// //   }, [files, setSelectedFiles, setSelectedFile]);

// //   const MAX_FILES = 5;
// //   const MAX_SIZE = 20 * 1024 * 1024; // 20MB
// //   const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff"];

// //   function pickValid(filesList) {
// //     const newErrors = [];
// //     let picked = [];
// //     const existing = new Set(files.map((f) => f.name + f.size));

// //     for (const f of filesList) {
// //       if (!ACCEPTED_TYPES.includes(f.type)) {
// //         newErrors.push(`Skipped ${f.name}: unsupported type (${f.type || "unknown"}).`);
// //         continue;
// //       }
// //       if (f.size > MAX_SIZE) {
// //         newErrors.push(`Skipped ${f.name}: larger than 20MB.`);
// //         continue;
// //       }
// //       const key = f.name + f.size;
// //       if (existing.has(key)) {
// //         newErrors.push(`Skipped ${f.name}: duplicate.`);
// //         continue;
// //       }
// //       picked.push(f);
// //     }

// //     const slots = Math.max(0, MAX_FILES - files.length);
// //     if (picked.length > slots) {
// //       newErrors.push(`Only ${slots} more file${slots === 1 ? "" : "s"} allowed (max ${MAX_FILES}).`);
// //       picked = picked.slice(0, slots);
// //     }

// //     if (newErrors.length) setErrors(newErrors);
// //     if (picked.length) setFiles((prev) => [...prev, ...picked]);
// //   }

// //   const onSingleChange = (e) => {
// //     const f = e.target.files?.[0];
// //     if (!f) return;
// //     pickValid([f]);
// //     e.target.value = "";
// //   };

// //   const onFolderChange = (e) => {
// //     const list = Array.from(e.target.files || []);
// //     if (!list.length) return;
// //     pickValid(list);
// //     e.target.value = "";
// //   };

// //   const onDrop = (e) => {
// //     e.preventDefault();
// //     const list = Array.from(e.dataTransfer.files || []);
// //     if (!list.length) return;
// //     pickValid(list);
// //   };

// //   const onDragOver = (e) => e.preventDefault();

// //   const removeAt = (idx) => setFiles((prev) => prev.filter((_, i) => i !== idx));
// //   const clearAll = () => { setFiles([]); setErrors([]); };

// //   const usedPct = Math.min(100, Math.round((files.length / MAX_FILES) * 100));
// //   const hasFiles = files.length > 0;

// //   return (
// //     <section id="upload" className="py-12 pb-2">
// //       {/* Card fits column; clean padding & shadow */}
// //       <div className="w-full overflow-hidden bg-white/90 backdrop-blur rounded-2xl border border-slate-200 shadow-md">
// //         {/* Header */}
// //         <div className="px-6 pt-5 pb-3 flex items-center gap-3">
// //           <div className="bg-emerald-100 p-2 rounded-lg">
// //             <svg className="h-5 w-5 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
// //               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
// //             </svg>
// //           </div>
// //           <div className="flex-1 min-w-0">
// //             <h2 className="text-xl font-bold text-slate-900 truncate">Upload Image(s)</h2>
// //             <p className="text-xs text-slate-500">
// //               JPG, PNG, TIFF, WEBP • ≤ 20MB each
// //             </p>
// //           </div>

// //           {/* Count badge */}
// //           <div className="shrink-0 rounded-full px-3 py-1 text-xs font-semibold bg-slate-100 text-slate-700">
// //             {files.length} / {MAX_FILES}
// //           </div>
// //         </div>

// //         {/* Capacity bar */}
// //         <div className="px-6">
// //           <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
// //             <div
// //               className={`h-full ${usedPct >= 100 ? "bg-rose-500" : "bg-emerald-500"}`}
// //               style={{ width: `${usedPct}%` }}
// //             />
// //           </div>
// //         </div>

// //         {/* Dropzone (fixed height) */}
// //         <div
// //           onDrop={onDrop}
// //           onDragOver={onDragOver}
// //           className="m-6 mt-4 rounded-xl border-2 border-dashed border-slate-300 hover:border-emerald-400 transition-colors bg-gradient-to-b from-slate-50 to-slate-100"
// //           style={{ height: 220 }}
// //           aria-label="Drop images here"
// //         >
// //           {!hasFiles ? (
// //             <label
// //               htmlFor="fileInputSingle"
// //               className="w-full h-full flex flex-col items-center justify-center cursor-pointer"
// //             >
// //               <div className="bg-emerald-100 p-4 rounded-full">
// //                 <svg className="h-10 w-10 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
// //                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3 19.5h18"/>
// //                 </svg>
// //               </div>
// //               <p className="mt-3 text-slate-700 font-semibold">Drop microscopy images here</p>
// //               <p className="text-slate-500 text-sm">
// //                 or <span className="text-emerald-600 font-medium underline">browse files</span>
// //               </p>
// //               <input id="fileInputSingle" type="file" accept="image/*" className="hidden" onChange={onSingleChange} />
// //             </label>
// //           ) : (
// //             // Horizontal strip, no vertical growth
// //             <div className="h-full w-full flex items-center px-3">
// //               <div className="flex gap-3 overflow-x-auto w-full">
// //                 {previews.map((p, i) => (
// //                   <figure
// //                     key={p.url}
// //                     className="relative shrink-0 w-28 h-28 rounded-xl overflow-hidden border border-slate-200 bg-white group"
// //                     title={p.name}
// //                   >
// //                     <img src={p.url} alt={p.name} className="w-full h-full object-cover" />
// //                     {/* filename */}
// //                     <figcaption className="absolute inset-x-0 bottom-0 px-1.5 py-1 bg-black/50 text-white text-[10px] truncate">
// //                       {p.name}
// //                     </figcaption>
// //                     {/* remove */}
// //                     <button
// //                       type="button"
// //                       onClick={() => removeAt(i)}
// //                       className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-rose-500 hover:bg-rose-600 text-white flex items-center justify-center shadow"
// //                       aria-label={`Remove ${p.name}`}
// //                     >
// //                       <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
// //                         <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
// //                       </svg>
// //                     </button>
// //                   </figure>
// //                 ))}
// //               </div>
// //             </div>
// //           )}
// //         </div>

// //         {/* Errors */}
// //         {errors.length > 0 && (
// //           <div className="px-6 -mt-2">
// //             <div className="rounded-lg border border-amber-300 bg-amber-50 text-amber-800 p-3 text-sm space-y-1">
// //               {errors.map((e, i) => (<div key={i}>{e}</div>))}
// //             </div>
// //           </div>
// //         )}

// //         {/* Controls */}
// //         <div className="px-6 pt-4 pb-6 flex flex-wrap items-center gap-2">
// //           <div className="flex gap-2">
// //             <label
// //               htmlFor="fileInputSingleBtn"
// //               className="inline-flex items-center gap-2 bg-slate-900 text-white font-medium py-2 px-4 rounded-lg shadow hover:bg-slate-950 cursor-pointer"
// //             >
// //               <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
// //                 <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V8m0 0l-3 3m3-3 3 3M4 16.5V18a2 2 0 002 2h12a2 2 0 002-2v-1.5"/>
// //               </svg>
// //               Add image
// //             </label>
// //             <input id="fileInputSingleBtn" type="file" accept="image/*" className="hidden" onChange={onSingleChange} />

// //             <label
// //               htmlFor="folderInput"
// //               className="inline-flex items-center gap-2 bg-emerald-600 text-white font-medium py-2 px-4 rounded-lg shadow hover:bg-emerald-700 cursor-pointer"
// //             >
// //               <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
// //                 <path strokeLinecap="round" strokeLinejoin="round" d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V7z"/>
// //               </svg>
// //               Add folder
// //             </label>
// //             <input
// //               id="folderInput"
// //               type="file"
// //               className="hidden"
// //               webkitdirectory=""
// //               mozdirectory=""
// //               directory=""
// //               onChange={onFolderChange}
// //               accept="image/*"
// //             />

// //             {hasFiles && (
// //               <button
// //                 type="button"
// //                 onClick={clearAll}
// //                 className="inline-flex items-center gap-2 bg-rose-500 hover:bg-rose-600 text-white font-medium py-2 px-4 rounded-lg shadow"
// //               >
// //                 <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
// //                   <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
// //                 </svg>
// //                 Clear all
// //               </button>
// //             )}
// //           </div>

// //           <button
// //             onClick={handleDetection}
// //             disabled={!hasFiles}
// //             className="ml-auto bg-gradient-to-r from-emerald-600 to-teal-600 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed text-white font-bold py-2.5 px-6 rounded-lg shadow hover:from-emerald-700 hover:to-teal-700 transition-colors"
// //           >
// //             Run Detection ({files.length || 0})
// //           </button>
// //         </div>
// //       </div>
// //     </section>
// //   );
// // }


// import { useEffect, useMemo, useState } from "react";

// export default function UploadSection({
//   handleDetection,
//   setSelectedFiles,
//   setSelectedFile,
//   setPxPerUm,        // NEW: tell parent the px_per_um value
//   setUploadMode,     // (optional) tell parent whether user intends "file" or "files"
// }) {
//   const [files, setFiles] = useState([]); // File[]
//   const [errors, setErrors] = useState([]); // string[]
//   const [mode, setMode] = useState("auto"); // 'auto' | 'file' | 'files'
//   const [pxPerUm, setPxPerUmLocal] = useState(""); // required

//   const previews = useMemo(
//     () => files.map((f) => ({ url: URL.createObjectURL(f), name: f.name, size: f.size })),
//     [files]
//   );

//   useEffect(() => () => previews.forEach((p) => URL.revokeObjectURL(p.url)), [previews]);

//   // Propagate up
//   useEffect(() => {
//     setSelectedFiles?.(files);
//     setSelectedFile?.(files[0] ?? null);
//   }, [files, setSelectedFiles, setSelectedFile]);

//   useEffect(() => {
//     setPxPerUm?.(pxPerUm);         // keep parent in sync
//   }, [pxPerUm, setPxPerUm]);

//   useEffect(() => {
//     setUploadMode?.(mode);         // optional: parent may want to know
//   }, [mode, setUploadMode]);

//   const MAX_FILES = 5;
//   const MAX_SIZE = 20 * 1024 * 1024; // 20MB
//   const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff"];

//   function pickValid(filesList) {
//     const newErrors = [];
//     let picked = [];
//     const existing = new Set(files.map((f) => f.name + f.size));

//     for (const f of filesList) {
//       if (!ACCEPTED_TYPES.includes(f.type)) {
//         newErrors.push(`Skipped ${f.name}: unsupported type (${f.type || "unknown"}).`);
//         continue;
//       }
//       if (f.size > MAX_SIZE) {
//         newErrors.push(`Skipped ${f.name}: larger than 20MB.`);
//         continue;
//       }
//       const key = f.name + f.size;
//       if (existing.has(key)) {
//         newErrors.push(`Skipped ${f.name}: duplicate.`);
//         continue;
//       }
//       picked.push(f);
//     }

//     const slots = Math.max(0, MAX_FILES - files.length);
//     if (picked.length > slots) {
//       newErrors.push(`Only ${slots} more file${slots === 1 ? "" : "s"} allowed (max ${MAX_FILES}).`);
//       picked = picked.slice(0, slots);
//     }

//     if (newErrors.length) setErrors(newErrors);
//     if (picked.length) setFiles((prev) => [...prev, ...picked]);
//   }

//   const onSingleChange = (e) => {
//     const f = e.target.files?.[0];
//     if (!f) return;
//     pickValid([f]);
//     setMode("file");
//     e.target.value = "";
//   };

//   const onFolderChange = (e) => {
//     const list = Array.from(e.target.files || []);
//     if (!list.length) return;
//     pickValid(list);
//     setMode("files");
//     e.target.value = "";
//   };

//   const onDrop = (e) => {
//     e.preventDefault();
//     const list = Array.from(e.dataTransfer.files || []);
//     if (!list.length) return;
//     pickValid(list);
//     setMode(list.length > 1 ? "files" : "file");
//   };

//   const onDragOver = (e) => e.preventDefault();

//   const removeAt = (idx) => setFiles((prev) => prev.filter((_, i) => i !== idx));
//   const clearAll = () => { setFiles([]); setErrors([]); setMode("auto"); };

//   const usedPct = Math.min(100, Math.round((files.length / MAX_FILES) * 100));
//   const hasFiles = files.length > 0;

//   // px_per_um validation
//   const pxError =
//     pxPerUm === ""
//       ? "Required"
//       : isNaN(Number(pxPerUm)) || Number(pxPerUm) <= 0
//       ? "Enter a positive number"
//       : null;

//   const canRun = hasFiles && !pxError;

//   return (
//     <section id="upload" className="py-12 pb-2">
//       <div className="w-full overflow-hidden bg-white/90 backdrop-blur rounded-2xl border border-slate-200 shadow-md">
//         {/* Header */}
//         <div className="px-6 pt-5 pb-3 flex items-center gap-3">
//           <div className="bg-emerald-100 p-2 rounded-lg">
//             <svg className="h-5 w-5 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
//               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
//             </svg>
//           </div>
//           <div className="flex-1 min-w-0">
//             <h2 className="text-xl font-bold text-slate-900 truncate">Upload Image(s)</h2>
//             <p className="text-xs text-slate-500">
//               JPG, PNG, TIFF, WEBP • ≤ 20MB each
//             </p>
//           </div>

//           {/* Count badge */}
//           <div className="shrink-0 rounded-full px-3 py-1 text-xs font-semibold bg-slate-100 text-slate-700">
//             {files.length} / {MAX_FILES}
//           </div>
//         </div>

//         {/* Mode + scale row */}
//         <div className="px-6 pb-2 flex flex-wrap items-center gap-3">
//           {/* Mode switch */}
//           <div className="inline-flex rounded-lg border border-slate-200 overflow-hidden">
//             <button
//               type="button"
//               onClick={() => setMode("file")}
//               className={`px-3 py-1.5 text-sm ${mode === "file" ? "bg-slate-900 text-white" : "bg-white text-slate-700"}`}
//               title="Send as single file"
//             >
//               File
//             </button>
//             <button
//               type="button"
//               onClick={() => setMode("files")}
//               className={`px-3 py-1.5 text-sm border-l border-slate-200 ${mode === "files" ? "bg-slate-900 text-white" : "bg-white text-slate-700"}`}
//               title="Send as multiple files (folder)"
//             >
//               Files
//             </button>
//             <button
//               type="button"
//               onClick={() => setMode("auto")}
//               className={`px-3 py-1.5 text-sm border-l border-slate-200 ${mode === "auto" ? "bg-slate-900 text-white" : "bg-white text-slate-700"}`}
//               title="Decide automatically based on count"
//             >
//               Auto
//             </button>
//           </div>

//           {/* px_per_um input (required) */}
//           <div className="flex items-center gap-2">
//             <label htmlFor="pxPerUm" className="text-sm font-medium text-slate-700">
//               Pixels/µm
//             </label>
//             <input
//               id="pxPerUm"
//               type="number"
//               min="0"
//               step="any"
//               placeholder="e.g. 12.5"
//               value={pxPerUm}
//               onChange={(e) => setPxPerUmLocal(e.target.value)}
//               className={`w-28 rounded-md border px-3 py-1.5 text-sm outline-none ${
//                 pxError ? "border-rose-300 focus:border-rose-400" : "border-slate-300 focus:border-emerald-400"
//               }`}
//             />
//             {pxError && <span className="text-xs text-rose-500">{pxError}</span>}
//           </div>
//         </div>

//         {/* Capacity bar */}
//         <div className="px-6">
//           <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
//             <div
//               className={`h-full ${usedPct >= 100 ? "bg-rose-500" : "bg-emerald-500"}`}
//               style={{ width: `${usedPct}%` }}
//             />
//           </div>
//         </div>

//         {/* Dropzone (fixed height) */}
//         <div
//           onDrop={onDrop}
//           onDragOver={onDragOver}
//           className="m-6 mt-4 rounded-xl border-2 border-dashed border-slate-300 hover:border-emerald-400 transition-colors bg-gradient-to-b from-slate-50 to-slate-100"
//           style={{ height: 220 }}
//           aria-label="Drop images here"
//         >
//           {files.length === 0 ? (
//             <label htmlFor="fileInputSingle" className="w-full h-full flex flex-col items-center justify-center cursor-pointer">
//               <div className="bg-emerald-100 p-4 rounded-full">
//                 <svg className="h-10 w-10 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3 19.5h18"/>
//                 </svg>
//               </div>
//               <p className="mt-3 text-slate-700 font-semibold">Drop microscopy images here</p>
//               <p className="text-slate-500 text-sm">
//                 or <span className="text-emerald-600 font-medium underline">browse files</span>
//               </p>
//               <input id="fileInputSingle" type="file" accept="image/*" className="hidden" onChange={onSingleChange} />
//             </label>
//           ) : (
//             <div className="h-full w-full flex items-center px-3">
//               <div className="flex gap-3 overflow-x-auto w-full">
//                 {previews.map((p, i) => (
//                   <figure key={p.url} className="relative shrink-0 w-28 h-28 rounded-xl overflow-hidden border border-slate-200 bg-white group" title={p.name}>
//                     <img src={p.url} alt={p.name} className="w-full h-full object-cover" />
//                     <figcaption className="absolute inset-x-0 bottom-0 px-1.5 py-1 bg-black/50 text-white text-[10px] truncate">{p.name}</figcaption>
//                     <button
//                       type="button"
//                       onClick={() => removeAt(i)}
//                       className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-rose-500 hover:bg-rose-600 text-white flex items-center justify-center shadow"
//                       aria-label={`Remove ${p.name}`}
//                     >
//                       <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
//                         <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
//                       </svg>
//                     </button>
//                   </figure>
//                 ))}
//               </div>
//             </div>
//           )}
//         </div>

//         {/* Errors */}
//         {errors.length > 0 && (
//           <div className="px-6 -mt-2">
//             <div className="rounded-lg border border-amber-300 bg-amber-50 text-amber-800 p-3 text-sm space-y-1">
//               {errors.map((e, i) => (<div key={i}>{e}</div>))}
//             </div>
//           </div>
//         )}

//         {/* Controls */}
//         <div className="px-6 pt-4 pb-6 flex flex-wrap items-center gap-2">
//           <div className="flex gap-2">
//             <label htmlFor="fileInputSingleBtn" className="inline-flex items-center gap-2 bg-slate-900 text-white font-medium py-2 px-4 rounded-lg shadow hover:bg-slate-950 cursor-pointer">
//               <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
//                 <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V8m0 0l-3 3m3-3 3 3M4 16.5V18a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1.5"/>
//               </svg>
//               Add image
//             </label>
//             <input id="fileInputSingleBtn" type="file" accept="image/*" className="hidden" onChange={onSingleChange} />

//             <label htmlFor="folderInput" className="inline-flex items-center gap-2 bg-emerald-600 text-white font-medium py-2 px-4 rounded-lg shadow hover:bg-emerald-700 cursor-pointer">
//               <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
//                 <path strokeLinecap="round" strokeLinejoin="round" d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/>
//               </svg>
//               Add folder
//             </label>
//             <input id="folderInput" type="file" className="hidden" webkitdirectory="" mozdirectory="" directory="" onChange={onFolderChange} accept="image/*" />

//             {hasFiles && (
//               <button type="button" onClick={clearAll} className="inline-flex items-center gap-2 bg-rose-500 hover:bg-rose-600 text-white font-medium py-2 px-4 rounded-lg shadow">
//                 <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
//                   <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
//                 </svg>
//                 Clear all
//               </button>
//             )}
//           </div>

//           <button
//             onClick={handleDetection}
//             disabled={!canRun}
//             className="ml-auto bg-gradient-to-r from-emerald-600 to-teal-600 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed text-white font-bold py-2.5 px-6 rounded-lg shadow hover:from-emerald-700 hover:to-teal-700 transition-colors"
//             title={!hasFiles ? "Add at least one image" : pxError ? "Enter pixels per µm" : "Run detection"}
//           >
//             Run Detection ({files.length || 0})
//           </button>
//         </div>
//       </div>
//     </section>
//   );
// }

import { useEffect, useMemo, useState } from "react";

export default function UploadSection({
  handleDetection,
  setSelectedFiles,
  setSelectedFile,
  setPxPerUm, // NEW: pass px_per_um up to App
}) {
  const [files, setFiles] = useState([]); // File[]
  const [errors, setErrors] = useState([]); // string[]

  const previews = useMemo(
    () => files.map((f) => ({ url: URL.createObjectURL(f), name: f.name, size: f.size })),
    [files]
  );

  useEffect(() => () => previews.forEach((p) => URL.revokeObjectURL(p.url)), [previews]);

  // Propagate up
  useEffect(() => {
    setSelectedFiles?.(files);
    setSelectedFile?.(files[0] ?? null);
  }, [files, setSelectedFiles, setSelectedFile]);

  const MAX_FILES = 5;
  const MAX_SIZE = 20 * 1024 * 1024; // 20MB
  const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff"];

  function pickValid(filesList) {
    const newErrors = [];
    let picked = [];
    const existing = new Set(files.map((f) => f.name + f.size));

    for (const f of filesList) {
      if (!ACCEPTED_TYPES.includes(f.type)) {
        newErrors.push(`Skipped ${f.name}: unsupported type (${f.type || "unknown"}).`);
        continue;
      }
      if (f.size > MAX_SIZE) {
        newErrors.push(`Skipped ${f.name}: larger than 20MB.`);
        continue;
      }
      const key = f.name + f.size;
      if (existing.has(key)) {
        newErrors.push(`Skipped ${f.name}: duplicate.`);
        continue;
      }
      picked.push(f);
    }

    const slots = Math.max(0, MAX_FILES - files.length);
    if (picked.length > slots) {
      newErrors.push(`Only ${slots} more file${slots === 1 ? "" : "s"} allowed (max ${MAX_FILES}).`);
      picked = picked.slice(0, slots);
    }

    if (newErrors.length) setErrors(newErrors);
    if (picked.length) setFiles((prev) => [...prev, ...picked]);
  }

  const onSingleChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    pickValid([f]);
    e.target.value = "";
  };

  const onFolderChange = (e) => {
    const list = Array.from(e.target.files || []);
    if (!list.length) return;
    pickValid(list);
    e.target.value = "";
  };

  const onDrop = (e) => {
    e.preventDefault();
    const list = Array.from(e.dataTransfer.files || []);
    if (!list.length) return;
    pickValid(list);
  };

  const onDragOver = (e) => e.preventDefault();

  const removeAt = (idx) => setFiles((prev) => prev.filter((_, i) => i !== idx));
  const clearAll = () => {
    setFiles([]);
    setErrors([]);
  };

  const usedPct = Math.min(100, Math.round((files.length / MAX_FILES) * 100));
  const hasFiles = files.length > 0;

  return (
    <section id="upload" className="py-12 pb-2">
      {/* Card fits column; clean padding & shadow */}
      <div className="w-full overflow-hidden bg-white/90 backdrop-blur rounded-2xl border border-slate-200 shadow-md">
        {/* Header */}
        <div className="px-6 pt-5 pb-3 flex items-center gap-3">
          <div className="bg-emerald-100 p-2 rounded-lg">
            <svg className="h-5 w-5 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-xl font-bold text-slate-900 truncate">Upload Image(s)</h2>
            <p className="text-xs text-slate-500">JPG, PNG, TIFF, WEBP • ≤ 20MB each</p>
          </div>

          {/* Count badge */}
          <div className="shrink-0 rounded-full px-3 py-1 text-xs font-semibold bg-slate-100 text-slate-700">
            {files.length} / {MAX_FILES}
          </div>
        </div>

        {/* Capacity bar */}
        <div className="px-6">
          <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-full ${usedPct >= 100 ? "bg-rose-500" : "bg-emerald-500"}`}
              style={{ width: `${usedPct}%` }}
            />
          </div>
        </div>

        {/* Dropzone (fixed height) */}
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          className="m-6 mt-4 rounded-xl border-2 border-dashed border-slate-300 hover:border-emerald-400 transition-colors bg-gradient-to-b from-slate-50 to-slate-100"
          style={{ height: 220 }}
          aria-label="Drop images here"
        >
          {!hasFiles ? (
            <label
              htmlFor="fileInputSingle"
              className="w-full h-full flex flex-col items-center justify-center cursor-pointer"
            >
              <div className="bg-emerald-100 p-4 rounded-full">
                <svg className="h-10 w-10 text-emerald-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3 19.5h18"/>
                </svg>
              </div>
              <p className="mt-3 text-slate-700 font-semibold">Drop microscopy images here</p>
              <p className="text-slate-500 text-sm">
                or <span className="text-emerald-600 font-medium underline">browse files</span>
              </p>
              <input id="fileInputSingle" type="file" accept="image/*" className="hidden" onChange={onSingleChange} />
            </label>
          ) : (
            // Horizontal strip, no vertical growth
            <div className="h-full w-full flex items-center px-3">
              <div className="flex gap-3 overflow-x-auto w-full">
                {previews.map((p, i) => (
                  <figure
                    key={p.url}
                    className="relative shrink-0 w-28 h-28 rounded-xl overflow-hidden border border-slate-200 bg-white group"
                    title={p.name}
                  >
                    <img src={p.url} alt={p.name} className="w-full h-full object-cover" />
                    {/* filename */}
                    <figcaption className="absolute inset-x-0 bottom-0 px-1.5 py-1 bg-black/50 text-white text-[10px] truncate">
                      {p.name}
                    </figcaption>
                    {/* remove */}
                    <button
                      type="button"
                      onClick={() => removeAt(i)}
                      className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-rose-500 hover:bg-rose-600 text-white flex items-center justify-center shadow"
                      aria-label={`Remove ${p.name}`}
                    >
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
                      </svg>
                    </button>
                  </figure>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Errors */}
        {errors.length > 0 && (
          <div className="px-6 -mt-2">
            <div className="rounded-lg border border-amber-300 bg-amber-50 text-amber-800 p-3 text-sm space-y-1">
              {errors.map((e, i) => (<div key={i}>{e}</div>))}
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="px-6 pt-4 pb-6 flex flex-wrap items-center gap-2">
          {/* NEW: compact px_per_um input (doesn't alter layout)
          <div className="flex items-center gap-2">
            <label htmlFor="pxPerUm" className="text-sm text-slate-700">px_per_um [px/µm]</label>
            <input
              id="pxPerUm"
              type="number"
              min="0"
              step="any"
              placeholder="e.g. 85"
              onChange={(e) => setPxPerUm?.(e.target.value)}
              className="w-28 px-3 py-2 border border-slate-300 rounded-lg text-sm focus:border-emerald-500 outline-none"
              required
            />
          </div> */}
          {/* NEW: compact px_per_um input (doesn't alter layout) */}
          <div className="px-6 pt-2">
            <div className="flex items-center gap-3 bg-white/70 backdrop-blur-sm border border-slate-200 rounded-xl px-4 py-3 shadow-sm">
              <label
                htmlFor="pxPerUm"
                className="whitespace-nowrap text-sm font-semibold text-slate-700"
              >
                px_per_um [px/µm]
              </label>
              <input
                id="pxPerUm"
                type="number"
                min="0"
                step="any"
                placeholder="e.g. 85"
                onChange={(e) => setPxPerUm?.(e.target.value)}
                className="w-28 px-3 py-2 border border-slate-300 rounded-lg text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none text-slate-800 placeholder-slate-400"
                required
              />
            </div>
            <p className="mt-1 text-xs text-slate-500 px-1">
              Calibration factor (pixels per micron).
            </p>
          </div>


          <div className="flex gap-2">
            <label
              htmlFor="fileInputSingleBtn"
              className="inline-flex items-center gap-2 bg-slate-900 text-white font-medium py-2 px-4 rounded-lg shadow hover:bg-slate-950 cursor-pointer"
            >
              <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V8m0 0l-3 3m3-3 3 3M4 16.5V18a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-1.5"/>
              </svg>
              Add image
            </label>
            <input id="fileInputSingleBtn" type="file" accept="image/*" className="hidden" onChange={onSingleChange} />

            <label
              htmlFor="folderInput"
              className="inline-flex items-center gap-2 bg-emerald-600 text-white font-medium py-2 px-4 rounded-lg shadow hover:bg-emerald-700 cursor-pointer"
            >
              <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/>
              </svg>
              Add folder
            </label>
            <input
              id="folderInput"
              type="file"
              className="hidden"
              webkitdirectory=""
              mozdirectory=""
              directory=""
              onChange={onFolderChange}
              accept="image/*"
            />

            {hasFiles && (
              <button
                type="button"
                onClick={clearAll}
                className="inline-flex items-center gap-2 bg-rose-500 hover:bg-rose-600 text-white font-medium py-2 px-4 rounded-lg shadow"
              >
                <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
                </svg>
                Clear all
              </button>
            )}
          </div>

          <button
            onClick={handleDetection}
            disabled={!hasFiles}
            className="ml-auto bg-gradient-to-r from-emerald-600 to-teal-600 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed text-white font-bold py-2.5 px-6 rounded-lg shadow hover:from-emerald-700 hover:to-teal-700 transition-colors"
          >
            Run Detection ({files.length || 0})
          </button>
        </div>
      </div>
    </section>
  );
}
