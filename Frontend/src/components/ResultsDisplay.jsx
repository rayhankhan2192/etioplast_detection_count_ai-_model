export default function ResultsDisplay({ result }) {
  return (
    <section id="results" className="py-12 pb-2">
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-md border border-slate-200/50 max-w-4xl mx-auto">
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
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>
          Detection Results
        </h2>

        <div className="min-h-[20rem] flex flex-col justify-center items-center">
          {!result?.output_image_url && (
            <div className="text-center space-y-4 text-slate-500 p-8">
              <div className="bg-slate-100 p-6 rounded-full mx-auto w-fit">
                <svg
                  className="h-16 w-16 text-slate-400"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="1"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Z"
                  />
                </svg>
              </div>
              <p className="font-semibold text-lg text-slate-700">
                Image analysis will appear here
              </p>
              <p className="text-sm text-slate-500">
                Upload a microscopy image and run detection to see annotated
                results
              </p>
            </div>
          )}

          {result?.output_image_url && (
            <img src={result?.output_image_url} alt="" srcset="" />
          )}
        </div>
      </div>
    </section>
  );
}
