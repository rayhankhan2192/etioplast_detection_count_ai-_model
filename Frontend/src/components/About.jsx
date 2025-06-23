export default function About() {
  return (
    <section id="about" className="mt-24 md:mt-32 text-center bg-grid-pattern">
      <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4">
        How It Works
      </h2>
      <p className="max-w-2xl mx-auto text-lg text-slate-600 mb-12">
        Advanced machine learning algorithms analyze electron microscopy images
        to identify and quantify etioplast structures with high precision.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl mx-auto px-4">
        {[
          {
            icon: "M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12",
            title: "Upload Image",
            desc: "Upload your electron microscopy image in standard formats (JPG, PNG, TIFF)",
          },
          {
            icon: "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z",
            title: "AI Analysis",
            desc: "Advanced neural networks identify and segment etioplast structures with high accuracy",
          },
          {
            icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z",
            title: "Get Results",
            desc: "Receive detailed quantification of all detected structures with precise measurements",
          },
        ].map((step, i) => (
          <div
            key={i}
            className="bg-white/90 backdrop-blur-md border border-slate-200 rounded-2xl shadow-lg p-8 hover:shadow-2xl transition-shadow duration-300"
          >
            <div className="bg-emerald-100 text-emerald-600 rounded-full h-16 w-16 flex items-center justify-center mx-auto mb-6">
              <svg
                className="h-8 w-8"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d={step.icon}
                />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-slate-900 mb-2">
              {step.title}
            </h3>
            <p className="text-slate-600 text-sm">{step.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
