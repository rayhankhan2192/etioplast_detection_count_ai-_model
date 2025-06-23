const Footer = () => {
  return (
    <footer className="bg-white/60 backdrop-blur-sm border-t border-slate-200/50 py-8 mt-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-br from-emerald-500 to-teal-600 p-2 rounded-lg shadow-lg shadow-emerald-500/20">
              <svg
                className="h-5 w-5 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <span className="font-bold text-slate-800">
              Etioplast Detector AI
            </span>
          </div>
          <p className="text-slate-600 text-sm">
            © 2025 Advanced Microscopy Analysis. Powered by Data Science Lab
            DIU.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
