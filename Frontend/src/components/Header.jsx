const Header = () => {
  return (
    <header className="glass-effect sticky top-0 z-50 border-b border-slate-200/50">
      <nav className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-br from-emerald-500 to-teal-600 p-2 rounded-lg shadow-lg shadow-emerald-500/20">
              <svg
                className="h-6 w-6 text-white"
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
            <span className="text-xl font-bold text-slate-800 tracking-tight">
              Etioplast Detector AI
            </span>
          </div>
          <div className="hidden md:flex items-center space-x-8">
            <a
              href="#app"
              className="text-slate-600 hover:text-emerald-600 font-medium transition-colors"
            >
              Home
            </a>
            <a
              href="#about"
              className="text-slate-600 hover:text-emerald-600 font-medium transition-colors"
            >
              About
            </a>
            <a
              href="#upload"
              className="text-slate-600 hover:text-emerald-600 font-medium transition-colors"
            >
              Upload
            </a>
            <a
              href="#results"
              className="text-slate-600 hover:text-emerald-600 font-medium transition-colors"
            >
              Results
            </a>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;
