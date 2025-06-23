const Hero = () => {
  return (
    <div className="text-center mb-16 animate-fade-in">
      <h1 className="text-4xl md:text-6xl font-extrabold text-slate-900 tracking-tighter mb-6">
        Upload Microscopy Image for{" "}
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-teal-600">
          Detection
        </span>
      </h1>
      <p className="mt-4 max-w-3xl mx-auto text-lg text-slate-600 leading-relaxed">
        Advanced AI-powered detection of etioplast structures in electron
        microscopy images. Get precise quantification of biological structures
        including PLB, prothylakoids, and plastoglobules.
      </p>
    </div>
  );
};

export default Hero;
