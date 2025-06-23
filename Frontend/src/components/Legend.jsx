export function Legend() {
  const items = [
    { label: "Etioplast", color: "bg-yellow-400" },
    { label: "PLB", color: "bg-purple-400" },
    { label: "Prothylakoid", color: "bg-red-400" },
    { label: "Plastoglobule", color: "bg-blue-400" },
  ];

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/50 p-4 shadow-md">
      <h3 className="text-slate-800 font-bold mb-4">Structure Color Legend</h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        {items.map((item, i) => (
          <div key={i} className="flex items-center space-x-2">
            <span
              className={`inline-block w-3 h-3 rounded-full ${item.color}`}
            ></span>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
