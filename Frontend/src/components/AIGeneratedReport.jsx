import React from "react";

export default function AIGeneratedReport({ result }) {
  // Hide the component entirely if there is no explanation
  if (!result?.explanation) return null;

  // Helper function to parse AI text into structured HTML
  const formatReport = (text) => {
    // 1. Force a line break before numbered items (e.g., "1. **Etioplast")
    let formatted = text.replace(/(?:\s|^)(\d+\.\s\*\*)/g, '\n\n$1');
    
    // 2. Force a line break before the summary/conclusion
    formatted = formatted.replace(/(Overall,\s|In summary,\s|In conclusion,\s)/g, '\n\n$1');

    // 3. Split the text by the newlines we just created
    const paragraphs = formatted.split('\n').filter((p) => p.trim() !== '');

    return paragraphs.map((para, index) => {
      // 4. Extract and format **bold** text
      const parts = para.split(/(\*\*.*?\*\*)/g);
      
      return (
        <p key={index} className="mb-4 text-slate-600 leading-relaxed last:mb-0">
          {parts.map((part, i) => {
            if (part.startsWith('**') && part.endsWith('**')) {
              // Remove the asterisks and wrap in a styled <strong> tag
              return (
                <strong key={i} className="font-semibold text-slate-900">
                  {part.slice(2, -2)}
                </strong>
              );
            }
            return part; // Return normal text
          })}
        </p>
      );
    });
  };

  return (
    <div className="max-w-4xl mx-auto mt-8">
      {/* Header with an AI Sparkle/Lightning Icon for professional flair */}
      <div className="flex items-center gap-2 mb-3 px-1">
        <svg 
          className="w-5 h-5 text-indigo-500" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth="2" 
            d="M13 10V3L4 14h7v7l9-11h-7z" 
          />
        </svg>
        <h2 className="text-lg font-bold text-slate-900">
          AI Analysis Report
        </h2>
      </div>

      {/* Report Container */}
      <div className="bg-slate-50/50 border border-slate-200/80 p-6 rounded-xl shadow-sm">
        {formatReport(result.explanation)}
      </div>
    </div>
  );
}