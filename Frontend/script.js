        // // DOM Elements
        // const imageUploadBox = document.getElementById('image-upload-box');
        // const fileInput = document.getElementById('file-input');
        // const imagePreview = document.getElementById('image-preview');
        // const uploadPrompt = document.getElementById('upload-prompt');
        // const submitButton = document.getElementById('submit-button');
        // const clearButton = document.getElementById('clear-button');
        // const submitIcon = document.getElementById('submit-icon');
        // const submitText = document.getElementById('submit-text');
        // const mobileMenuButton = document.getElementById('mobile-menu-button');
        // const mobileMenu = document.getElementById('mobile-menu');
        
        // // Result Elements
        // const placeholderImages = document.getElementById('placeholder-images');
        // const processingSpinner = document.getElementById('processing-spinner');
        // const imageDisplay = document.getElementById('image-display');
        // const resultImage = document.getElementById('result-image');
        // const originalTab = document.getElementById('original-tab');
        // const annotatedTab = document.getElementById('annotated-tab');
        
        // // Quantification Elements
        // const placeholderQuantification = document.getElementById('placeholder-quantification');
        // const quantificationResults = document.getElementById('quantification-results');
        
        // // State
        // let uploadedImage = null;
        // let originalImageData = null;
        // let annotatedImageData = null;
        // let currentView = 'original';
        
        // // Mobile Menu Toggle
        // mobileMenuButton.addEventListener('click', () => {
        //     mobileMenu.classList.toggle('hidden');
        // });
        
        // // File Upload Handlers
        // imageUploadBox.addEventListener('click', () => {
        //     if (!uploadedImage) {
        //         fileInput.click();
        //     }
        // });
        
        // imageUploadBox.addEventListener('dragover', (e) => {
        //     e.preventDefault();
        //     imageUploadBox.classList.add('border-emerald-400', 'bg-emerald-50/50');
        // });
        
        // imageUploadBox.addEventListener('dragleave', (e) => {
        //     e.preventDefault();
        //     imageUploadBox.classList.remove('border-emerald-400', 'bg-emerald-50/50');
        // });
        
        // imageUploadBox.addEventListener('drop', (e) => {
        //     e.preventDefault();
        //     imageUploadBox.classList.remove('border-emerald-400', 'bg-emerald-50/50');
        //     const files = e.dataTransfer.files;
        //     if (files.length > 0) {
        //         handleFileUpload(files[0]);
        //     }
        // });
        
        // fileInput.addEventListener('change', (e) => {
        //     if (e.target.files.length > 0) {
        //         handleFileUpload(e.target.files[0]);
        //     }
        // });
        
        // // Clear Image Handler
        // clearButton.addEventListener('click', () => {
        //     clearUploadedImage();
        // });
        
        // // Submit Handler
        // submitButton.addEventListener('click', () => {
        //     if (uploadedImage) {
        //         runDetection();
        //     }
        // });
        
        // // Tab Switching
        // originalTab.addEventListener('click', () => {
        //     switchTab('original');
        // });
        
        // annotatedTab.addEventListener('click', () => {
        //     switchTab('annotated');
        // });
        
        // // Functions
        // function handleFileUpload(file) {
        //     if (!file.type.startsWith('image/')) {
        //         alert('Please select a valid image file.');
        //         return;
        //     }
            
        //     if (file.size > 10 * 1024 * 1024) { // 10MB limit
        //         alert('File size must be less than 10MB.');
        //         return;
        //     }
            
        //     const reader = new FileReader();
        //     reader.onload = (e) => {
        //         uploadedImage = file;
        //         originalImageData = e.target.result;
                
        //         imagePreview.src = e.target.result;
        //         imagePreview.classList.remove('hidden');
        //         uploadPrompt.classList.add('hidden');
        //         clearButton.classList.remove('hidden');
        //         submitButton.disabled = false;
                
        //         // Reset results
        //         resetResults();
        //     };
        //     reader.readAsDataURL(file);
        // }
        
        // function clearUploadedImage() {
        //     uploadedImage = null;
        //     originalImageData = null;
        //     annotatedImageData = null;
            
        //     imagePreview.classList.add('hidden');
        //     uploadPrompt.classList.remove('hidden');
        //     clearButton.classList.add('hidden');
        //     submitButton.disabled = true;
        //     fileInput.value = '';
            
        //     // Reset results
        //     resetResults();
        // }
        
        // function resetResults() {
        //     // Hide results and show placeholders
        //     imageDisplay.classList.add('hidden');
        //     placeholderImages.classList.remove('hidden');
        //     quantificationResults.classList.add('hidden');
        //     placeholderQuantification.classList.remove('hidden');
        //     processingSpinner.classList.add('hidden');
        // }
        
        // function runDetection() {
        //     // Show processing state
        //     placeholderImages.classList.add('hidden');
        //     processingSpinner.classList.remove('hidden');
            
        //     // Update button state
        //     submitButton.disabled = true;
        //     submitText.textContent = 'Processing...';
        //     submitIcon.innerHTML = `
        //         <svg class="h-5 w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        //             <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        //             <path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        //         </svg>
        //     `;
            
        //     // Simulate AI processing (replace with actual API call)
        //     setTimeout(() => {
        //         // Create mock annotated image (in reality, this would come from your ML model)
        //         annotatedImageData = originalImageData; // For demo, use same image
                
        //         // Show results
        //         processingSpinner.classList.add('hidden');
        //         imageDisplay.classList.remove('hidden');
        //         placeholderQuantification.classList.add('hidden');
        //         quantificationResults.classList.remove('hidden');
                
        //         // Set initial image view
        //         switchTab('original');
                
        //         // Update quantification with realistic values
        //         updateQuantificationResults();
                
        //         // Reset button state
        //         submitButton.disabled = false;
        //         submitText.textContent = 'Run Detection';
        //         submitIcon.innerHTML = `
        //             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
        //         `;
                
        //         // Scroll to results
        //         document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        //     }, 3000);
        // }
        
        // function switchTab(tab) {
        //     currentView = tab;
            
        //     if (tab === 'original') {
        //         originalTab.classList.remove('bg-slate-200', 'text-slate-700');
        //         originalTab.classList.add('bg-emerald-600', 'text-white');
        //         annotatedTab.classList.remove('bg-emerald-600', 'text-white');
        //         annotatedTab.classList.add('bg-slate-200', 'text-slate-700');
        //         resultImage.src = originalImageData;
        //     } else {
        //         annotatedTab.classList.remove('bg-slate-200', 'text-slate-700');
        //         annotatedTab.classList.add('bg-emerald-600', 'text-white');
        //         originalTab.classList.remove('bg-emerald-600', 'text-white');
        //         originalTab.classList.add('bg-slate-200', 'text-slate-700');
        //         resultImage.src = annotatedImageData;
        //     }
        // }
        
        // function updateQuantificationResults() {
        //     // Generate realistic random values
        //     const etioplastArea = (Math.random() * 15 + 8).toFixed(1);
        //     const plbArea = (Math.random() * 5 + 2).toFixed(1);
        //     const prothylakoidCount = Math.floor(Math.random() * 8 + 2);
        //     const prothylakoidLength = (Math.random() * 60 + 30).toFixed(1);
        //     const plastoglobuleCount = Math.floor(Math.random() * 30 + 15);
        //     const plastoglobuleDiameter = (Math.random() * 0.4 + 0.15).toFixed(2);
            
        //     // Update DOM elements
        //     document.getElementById('etioplast-area').textContent = `${etioplastArea} µm²`;
        //     document.getElementById('plb-area').textContent = `${plbArea} µm²`;
        //     document.getElementById('prothylakoid-count').textContent = prothylakoidCount.toString();
        //     document.getElementById('prothylakoid-length').textContent = `${prothylakoidLength} µm`;
        //     document.getElementById('plastoglobule-count').textContent = plastoglobuleCount.toString();
        //     document.getElementById('plastoglobule-diameter').textContent = `${plastoglobuleDiameter} µm`;
        // }
        
        // // Smooth scrolling for navigation links
        // document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        //     anchor.addEventListener('click', function (e) {
        //         e.preventDefault();
        //         const target = document.querySelector(this.getAttribute('href'));
        //         if (target) {
        //             target.scrollIntoView({
        //                 behavior: 'smooth',
        //                 block: 'start'
        //             });
        //         }
        //         // Close mobile menu if open
        //         mobileMenu.classList.add('hidden');
        //     });
        // });
        
        // // Initialize
        // document.addEventListener('DOMContentLoaded', () => {
        //     // Add entrance animations
        //     document.body.classList.add('animate-fade-in');
        // });


// document.addEventListener('DOMContentLoaded', () => {
//     // DOM Elements
//     const imageUploadBox = document.getElementById('image-upload-box');
//     const fileInput = document.getElementById('file-input');
//     const imagePreview = document.getElementById('image-preview');
//     const uploadPrompt = document.getElementById('upload-prompt');
//     const submitButton = document.getElementById('submit-button');
//     const clearButton = document.getElementById('clear-button');
//     const submitIcon = document.getElementById('submit-icon');
//     const submitText = document.getElementById('submit-text');
//     const mobileMenuButton = document.getElementById('mobile-menu-button');
//     const mobileMenu = document.getElementById('mobile-menu');
    
//     // Result Elements
//     const placeholderImages = document.getElementById('placeholder-images');
//     const processingSpinner = document.getElementById('processing-spinner');
//     const imageDisplay = document.getElementById('image-display');
//     const resultImage = document.getElementById('result-image');
//     const originalTab = document.getElementById('original-tab');
//     const annotatedTab = document.getElementById('annotated-tab');
    
//     // Quantification Elements
//     const placeholderQuantification = document.getElementById('placeholder-quantification');
//     const quantificationResults = document.getElementById('quantification-results');
    
//     // State
//     let uploadedImage = null;
//     let originalImageData = null;
//     let annotatedImageData = null;
//     let currentView = 'original';
    
//     // Mobile Menu Toggle
//     mobileMenuButton.addEventListener('click', () => {
//         mobileMenu.classList.toggle('hidden');
//     });
    
//     // File Upload Handlers
//     imageUploadBox.addEventListener('click', () => {
//         if (!uploadedImage) {
//             fileInput.click();
//         }
//     });
    
//     imageUploadBox.addEventListener('dragover', (e) => {
//         e.preventDefault();
//         imageUploadBox.classList.add('border-emerald-400', 'bg-emerald-50/50');
//     });
    
//     imageUploadBox.addEventListener('dragleave', (e) => {
//         e.preventDefault();
//         imageUploadBox.classList.remove('border-emerald-400', 'bg-emerald-50/50');
//     });
    
//     imageUploadBox.addEventListener('drop', (e) => {
//         e.preventDefault();
//         imageUploadBox.classList.remove('border-emerald-400', 'bg-emerald-50/50');
//         const files = e.dataTransfer.files;
//         if (files.length > 0) {
//             handleFileUpload(files[0]);
//         }
//     });
    
//     fileInput.addEventListener('change', (e) => {
//         if (e.target.files.length > 0) {
//             handleFileUpload(e.target.files[0]);
//         }
//     });
    
//     // Clear Image Handler
//     clearButton.addEventListener('click', () => {
//         clearUploadedImage();
//     });
    
//     // Submit Handler
//     submitButton.addEventListener('click', () => {
//         if (uploadedImage) {
//             runDetection();
//         }
//     });
    
//     // Tab Switching
//     originalTab.addEventListener('click', () => {
//         switchTab('original');
//     });
    
//     annotatedTab.addEventListener('click', () => {
//         switchTab('annotated');
//     });
    
//     // Functions
//     function handleFileUpload(file) {
//         if (!file.type.startsWith('image/')) {
//             alert('Please select a valid image file.');
//             return;
//         }
        
//         if (file.size > 10 * 1024 * 1024) { // 10MB limit
//             alert('File size must be less than 10MB.');
//             return;
//         }
        
//         const reader = new FileReader();
//         reader.onload = (e) => {
//             uploadedImage = file;
//             originalImageData = e.target.result;
            
//             imagePreview.src = e.target.result;
//             imagePreview.classList.remove('hidden');
//             uploadPrompt.classList.add('hidden');
//             clearButton.classList.remove('hidden');
//             submitButton.disabled = false;
            
//             // Reset results
//             resetResults();
//         };
//         reader.readAsDataURL(file);
//     }
    
//     function clearUploadedImage() {
//         uploadedImage = null;
//         originalImageData = null;
//         annotatedImageData = null;
        
//         imagePreview.classList.add('hidden');
//         uploadPrompt.classList.remove('hidden');
//         clearButton.classList.add('hidden');
//         submitButton.disabled = true;
//         fileInput.value = '';
        
//         // Reset results
//         resetResults();
//     }
    
//     function resetResults() {
//         // Hide results and show placeholders
//         imageDisplay.classList.add('hidden');
//         placeholderImages.classList.remove('hidden');
//         quantificationResults.classList.add('hidden');
//         placeholderQuantification.classList.remove('hidden');
//         processingSpinner.classList.add('hidden');
//     }
    
//     async function runDetection() {
//         // Show processing state
//         placeholderImages.classList.add('hidden');
//         processingSpinner.classList.remove('hidden');
        
//         // Update button state
//         submitButton.disabled = true;
//         submitText.textContent = 'Processing...';
//         submitIcon.innerHTML = `
//             <svg class="h-5 w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
//                 <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
//                 <path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
//             </svg>
//         `;
        
//         try {
//             // Prepare FormData for API call
//             const formData = new FormData();
//             formData.append('image', uploadedImage);
            
//             // Make API call
//             const response = await fetch('http://127.0.0.1:8000/api/detect-analyze/', {
//                 method: 'POST',
//                 body: formData,
//             });
            
//             if (!response.ok) {
//                 throw new Error(`API request failed with status ${response.status}`);
//             }
            
//             const data = await response.json();
//             const { analysis, output_image_url } = data;
            
//             // Update annotated image
//             annotatedImageData = output_image_url;
            
//             // Show results
//             processingSpinner.classList.add('hidden');
//             imageDisplay.classList.remove('hidden');
//             placeholderQuantification.classList.add('hidden');
//             quantificationResults.classList.remove('hidden');
            
//             // Set initial image view
//             switchTab('original');
            
//             // Update quantification with API data
//             updateQuantificationResults(analysis);
            
//             // Reset button state
//             submitButton.disabled = false;
//             submitText.textContent = 'Run Detection';
//             submitIcon.innerHTML = `
//                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
//             `;
            
//             // Scroll to results
//             document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
//         } catch (error) {
//             console.error('Error during detection:', error);
//             alert('Failed to process image. Please try again.');
//             processingSpinner.classList.add('hidden');
//             placeholderImages.classList.remove('hidden');
//             submitButton.disabled = false;
//             submitText.textContent = 'Run Detection';
//             submitIcon.innerHTML = `
//                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
//             `;
//         }
//     }
    
//     function switchTab(tab) {
//         currentView = tab;
        
//         if (tab === 'original') {
//             originalTab.classList.remove('bg-slate-200', 'text-slate-700');
//             originalTab.classList.add('bg-emerald-600', 'text-white');
//             annotatedTab.classList.remove('bg-emerald-600', 'text-white');
//             annotatedTab.classList.add('bg-slate-200', 'text-slate-700');
//             resultImage.src = originalImageData;
//         } else {
//             annotatedTab.classList.remove('bg-slate-200', 'text-slate-700');
//             annotatedTab.classList.add('bg-emerald-600', 'text-white');
//             originalTab.classList.remove('bg-emerald-600', 'text-white');
//             originalTab.classList.add('bg-slate-200', 'text-slate-700');
//             resultImage.src = annotatedImageData;
//         }
//     }
    
//     function updateQuantificationResults(analysis) {
//         // Update DOM elements with API data
//         document.getElementById('etioplast-area').textContent = `${analysis.Etioplast.area_um2.toFixed(2)} µm²`;
//         document.getElementById('plb-area').textContent = `${analysis.PLB.area_um2.toFixed(2)} µm²`;
//         document.getElementById('prothylakoid-count').textContent = analysis.Prothylakoid.count.toString();
//         document.getElementById('prothylakoid-length').textContent = `${analysis.Prothylakoid.total_length_um.toFixed(2)} µm`;
//         document.getElementById('plastoglobule-count').textContent = analysis.Plastoglobule.count.toString();
//         document.getElementById('plastoglobule-diameter').textContent = `${analysis.Plastoglobule.avg_diameter_um.toFixed(2)} µm`;
//     }
    
//     // Smooth scrolling for navigation links
//     document.querySelectorAll('a[href^="#"]').forEach(anchor => {
//         anchor.addEventListener('click', function (e) {
//             e.preventDefault();
//             const target = document.querySelector(this.getAttribute('href'));
//             if (target) {
//                 target.scrollIntoView({
//                     behavior: 'smooth',
//                     block: 'start'
//                 });
//             }
//             // Close mobile menu if open
//             mobileMenu.classList.add('hidden');
//         });
//     });
    
//     // Initialize
//     document.body.classList.add('animate-fade-in');
// });




document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const imageUploadBox = document.getElementById('image-upload-box');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadPrompt = document.getElementById('upload-prompt');
    const submitButton = document.getElementById('submit-button');
    const clearButton = document.getElementById('clear-button');
    const submitIcon = document.getElementById('submit-icon');
    const submitText = document.getElementById('submit-text');
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    // Result Elements
    const placeholderImages = document.getElementById('placeholder-images');
    const processingSpinner = document.getElementById('processing-spinner');
    const imageDisplay = document.getElementById('image-display');
    const resultImage = document.getElementById('result-image');
    const originalTab = document.getElementById('original-tab');
    const annotatedTab = document.getElementById('annotated-tab');
    
    // Quantification Elements
    const placeholderQuantification = document.getElementById('placeholder-quantification');
    const quantificationResults = document.getElementById('quantification-results');
    
    // State
    let uploadedImage = null;
    let originalImageData = null;
    let annotatedImageData = null;
    let currentView = 'original';
    
    // Mobile Menu Toggle
    mobileMenuButton.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
    });
    
    // File Upload Handlers
    imageUploadBox.addEventListener('click', () => {
        if (!uploadedImage) {
            fileInput.click();
        }
    });
    
    imageUploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageUploadBox.classList.add('border-emerald-400', 'bg-emerald-50/50');
    });
    
    imageUploadBox.addEventListener('dragleave', (e) => {
        e.preventDefault();
        imageUploadBox.classList.remove('border-emerald-400', 'bg-emerald-50/50');
    });
    
    imageUploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        imageUploadBox.classList.remove('border-emerald-400', 'bg-emerald-50/50');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Clear Image Handler
    clearButton.addEventListener('click', () => {
        clearUploadedImage();
    });
    
    // Submit Handler
    submitButton.addEventListener('click', () => {
        if (uploadedImage) {
            runDetection();
        }
    });
    
    // Tab Switching
    originalTab.addEventListener('click', () => {
        switchTab('original');
    });
    
    annotatedTab.addEventListener('click', () => {
        switchTab('annotated');
    });
    
    // Functions
    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            alert('File size must be less than 10MB.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage = file;
            originalImageData = e.target.result;
            
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadPrompt.classList.add('hidden');
            clearButton.classList.remove('hidden');
            submitButton.disabled = false;
            
            // Reset results
            resetResults();
        };
        reader.readAsDataURL(file);
    }
    
    function clearUploadedImage() {
        uploadedImage = null;
        originalImageData = null;
        annotatedImageData = null;
        
        imagePreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        clearButton.classList.add('hidden');
        submitButton.disabled = true;
        fileInput.value = '';
        
        // Reset results
        resetResults();
    }
    
    function resetResults() {
        // Hide results and show placeholders
        imageDisplay.classList.add('hidden');
        placeholderImages.classList.remove('hidden');
        quantificationResults.classList.add('hidden');
        placeholderQuantification.classList.remove('hidden');
        processingSpinner.classList.add('hidden');
    }
    
    async function runDetection() {
        // Show processing state
        placeholderImages.classList.add('hidden');
        processingSpinner.classList.remove('hidden');
        
        // Update button state
        submitButton.disabled = true;
        submitText.textContent = 'Processing...';
        submitIcon.innerHTML = `
            <svg class="h-5 w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
        `;
        
        try {
            // Prepare FormData for API call
            const formData = new FormData();
            formData.append('image', uploadedImage);
            console.log('FormData prepared:', formData);
            // Make API call
            const response = await fetch('http://127.0.0.1:8000/api/detect-analyze/', {
                method: 'POST',
                body: formData,
            });
            console.log(response)
            if (!response.ok) {
                throw new Error(`API request failed with status ${response.status}`);
            }
            
            const data = await response.json();
            const { analysis, output_image_url } = data;
            
            // Update annotated image
            annotatedImageData = output_image_url;
            
            // Show results
            processingSpinner.classList.add('hidden');
            imageDisplay.classList.remove('hidden');
            placeholderQuantification.classList.add('hidden');
            quantificationResults.classList.remove('hidden');
            
            // Set initial image view
            switchTab('original');
            
            // Update quantification with API data
            updateQuantificationResults(analysis);
            
            // Reset button state
            submitButton.disabled = false;
            submitText.textContent = 'Run Detection';
            submitIcon.innerHTML = `
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            `;
            
            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error('Error during detection:', error);
            //alert('Failed to process image. Please try again.');
            processingSpinner.classList.add('hidden');
            placeholderImages.classList.remove('hidden');
            submitButton.disabled = false;
            submitText.textContent = 'Run Detection';
            submitIcon.innerHTML = `
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            `;
        }
    }
    
    function switchTab(tab) {
        currentView = tab;
        
        if (tab === 'original') {
            originalTab.classList.remove('bg-slate-200', 'text-slate-700');
            originalTab.classList.add('bg-emerald-600', 'text-white');
            annotatedTab.classList.remove('bg-emerald-600', 'text-white');
            annotatedTab.classList.add('bg-slate-200', 'text-slate-700');
            resultImage.src = originalImageData;
        } else {
            annotatedTab.classList.remove('bg-slate-200', 'text-slate-700');
            annotatedTab.classList.add('bg-emerald-600', 'text-white');
            originalTab.classList.remove('bg-emerald-600', 'text-white');
            originalTab.classList.add('bg-slate-200', 'text-slate-700');
            resultImage.src = annotatedImageData;
        }
    }
    
    function updateQuantificationResults(analysis) {
        // Update DOM elements with API data
        document.getElementById('etioplast-area').textContent = `${analysis.Etioplast.area_um2.toFixed(2)} µm²`;
        document.getElementById('plb-area').textContent = `${analysis.PLB.area_um2.toFixed(2)} µm²`;
        document.getElementById('prothylakoid-count').textContent = analysis.Prothylakoid.count.toString();
        document.getElementById('prothylakoid-length').textContent = `${analysis.Prothylakoid.total_length_um.toFixed(2)} µm`;
        document.getElementById('plastoglobule-count').textContent = analysis.Plastoglobule.count.toString();
        document.getElementById('plastoglobule-diameter').textContent = `${analysis.Plastoglobule.avg_diameter_um.toFixed(2)} µm`;
    }
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
            // Close mobile menu if open
            mobileMenu.classList.add('hidden');
        });
    });
    
    // Initialize
    document.body.classList.add('animate-fade-in');
});