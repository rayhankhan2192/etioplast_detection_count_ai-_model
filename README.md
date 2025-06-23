# Chloroplast Quantification

This project focuses on the automated quantification of subcellular structures within etioplasts using deep learning and web technologies. A dataset of 500 electron microscopy images (~500 nm resolution) was manually annotated to detect key components: Etioplast, Prothylakoid, Prolamellar Body (PLB), and Plastoglobule. A YOLOv8s model was trained on these annotations for accurate detection and segmentation.

The backend is built with Django, and the frontend uses React, forming a user-friendly web platform that enables users to upload images and receive quantification results, including:

Etioplast area (µm²)

PLB area

Number of prothylakoids

Total prothylakoid length

Number of plastoglobules

Diameter of plastoglobules

Additionally, the system provides generative explanations of results using an AI-based assistant to help interpret biological significance and patterns.