# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (Required for OpenCV/AI image processing)
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project 
COPY . /app/

# Shift working directory to where manage.py is located
WORKDIR /app/mainserver

# Expose the port the app runs on
EXPOSE 8000

# Command to run the server (Development mode fallback: CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"])
# For production, install gunicorn in your requirements.txt and use:
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "mainserver.wsgi:application"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "300", "--workers", "2", "mainserver.wsgi:application"]