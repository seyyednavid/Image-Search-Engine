# Use an official Python runtime as base
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app


# Install system dependencies (needed for numpy, pillow, opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the app code
COPY models/ ./models
COPY . .

# Expose port for Flask
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
