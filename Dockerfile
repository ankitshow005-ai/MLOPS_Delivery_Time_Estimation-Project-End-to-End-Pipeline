# Use lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 5000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]