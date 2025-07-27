# Use a lightweight Python image
FROM python:3.15

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the SentenceTransformer model is downloaded during build
RUN python models/download_model.py

# Expose no ports since this is an offline batch system

# Run the app
CMD ["python", "app.py"]
