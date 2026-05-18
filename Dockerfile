# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies specified in requirements.txt
# --no-cache-dir keeps the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This copies everything from your local directory into the container's /app directory
COPY . .

# Hugging Face Spaces requires the app to run on port 7860
EXPOSE 7860

# Run the FastAPI application using uvicorn
# App location is app.main:app based on your render.yaml
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
