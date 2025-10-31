# Use official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy Python scripts from src/app into container
COPY src/app/ ./src/app/

# Copy data folder into container
COPY src/data/ ./src/data/

# Copy requirements file
COPY src/app/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the main script
CMD ["python", "src/app/main.py"]