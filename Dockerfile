# Use an official lightweight Python image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python scripts into the container
COPY src/app/ ./src/app/

# Copy data into the container
COPY src/data/ ./src/data/

# Command to run your Python script when the container starts
CMD ["python", "src/app/main.py"]
