# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt requirements.txt

# Upgrade pip and install required packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Expose port 5000 for the Flask API
EXPOSE 5000

# Define environment variable (optional, for production you might set FLASK_ENV=production)
ENV FLASK_APP=app.py

# Run the command to start the Flask app
CMD ["python", "app.py"]
