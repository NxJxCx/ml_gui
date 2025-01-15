# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Copy the current directory contents into the container
COPY . /app

# Set the working directory in the container
WORKDIR /app

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your application runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
