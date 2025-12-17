FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the necessary files
COPY requirements.txt .
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port for Flask
EXPOSE 7860

# Start the application
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860"]