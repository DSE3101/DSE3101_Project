# Example for a Flask or Dash application using Gunicorn
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the correct port
EXPOSE $PORT

# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:server"]
