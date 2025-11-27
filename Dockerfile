# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for better caching)
COPY requirements.txt .

# Istall dependencies
# W use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Cpy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# We bind to 0.0.0.0 so the container is accessible from outside
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
