FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scheduler code
COPY main.py .

# Run scheduler
CMD ["python", "main.py"]
