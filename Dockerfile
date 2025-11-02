# 1. Use the official Python Base Image
FROM python:3.10-slim

# 2. Set the working dir
WORKDIR /app

# 3. Copy files
COPY . /app

# 4. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose port
EXPOSE 8200

# 6. Command to run the server
CMD ["uvicorn", "iris_fqast_api:app", "--host", "0.0.0.0", "--port", "8200"]
