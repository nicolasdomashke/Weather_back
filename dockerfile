# Use the official Python image.
FROM python:3.12

COPY requirements.txt /app/requirements.txt

# Set the working directory in the container.
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app.
COPY . /app

# Make port 8000 available to the world outside this container.
EXPOSE 8000

# Run FastAPI using uvicorn.
CMD ["uvicorn", "parser:app", "--host", "0.0.0.0", "--port", "8000"]
