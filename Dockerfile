# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any necessary system dependencies (e.g., if needed for certain Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Install the Python dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 6: Expose the port your application runs on (if using Flask or FastAPI, the default is 5000)
EXPOSE 5000

# Step 7: Define environment variables (optional)
ENV FLASK_ENV=production

# Step 8: Specify the command to run your app using `gunicorn` for performance
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

# Step 9: Clean up any unnecessary files (optional step)
RUN apt-get remove --purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean
