# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app will run on
EXPOSE 7860

# Run the main.py script when the container launches
CMD ["python", "main.py"]
