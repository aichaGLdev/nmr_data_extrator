# Use the official Python 3.11 image as the base image
FROM python:3.11
# Set the working directory
WORKDIR /usr/src/pyapp
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
# Copy the contents of the local directory into the container at /usr/src/pyapp
COPY . /usr/src/pyapp
# Create a virtual environment and activate it
RUN python -m venv venv
RUN . venv/bin/activate
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Create uploads directory
RUN mkdir -p /usr/src/pyapp/uploads
# Set the uploads directory as a volume
VOLUME /usr/src/pyapp/uploads
# Set permissions for the uploads directory
RUN chmod -R 777 /usr/src/pyapp/uploads
# Expose port 7000
EXPOSE 7000
# Specify the command to run on container start
CMD ["python", "app.py"]
