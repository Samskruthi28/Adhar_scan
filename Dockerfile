FROM python:3.10

# Set the working directory
WORKDIR /app

# Update the package lists and install necessary dependencies
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN    apt-get install -y tesseract-ocr python3 python3-pip

# Set the environment variable for pytesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# Copy the Flask app files to the container
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port on which the Flask app will run (change it if necessary)
EXPOSE 4000

# Set the entry point for the container
CMD ["python3", "adhar.py"]
