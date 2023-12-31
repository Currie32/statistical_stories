# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Expose port 8080 for running the website
EXPOSE 8080

# Copy the required files
COPY app.py app.yaml footer.py requirements.txt robots.txt sitemap.xml ./
COPY assets assets
COPY pages pages

# Create a virtual environment and activate it
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run the app
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]
