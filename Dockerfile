# Choose Python 3.8 or higher base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Copy the .env.example file into the container
COPY .env.example /app/.env

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project code into the container
COPY . /app/

# Define the command to run your program
CMD ["python", "bot.py"]
