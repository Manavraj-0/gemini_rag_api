FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your project files (main.py, faiss_index, etc.) into the container
COPY . .

# Tell the container to run your app on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]