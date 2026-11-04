FROM python:3.12-slim
WORKDIR /work
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY app/ app/
COPY netflix_titles.csv ./
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
