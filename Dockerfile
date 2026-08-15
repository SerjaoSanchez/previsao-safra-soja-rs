FROM python:3.11-slim

# libgomp1: dependência de runtime do LightGBM (OpenMP), ausente na imagem slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/soja_rs/app.py", "--server.address=0.0.0.0"]
