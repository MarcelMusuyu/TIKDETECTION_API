FROM python:3.10-slim

WORKDIR /code

# Installation des dépendances avec correction des miroirs et paquets mis à jour
RUN apt-get update --fix-missing && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

# Commande pour Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-80}"]
