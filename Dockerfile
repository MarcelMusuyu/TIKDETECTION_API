FROM python:3.10-slim

WORKDIR /code

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copie du code et des modèles
COPY ./app /code/app

# Commande de lancement utilisant le port dynamique de Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-80}"]