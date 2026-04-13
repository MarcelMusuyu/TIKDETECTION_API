# Utiliser l'image Python officielle
FROM python:3.10-slim

# Créer un utilisateur pour Hugging Face (sécurité)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /home/user/app

# Installer les dépendances système (nécessite d'être root temporairement)
USER root
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER user

# Installer les dépendances Python
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copier le code
COPY --chown=user . .

# Hugging Face attend que l'application écoute sur le port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
