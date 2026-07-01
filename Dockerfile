# Imagen base de Python
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el modelo entrenado
COPY mlops_pipeline/src/modelo_arbol.pkl .

# Copiar el script de la API
COPY mlops_pipeline/src/model_deploy.py .

# Copiar las dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 8000

# Comando para arrancar la API
CMD ["uvicorn", "model_deploy:app", "--host=0.0.0.0", "--port=8000"]