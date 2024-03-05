# Utiliza la imagen oficial de Python
FROM python:3.10

# Establece el directorio de trabajo en /server
WORKDIR /server

# Copia el archivo de requerimientos al contenedor
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia solo los archivos necesarios desde el directorio local (main.py y data_train.csv) al contenedor
COPY server/main.py .
COPY server/X_train.csv .
COPY server/y_train.csv .

# Comando para ejecutar tu aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]