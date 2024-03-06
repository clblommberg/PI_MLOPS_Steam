**Proyecto de Análisis de Datos y Sistema Recomendaciones para Juegos de Steam**

---

### Descripción del Proyecto

Este proyecto utiliza FastAPI y Pydantic para crear una API que ofrece análisis y recomendaciones relacionadas con juegos de Steam. El proyecto se enfoca en el procesamiento y análisis de datos, abordando áreas como análisis de sentimientos en reseñas, estadísticas de usuarios, información de desarrolladores y más.

- Accede a Deploy del Proyecto:
[Web API](https://pi-mlops-steam-3u6q.onrender.com/api/v1/docs)

### Estructura del Proyecto

- **Main Script (`server/main.py`):**
  - Contiene la lógica principal de la API utilizando el framework FastAPI.
  - Define varios modelos Pydantic para la validación de datos de entrada y salida.
  - Implementa varios endpoints para acceder a diferentes funcionalidades del proyecto.

- **Procesamiento de Sentimientos (`SentimentAnalysisProcessor`):**
  - Clase encargada de realizar el análisis de sentimientos en las reseñas de los juegos.
  - Procesa el DataFrame post ETL y genera una nueva versión con información de sentimientos.

- **Modelado de Algoritmo (`cosine_similarity`):**
  - Clase encargada de generar la matriz de carasteristicas para las las reseñas de los juegos.
  - Recibe un dataframe escalado (Post Métodos Label encoder y Hot Enconder) lo que ayuda al redimiento para las estimaciones en el sistema de recomendación.

- **Notebook de Data Science (`notebooks/*.ipynb`):**
  - Contiene el desarrollo desde la etapa inicial EDA, ETL(Modelamiento), Entrenamiento y adicionalmente al marco de presentación para Henry, propone predecir ¿Si un usuario recomendara o No el juego? combinando `SentimentAnalysisProcessor` y `DecisionTreeClassifier`, en proceso para realizar estimaciones en el proyecto.

---
### Endpoints

1. **Información del Desarrollador:**
   - **Endpoint:** `/developer-items-data`
   - **Descripción:** Obtiene información detallada sobre un desarrollador específico, incluyendo la cantidad de items y el contenido gratuito en diferentes años.

2. **Estadísticas de Usuario:**
   - **Endpoint:** `/user-data-stats/{user_id:str}`
   - **Descripción:** Proporciona estadísticas para un usuario específico basadas en su ID de Steam, incluyendo el dinero gastado, el porcentaje de recomendación y la cantidad total de items.

3. **Horas Jugadas por Género:**
   - **Endpoint:** `/user-playtime-by-genre`
   - **Descripción:** Muestra las horas jugadas por usuarios para un género específico y el usuario con más horas jugadas en ese género.

4. **Desarrolladores Principales por Año:**
   - **Endpoint:** `/top-developers-by-year`
   - **Descripción:** Identifica a los desarrolladores principales en un año específico según la cantidad de juegos recomendados.

5. **Análisis de Reseñas por Desarrollador:**
   - **Endpoint:** `/developer-reviews-analysis`
   - **Descripción:** Analiza las reseñas de un desarrollador específico, contando la cantidad de reseñas positivas, negativas y neutrales.

6. **Sistema de Recomendación: Juegos Similares:**
   - **Endpoint:** `/similar_item_id`
   - **Descripción:** Se ha implementado un sistema de recomendación basado en la similitud de juegos. Referenciando Id productos  pueden obtener una lista de juegos similares a uno de referencia.
7. **Sistema de Recomendación: Juegos Similares con User ID:**
   - **Endpoint:** `/get_similar_games`
   - **Descripción:** Se ha implementado un sistema de recomendación basado en la similitud de juegos. Los usuarios pueden obtener una lista de juegos similares a uno de referencia.
---
##  Machine Learning a Sistemas de Recomendación 
### Indicador de reviews de usuarios de Steam.
#### Preparación de Datos
- **Hallazgos**:

- En la tabla steam_games, todas las columnas tienen un porcentaje significativo de valores nulos o vacíos, que oscila entre aproximadamente el 20% y el 26.6%.
- En la tabla user_reviews, la mayoría de las columnas tienen el 100% de valores nulos o vacíos, excepto las columnas review, funny y last_edited, que tienen algunos valores presentes.
- En la tabla users_items, todas las columnas tienen el 100% de valores nulos o vacíos.
- Se estandarizaron 47675 registros en  modelo de usuarios Steam para el análisis.
- Se dividió el conjunto de datos en conjuntos de entrenamiento y prueba, con un 80% para entrenamiento y un 20% para prueba.
- Se categorizaron 18 variables como factores de riesgo relevantes.
- Revisar supuestos:<br>
[1_EDA.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/1_EDA.ipynb)<br>
[2_Modelamiento.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/2_Modelamiento.ipynb)<br>

### Entrenamiento y Predicciones `SentimentAnalysisProcessor` y `DecisionTreeClassifier`.
- Se entrenó un modelo de Árboles de Decisión para predecir riesgo de biopsias. 
- El modelo tuvo una precisión de `78%` y capacidad de detección (recall) de `63%` en el conjunto de prueba.
- Predijo correctamente un`78%` de casos de alto riesgo en la validación.
- Revisar supuestos:<br>
[3_Entrenamiento.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/3_Entrenamiento.ipynb)<br>
[4_Prediciones.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/4_Prediciones.ipynb)

### Explicación a Usuarios y/o Inversores.
- Con la precisión actual del `78%`, más de la mitad de las predicciones serían correctas, lo cual es aceptable.  
- Para uso en proyectos reales se debería alcanzar al menos una precisión cerca del 80%, con recall sobre 60%.
- Estamos trabajando en conseguir datos adicionales de usuarios de Steam para mejorar la detección de los casos complejos.
- Agradecemos su paciencia; los mantendremos informados sobre el progreso en las próximas semanas.

En resumen, el desempeño actual del `78%`de precisión es viable para en proyectos reales confiable. Trabajaremos en mejorar la calidad del modelo mediante la incorporación de más datos de en sistemas de recomendación. 
### Consideraciones Importantes

- El proyecto utiliza el modelo NLTK para el análisis de sentimientos en las reseñas de juegos.
- Incorporar un sistema de recomendación de juegos basado en el historial de juegos y preferencias de los usuarios.
- Implementar autenticación y autorización para proteger ciertos endpoints sensibles.
- Mejorar la modularidad del código para facilitar la incorporación de nuevas funcionalidades.
