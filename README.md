**Proyecto de Análisis de Datos y Recomendaciones para Juegos de Steam**

---

### Descripción del Proyecto

Este proyecto utiliza FastAPI y Pydantic para crear una API que ofrece análisis y recomendaciones relacionadas con juegos de Steam. El proyecto se enfoca en el procesamiento y análisis de datos, abordando áreas como análisis de sentimientos en reseñas, estadísticas de usuarios, información de desarrolladores y más.

- Accede a Deploy del Proyecto:
[Web API](https://pi-mlops-steam-3u6q.onrender.com/api/v1/docs)

### Estructura del Proyecto

- **Main Script (`main.py`):**
  - Contiene la lógica principal de la API utilizando el framework FastAPI.
  - Define varios modelos Pydantic para la validación de datos de entrada y salida.
  - Implementa varios endpoints para acceder a diferentes funcionalidades del proyecto.

- **Procesamiento de Sentimientos (`SentimentAnalysisProcessor`):**
  - Clase encargada de realizar el análisis de sentimientos en las reseñas de los juegos.
  - Procesa el DataFrame inicial y genera una nueva versión con información de sentimientos.

- **Notebook de Pruebas (`notebook_test.ipynb`):**
  - Contiene pruebas y ejemplos de cómo utilizar y probar las funciones del proyecto.


### Endpoints Principales

1. **Información del Desarrollador:**
   - **Endpoint:** `/developer-info`
   - **Descripción:** Obtiene información detallada sobre un desarrollador específico, incluyendo la cantidad de items y el contenido gratuito en diferentes años.

2. **Estadísticas de Usuario:**
   - **Endpoint:** `/user-stats/{steam_id:str}`
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

### Consideraciones Importantes

- El proyecto utiliza el modelo NLTK para el análisis de sentimientos en las reseñas de juegos.
- Asegúrate de tener los permisos necesarios para ejecutar la aplicación y acceder a los archivos de datos.

### Posibles Mejoras Futuras

- Incorporar un sistema de recomendación de juegos basado en el historial de juegos y preferencias de los usuarios.
- Implementar autenticación y autorización para proteger ciertos endpoints sensibles.
- Mejorar la modularidad del código para facilitar la incorporación de nuevas funcionalidades.

Este README proporciona una visión general del proyecto y su estructura. Asegúrate de documentar cualquier cambio adicional y seguir las mejores prácticas de desarrollo para garantizar la mantenibilidad y comprensión del código. ¡Disfruta explorando y trabajando en tu proyecto de análisis de datos y recomendaciones para juegos de Steam!