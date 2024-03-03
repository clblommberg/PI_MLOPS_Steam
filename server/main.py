from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import nltk

nltk.download('vader_lexicon')

# Crear instancia de FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a tu API personalizada con FastAPI!"}

# Endpoint para redirigir a la documentación
@app.get("/docs", include_in_schema=False)
def custom_docs_redirect():
    return RedirectResponse(url="/docs")

from nltk.sentiment import SentimentIntensityAnalyzer
# Load the pre-trained sentiment analysis model

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes ajustar esto según tus necesidades
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sentiment_analyzer_model = SentimentIntensityAnalyzer()


# Load the pre-processed DataFrame
df = pd.read_csv("data_train.csv")


class SentimentAnalysisProcessor:
    def __init__(self, threshold_low=-0.5, threshold_high=0.5):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.analyzer = SentimentIntensityAnalyzer()

    def process_data(self, input_df):
        # Crear una copia del DataFrame de entrada para evitar modificaciones no deseadas
        allowed_genres = ['strategy', 'indie', 'rpg', 'action', 'simulation', 'adventure']
        df_a = input_df[input_df['genres_str'].isin(allowed_genres)]
        df = df_a.copy()

        # Aplicar el análisis de sentimientos a la columna 'review'
        df['scores_review'] = df['review'].apply(lambda x: self.analyzer.polarity_scores(x)['compound'])

        # Crear la nueva columna 'score_new' según las bandas definidas
        df['score_new'] = df['scores_review'].apply(lambda x: 0 if x < self.threshold_low else (2 if x > self.threshold_high else 1))

        # Clasificar las reseñas numéricamente
        df['sentiment_numeric'] = df['score_new']

        # Clasificar las reseñas como "Positive", "Negative" o "Neutral"
        df['sentiment_more_less'] = df['score_new'].apply(lambda x: 'Positive' if x == 2 else ('Negative' if x == 0 else 'Neutral'))

        # Eliminar las columnas 'review' y 'scores_review' si es necesario
        columns_to_drop = ['review', 'scores_review', 'release_date', 'posted_date', 'item_name', 'sentiment_numeric']
        df = df.drop(columns=columns_to_drop)
        df['steam_id'] = df['steam_id'].astype('object')
        return df_a, df

# Crear una instancia de la clase
sentiment_processor = SentimentAnalysisProcessor()
# Pasar el DataFrame a la clase para el procesamiento
df_init, output_df = sentiment_processor.process_data(df)

# Definir modelo Pydantic para la información de juego
class GameInfo(BaseModel):
    genres_str: str
    app_name: str
    price: float
    game_id: int
    developer: str
    year: int
    recommend: int
    items_count: int
    steam_id: int
    playtime_forever: int
    playtime_2weeks: int
    score_new: int
    sentiment_more_less: str

# Definir modelos Pydantic para parámetros y resultados
class DeveloperParams(BaseModel):
    dev: str

# Definir modelo Pydantic para el resultado del desarrollador
class DeveloperResult(BaseModel):
    Año: int
    Cantidad_de_Items: int
    Contenido_Free: str
    
# Definir función para obtener información del desarrollador
def get_developer_info(params: DeveloperParams = Depends()):
    try:
        # Filtrar el DataFrame para obtener solo las filas del desarrollador especificado
        dev_df = output_df[output_df['developer'] == params.dev]

        if not dev_df.empty:
            # Agrupar por año y sumar la cantidad de items
            grouped = dev_df.groupby('year')['items_count'].sum().reset_index()

            result = []

            for row in grouped.itertuples(index=False):
                # Filtrar solo items de este desarrollador y año
                year_dev_df = dev_df[(dev_df['developer'] == params.dev) & (dev_df['year'] == row.year)]

                # Contar items gratuitos (precio <= 0)
                free_items = year_dev_df[year_dev_df['price'] <= 0]['items_count'].sum()

                # Calcular % de items gratuitos
                pct_free = (free_items / row.items_count) * 100 if row.items_count > 0 else 0

                # Redondear % a 2 decimales
                pct_free = round(pct_free, 2)

                result.append(DeveloperResult(
                    Año=int(row.year),
                    Cantidad_de_Items=int(row.items_count),
                    Contenido_Free=f"{pct_free}%"
                ))

            return result

        else:
            # Devolver una respuesta de error si no se encuentra información para el desarrollador
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró información del desarrollador '{params.dev}'"
            )

    except Exception as e:
        # Manejar cualquier excepción y devolver una respuesta de error
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )

# Agregar el nuevo endpoint a la aplicación
@app.get("/developer-info", response_model=List[DeveloperResult])
def get_developer_info_endpoint(result: List[DeveloperResult] = Depends(get_developer_info)):
    return result

# Definir modelo Pydantic para las estadísticas de usuario
class UserStats(BaseModel):
    Usuario: str
    Dinero_gastado: str
    Porcentaje_recomendacion: str
    Cantidad_items: int

# Definir modelo Pydantic para parámetros de SteamId
class SteamIdParams(BaseModel):
    steam_id: str

def convert_to_python_types(record):
    record_dict = record.to_dict(orient='records')[0]
    
    for key, value in record_dict.items():
        if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
            record_dict[key] = pd.to_numeric(value)  # Convertir a tipo nativo de Python
        elif isinstance(value, pd.Timestamp):
            record_dict[key] = value.strftime('%Y-%m-%dT%H:%M:%S.%fZ')  # Formatear fechas
        else:
            record_dict[key] = value
    
    return record_dict

# Definir función para obtener estadísticas de usuario
def get_user_stats(params: SteamIdParams = Depends()):
    try:
        # Convertir 'params.steam_id' a cadena para asegurar la comparación
        user_df = output_df[output_df['steam_id'].astype(str) == params.steam_id]

        # Verificar si hay datos para el usuario
        if user_df.empty:
            raise HTTPException(status_code=404, detail="No se encontraron datos para este usuario.")

        # Calcular el dinero gastado (suma de price * items_count)
        money_spent = (user_df['price'] * user_df['items_count']).sum()

        # Sumar las recomendaciones donde recommend es igual a 1
        total_recommendations = user_df[user_df['recommend'] == 1]['items_count'].sum()

        # Sumar la cantidad total de items
        total_items = user_df['items_count'].sum()

        # Calcular el porcentaje de recomendación basado en las sumas
        pct_recommended = (total_recommendations / total_items) * 100 if total_items > 0 else 0

        # Formatear el porcentaje de recomendación
        pct_recommended_str = '{:.2f}%'.format(pct_recommended)

        # Crear y devolver el objeto Pydantic con la información
        return UserStats(
            Usuario=params.steam_id,
            Dinero_gastado='${:.2f} USD'.format(money_spent),
            Porcentaje_recomendacion=pct_recommended_str,
            Cantidad_items=total_items,
        )

    except Exception as e:
        # Manejar cualquier excepción y devolver una respuesta de error
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/user-stats/{steam_id:str}", response_model=UserStats)
def get_user_stats_endpoint(stats: UserStats = Depends(get_user_stats)):
    return stats


# Definir modelo Pydantic para parámetros de género
class GenreParams(BaseModel):
    genre: str

# Definir función para el nuevo endpoint
def get_user_playtime_by_genre(params: GenreParams = Depends()):
    try:
        # Filtrar el DataFrame para el género específico
        genre_df = output_df[output_df['genres_str'] == params.genre].copy()

        # Verificar si hay datos para el género
        if genre_df.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el género {params.genre}.")

        # Encontrar el usuario con más horas jugadas
        max_playtime_user = genre_df.loc[genre_df['playtime_forever'].idxmax()]['steam_id']

        # Convertir las horas de minutos a horas y redondear a números enteros
        genre_df['playtime_hours'] = genre_df['playtime_forever'] // 60

        # Calcular la acumulación de horas jugadas por año
        playtime_by_year = genre_df.groupby('year')['playtime_hours'].sum().reset_index()
        playtime_list = [{"Año": int(year), "Horas": int(hours)} for year, hours in zip(playtime_by_year['year'], playtime_by_year['playtime_hours'])]

        # Crear y devolver el diccionario con la información
        return {
            f"Usuario con más horas jugadas para {params.genre}": max_playtime_user,
            "Horas jugadas": playtime_list,
        }

    except Exception as e:
        # Manejar cualquier excepción y devolver una respuesta de error
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# Agregar el nuevo endpoint a la aplicación
@app.get("/user-playtime-by-genre", response_model=dict)
def user_playtime_by_genre_endpoint(result: dict = Depends(get_user_playtime_by_genre)):
    return result


# Definir modelo Pydantic para parámetros de año
class YearParams(BaseModel):
    year: int

# Definir función para el nuevo endpoint
def get_top_developers_by_year(params: YearParams = Depends()):
    try:
        # Filtrar el DataFrame para el año específico
        year_df = output_df[output_df['year'] == params.year]

        # Verificar si hay datos para el año
        if year_df.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el año {params.year}.")

        # Filtrar por juegos recomendados y con comentarios positivos
        recommended_df = year_df[(year_df['recommend'] == 1) & (year_df['score_new'] > 0)]

        # Agrupar por desarrollador y contar la cantidad de juegos recomendados
        developer_counts = recommended_df.groupby('developer')['recommend'].sum().reset_index()

        # Ordenar en orden descendente y obtener el top 3
        top_developers = developer_counts.sort_values(by='recommend', ascending=False).head(3)

        # Crear la lista de resultados en el formato deseado
        result_list = [{"Puesto {}: {}".format(i + 1, row['developer']): row['recommend']} for i, row in top_developers.iterrows()]

        # Devolver la lista de resultados
        return result_list

    except Exception as e:
        # Manejar cualquier excepción y devolver una respuesta de error
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# Agregar el nuevo endpoint a la aplicación
@app.get("/top-developers-by-year", response_model=List[dict])
def top_developers_by_year_endpoint(result: List[dict] = Depends(get_top_developers_by_year)):
    return result


# Definir modelo Pydantic para parámetros de desarrollador
class DeveloperParams(BaseModel):
    dev: str

# Definir función para el nuevo endpoint
def get_developer_reviews_analysis(params: DeveloperParams = Depends()):
    try:
        # Filtrar el DataFrame para el desarrollador específico
        developer_df = output_df[output_df['developer'] == params.dev].copy()

        # Verificar si hay datos para el desarrollador
        if developer_df.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el desarrollador {params.dev}.")

        # Convertir tipos de datos de NumPy a tipos nativos de Python
        developer_df['sentiment_more_less'] = developer_df['sentiment_more_less'].astype(str)

        # Contar la cantidad de reseñas positivas, negativas y neutrales
        pos = int((developer_df['sentiment_more_less'] == 'Positive').sum())
        neg = int((developer_df['sentiment_more_less'] == 'Negative').sum())
        neu = int((developer_df['sentiment_more_less'] == 'Neutral').sum())

        # Crear el diccionario de resultados en el formato deseado
        results = {params.dev: [{'Positive': pos}, {'Negative': neg}, {'Neutral': neu}]}

        # Devolver el diccionario de resultados
        return results

    except Exception as e:
        # Manejar cualquier excepción y devolver una respuesta de error
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    
# Agregar el nuevo endpoint a la aplicación
@app.get("/developer-reviews-analysis", response_model=Union[dict, List[dict]])
def developer_reviews_analysis_endpoint(result: Union[dict, List[dict]] = Depends(get_developer_reviews_analysis)):
    return result