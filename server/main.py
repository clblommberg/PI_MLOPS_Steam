from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.openapi.models import Response


from pydantic import BaseModel
from typing import List,Optional,  Union, Tuple 
import pandas as pd
import numpy as np
import json

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('vader_lexicon')

app = FastAPI(
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs"
)

from nltk.sentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sentiment_analyzer_model = SentimentIntensityAnalyzer()

# Endpoint para redirigir a la documentación
@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "¡Bienvenido a tu API personalizada con FastAPI!"}

@app.get("/docs", include_in_schema=False)
def custom_docs_redirect():
    return RedirectResponse(url="/api/v1/docs")



df = pd.read_csv('server/data_train.csv')

class SentimentAnalysisProcessor:
    """
    Clase para realizar análisis de sentimientos en reseñas de juegos.

    Parameters:
    - threshold_low (float): Umbral inferior para clasificar sentimientos como negativos. Valor predeterminado: -0.5.
    - threshold_high (float): Umbral superior para clasificar sentimientos como positivos. Valor predeterminado: 0.5.

    Methods:
    - process_data(input_df: pd.DataFrame) -> pd.DataFrame:
        Procesa un DataFrame de entrada aplicando análisis de sentimientos a las reseñas.

    Example:
    >>> sentiment_processor = SentimentAnalysisProcessor()
    >>> output_df = sentiment_processor.process_data(df)
    """
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
        columns_to_drop = ['review', 'scores_review', 'release_date', 'posted_date', 'item_name', 'sentiment_numeric', 'game_id', 'steam_id']
        df = df.drop(columns=columns_to_drop)
        df['users_items_item_id'] = df['users_items_item_id'].astype('object')
        df['user_id'] = df['user_id'].astype('object')
        return df

# Crear una instancia de la clase
sentiment_processor = SentimentAnalysisProcessor()
# Pasar el DataFrame a la clase para el procesamiento
output_df = sentiment_processor.process_data(df)


# Pydantic Información de Juego
class GameInfo(BaseModel):
    genres_str: str
    app_name: str
    price: float
    game_id: int
    developer: str
    year: int
    recommend: int
    items_count: int
    user_id: int
    playtime_forever: int
    playtime_2weeks: int
    score_new: int
    sentiment_more_less: str

# Pydantic para parámetros y resultados desarrollador
class DeveloperParams(BaseModel):
    dev: str

class DeveloperResult(BaseModel):
    Año: int
    Cantidad_de_Items: int
    Contenido_Free: str


# Definir función para obtener información del desarrollador
def get_developer_info(params: DeveloperParams = Depends()):
    """
    Obtiene información del desarrollador sobre la cantidad de items y el contenido gratuito lanzado por año.

    Parameters:
    - params (DeveloperParams): Parámetros de entrada que incluyen el nombre del desarrollador.

    Returns:
    - List[DeveloperResult]: Lista de objetos DeveloperResult con información del desarrollador por año.

    Raises:
    - HTTPException(404): Si no se encuentra información para el desarrollador especificado.
    - HTTPException(500): Si ocurre un error interno durante la ejecución.
    """
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


@app.get("/developer-items-data", response_model=List[DeveloperResult])
def get_developer_info_endpoint(
    dev: str = "Valve",  # Puedes probar con diferentes valores cambiando este parámetro
    result: List[DeveloperResult] = Depends(get_developer_info)
):
    """
    Endpoint para obtener información del desarrollador sobre la cantidad de items y el contenido gratuito lanzado por año.

    Parameters:
    - dev (str): Nombre del desarrollador del cual se busca la información (por defecto es "Valve").

    Returns:
    - List[DeveloperResult]: Lista de objetos DeveloperResult con información del desarrollador por año.
    """
    return result


# Pydantic Estadísticas de usuario
class UserStats(BaseModel):
    Usuario: str
    Dinero_gastado: str
    Porcentaje_recomendacion: str
    Cantidad_items: int

# Pydantic Parámetros de SteamId
class SteamIdParams(BaseModel):
    user_id: str

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


def get_user_stats(params: SteamIdParams = Depends()):
    """
    Obtiene estadísticas del usuario de Steam, incluyendo dinero gastado, porcentaje de recomendación y cantidad de items.

    Parameters:
    - params (SteamIdParams): Parámetros de entrada que incluyen el identificador único del usuario de Steam.

    Returns:
    - UserStats: Objeto UserStats con estadísticas del usuario.

    Raises:
    - HTTPException(404): Si no se encuentran datos para el usuario especificado.
    - HTTPException(500): Si ocurre un error interno durante la ejecución.
    """
    try:
        # Convertir 'params.user_id' a cadena para asegurar la comparación
        user_df = output_df[output_df['user_id'].astype(str) == params.user_id]

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
            Usuario=params.user_id,
            Dinero_gastado='${:.2f} USD'.format(money_spent),
            Porcentaje_recomendacion=pct_recommended_str,
            Cantidad_items=total_items,
        )

    except Exception as e:
        # Manejar cualquier excepción y devolver una respuesta de error
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# 76561197971591953	Counter-Strike
@app.get("/user-data-stats/{user_id:str}", response_model=UserStats)
def get_user_stats_endpoint(stats: UserStats = Depends(get_user_stats)):
    """
    Obtiene estadísticas del usuario através del Id Steam.
    Supuesto el usuario  genera cambios de id_usuario constamente pero el id_steam se mantiene.

    Parameters:
    - user_id (str): Identificador Steam del usuario del cual se busca la información chidvd.

    Returns:
    - List[DeveloperResult]: Lista con información sobre la cantidad de dinero gastado por el 
    usuario, el porcentaje de recomendación en base a recommend y cantidad de items.
    """
    return stats


# Definir modelo Pydantic para parámetros de género
class GenreParams(BaseModel):
    genre: str

def get_user_playtime_by_genre(params: GenreParams = Depends()):
    try:
        # Convertir el género ingresado a minúsculas y sin espacios
        normalized_genre = params.genre.lower().replace(" ", "")

        # Filtrar el DataFrame para el género específico
        genre_df = output_df[output_df['genres_str'].str.lower().str.replace(" ", "") == normalized_genre].copy()

        # Verificar si hay datos para el género
        if genre_df.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el género {params.genre}.")

        # Encontrar el usuario con más horas jugadas
        max_playtime_user = genre_df.loc[genre_df['playtime_forever'].idxmax()]['user_id']

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
    """
    Obtiene estadísticas de horas jugadas por género de juego a través de un endpoint API.

    Parameters:
    - result (dict): Resultado obtenido mediante la función get_user_playtime_by_genre = simulation.

    Returns:
    - dict: Diccionario con información sobre el usuario con más horas jugadas para el género y la acumulación de horas jugadas por año.
    """
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
    """
    Obtiene estadísticas de los principales desarrolladores por la cantidad de juegos recomendados para un año específico a través de un endpoint API.

    Parameters:
    - result (List[dict]): Resultado obtenido mediante la función get_top_developers_by_year : Rango 1998 entre 2017.

    Returns:
    - List[dict]: Lista de diccionarios con información sobre los principales desarrolladores y la cantidad de juegos recomendados.
    """
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
    """
    Obtiene estadísticas del análisis de reseñas para un desarrollador específico a través de un endpoint API.

    Parameters:
    - result (Union[dict, List[dict]]): Resultado obtenido mediante la función get_developer_reviews_analysis: Valve.

    Returns:
    - Union[dict, List[dict]]: Diccionario o lista de diccionarios con información sobre las reseñas positivas, negativas y neutrales del desarrollador.
    """
    return result

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        for col in self.columns:
            le = self.label_encoders[col]
            X.loc[:, col] = le.transform(X[col])
        return X

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_columns = []
        self.ordinal_columns = []
        self.categorical_columns = []

    def fit(self, X, y=None):
        # Obtener las columnas numéricas, ordinales y categóricas
        self.numeric_columns = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
        self.ordinal_columns = ['recommend']  # Puedes añadir aquí tus columnas ordinales
        self.categorical_columns = X.select_dtypes(include=['object']).columns
        return self

    def transform(self, X):
        # Crear una nueva variable excluyendo las columnas ordinales
        categorical_col_excluded_ordinal = [col for col in self.categorical_columns if col not in self.ordinal_columns]
        numeric_col_excluded_ordinal = [col for col in self.numeric_columns if col not in self.ordinal_columns]

        # Convertir las columnas categóricas a tipo str
        X.loc[:, categorical_col_excluded_ordinal] = X[categorical_col_excluded_ordinal].astype(str)

        # Definir las transformaciones para las columnas numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Definir las transformaciones para las columnas ordinales
        ordinal_transformer = MultiColumnLabelEncoder(columns=self.ordinal_columns)

        # Aplicar las transformaciones
        transformed_data = numeric_transformer.fit_transform(X[numeric_col_excluded_ordinal])
        transformed_data = np.concatenate([transformed_data, ordinal_transformer.fit_transform(X[self.ordinal_columns])], axis=1)

        # Obtener los nombres de las columnas después de la transformación
        numeric_feature_names = numeric_transformer.named_steps['scaler'].get_feature_names_out(input_features=numeric_col_excluded_ordinal)
        ordinal_feature_names = self.ordinal_columns
        column_names = np.concatenate([numeric_feature_names, ordinal_feature_names])

        return transformed_data, column_names
    
def process_dataframe(input_df: pd.DataFrame, columns_to_drop: list) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Create an instance of the class
    label_processor = MultiColumnLabelEncoder() 
    custom_processor = CustomPreprocessor()

    # Drop specified columns
    processed_df = input_df.drop(columns=columns_to_drop)

    # Apply the pipeline to the DataFrame
    transformed_data, column_names = custom_processor.fit_transform(processed_df)

    return processed_df, pd.DataFrame(transformed_data, columns=column_names)

# Example usage
columns_to_drop = ['app_name', 'developer', 'user_id','user_reviews_item_id', 'users_items_item_id','sentiment_more_less']
processed_df, transformed_df = process_dataframe(output_df, columns_to_drop)


def calculate_similarity_matrix(df):
    # Seleccionar solo las columnas numéricas para el cálculo de similitud
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]

    # Calcular la matriz de similitud coseno
    similarity_matrix = cosine_similarity(df_numeric)

    return similarity_matrix

# Ejemplo de uso con tu DataFrame
output_df_ml = pd.concat([output_df.loc[:, ['users_items_item_id', 'app_name']], transformed_df], axis=1)
modelo_item_id = output_df_ml.groupby(["users_items_item_id", "app_name"]).sum().reset_index()
# Llamar a la función con el DataFrame con índices personalizados
#similarity_matrix = calculate_similarity_matrix(modelo_item_id)

similarity_matrix = cosine_similarity(modelo_item_id.select_dtypes(include=[np.number]))


class SimilarGamesRequest(BaseModel):
    reference_item_id: int
    num_similar_games: int

class SimilarGamesResponse(BaseModel):
    similar_games: List[str]

@app.post("/similar_item_id", response_model=SimilarGamesResponse)
def get_similar_games(request: SimilarGamesRequest):
    """
    Obtiene juegos similares basados en el índice de referencia y el número de juegos similares solicitados.

    Parameters:
    - request (SimilarGamesRequest): Parámetros de entrada que incluyen el índice de referencia y el número de juegos similares.
    - "reference_index": 10,      # hace referencia al Juego Counter-Strike
    - "num_similar_games": 5      # puedes solicitar en número diferentes juegos del sistema de recomendación.
    Returns:
    - SimilarGamesResponse: Respuesta que contiene la lista de nombres de juegos similares.
    
    Raises:
    - HTTPException(404): Si el índice de referencia no es válido.
    - HTTPException(500): Si ocurre un error interno durante la ejecución.
    """
    try:
        # Buscar el índice correspondiente al users_items_item_id en el DataFrame
        reference_index = modelo_item_id[modelo_item_id['users_items_item_id'] == request.reference_item_id].index[0]
        
        # Obtener las puntuaciones de similitud para el juego de referencia
        similarity_scores = similarity_matrix[reference_index]

        # Obtener los índices ordenados por similitud
        sorted_indices = similarity_scores.argsort()[::-1]

        # Obtener los índices de juegos similares (excluyendo el juego de referencia)
        similar_games_indices = sorted_indices[1:request.num_similar_games + 1]

        # Obtener los nombres de juegos similares
        similar_games = modelo_item_id['app_name'].iloc[similar_games_indices]

        return SimilarGamesResponse(similar_games=similar_games.tolist())

    except IndexError:
        # Manejar el caso en el que no se encuentra el juego de referencia
        raise HTTPException(status_code=404, detail=f"Juego de referencia con ID {request.reference_item_id} no encontrado")



def calculate_similarity_matrix(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    similarity_matrix = cosine_similarity(df_numeric)
    return similarity_matrix

output_df_ml_2 = pd.concat([output_df.loc[:, ['user_id', 'app_name']], transformed_df], axis=1)
modelo_id_steam = output_df_ml_2.groupby(["user_id", "app_name"]).sum().reset_index()
similarity_matrix_steam = calculate_similarity_matrix(modelo_id_steam)

class SimilarGamesRequestNewName(BaseModel):
    reference_index: Union[str, int]
    num_similar_games: int

class SimilarGamesResponseNewName(BaseModel):
    similar_games: List[str]

def get_similar_games_new_name(request: SimilarGamesRequestNewName, similarity_matrix: np.ndarray, modelo_id_steam: pd.DataFrame) -> SimilarGamesResponseNewName:
    str_reference_index = str(request.reference_index)
    matching_rows = modelo_id_steam[
        modelo_id_steam['user_id'].astype(str).str.lower() == str_reference_index.lower()
    ]

    if matching_rows.empty:
        return SimilarGamesResponseNewName(similar_games=[])

    reference_index = matching_rows.index[0]
    similarity_scores = similarity_matrix[reference_index]
    sorted_indices = similarity_scores.argsort()[::-1]
    similar_games_indices = sorted_indices[1:request.num_similar_games + 1]
    similar_games = modelo_id_steam['app_name'].iloc[similar_games_indices]

    return SimilarGamesResponseNewName(similar_games=similar_games.to_list())

@app.post("/get_similar_games")
async def get_similar_games_endpoint(request: SimilarGamesRequestNewName):
    """
    Obtiene juegos similares basados en el índice o User ID de referencia y el número de juegos similares solicitados.

    Parameters:
    - request (SimilarGamesRequest): Parámetros de entrada que incluyen el índice o User ID de referencia y el número de juegos similares.
    - "reference_index": "DarthRhys",      # hace referencia al Identificador Steam de cada usuario
    - "num_similar_games": 5                       # puedes solicitar en número diferentes juegos del sistema de recomendación
    Returns:
    - SimilarGamesResponse: Respuesta que contiene la lista de nombres de juegos similares.
    
    Raises:
    - HTTPException(404): Si el User ID de referencia no se encuentra en la base de datos.
    - HTTPException(500): Si ocurre un error interno durante la ejecución.
    """
    try:
        result = get_similar_games_new_name(request, similarity_matrix_steam, modelo_id_steam)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
