from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


app = FastAPI()

def cargar_dataframes():
    df_games = pd.read_parquet(r'D:\1er Proyecto\datasets_finales\games.parquet')
    df_review = pd.read_parquet(r'D:\1er Proyecto\datasets_finales\reviews.parquet')
    df_item = pd.read_parquet(r'D:\1er Proyecto\datasets_finales\user_items.parquet')
    return df_games, df_review, df_item

# Cargar los DataFrames y inicializar el modelo al inicio de la aplicación
df_games, df_review, df_item = cargar_dataframes()

@app.get("/developer/{empresa}")
def developer(empresa: str):
    df = df_games.copy()

    # Convertir el nombre de la empresa a minúsculas para hacer la búsqueda insensible a mayúsculas y minúsculas
    empresa = empresa.lower()
    
    # Filtrar el DataFrame por la empresa desarrolladora que contiene el nombre especificado
    df_empresa = df[df['developer'].str.lower().str.contains(empresa)]
    
    # Verificar si hay datos para la empresa desarrolladora especificada
    if df_empresa.empty:
        return f"No hay datos disponibles para la empresa desarrolladora que contiene '{empresa}'."
        
    # Agrupar por año
    grouped = df_empresa.groupby('year')
    
    # Calcular cantidad de items y porcentaje de contenido Free por año
    resultados = []
    for year, group in grouped:
        total_items = len(group)
        free_items = len(group[group['price'] == 0])
        porcentaje_free = (free_items / total_items) * 100 if total_items > 0 else 0
        resultados.append({
            'Año': year,
            'Cantidad de Items': total_items,
            'Porcentaje Free': porcentaje_free
        })
    
    # Ordenar los resultados por año
    resultados.sort(key=lambda x: x['Año'])
    
    # Construir el texto con los resultados año por año
    texto = ""
    for resultado in resultados:
        texto += f"Año: {resultado['Año']}, Cantidad de Items: {resultado['Cantidad de Items']}, Porcentaje Free: {resultado['Porcentaje Free']:.2f}%\n"

    return texto

@app.get("/userdata/{id}")
def userdata(user_name) -> dict:
    items = df_item.copy()
    games = df_games.copy()
    reviews = df_review.copy()

    #Identificación usuario
    user_name = str(user_name)
    #Obtener cantidad de items
    user_items = items[items['user_id'] == user_name] #conexión de items de un usuario por su id
    num_items = len(user_items['item_id'].unique())
    #Obtención de dinero gastado
    user_items_prices = user_items.merge(games, left_on='item_id', right_on='item_id', how='inner')
    total_gastado = user_items_prices['price'].sum()
    #Porcentaje de recomendación
    user_reviews = reviews[reviews['user_id'] == user_name] #conexión de reviews de un usuario por su id
    total_reviews = len(user_reviews)
    positive_reviews = user_reviews[user_reviews['recommend']==1]
    num_positive_reviews = len(positive_reviews)
    if total_reviews != 0:
        porcentaje_positive_reviews = (num_positive_reviews / total_reviews) * 100
    else:
        porcentaje_positive_reviews = 0
    # Crear diccionario con la información
    user_data = {
        'Total gastado (USD)': total_gastado,
        'Porcentaje de recomendación positiva': porcentaje_positive_reviews,
        'Cantidad de juegos del usuario': num_items
    }
    return user_data

@app.get("/UserForGenre/{genero}")
def UserForGenre(genero: str):
  df_genres = df_games.copy()
  df_genres['genres'] = df_genres['genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
  df_genres.drop(['name', 'tags','specs','price','developer'], axis=1)
  
  genero = '[' + genero + ']'
  
  # Filtrar el DataFrame para dejar solo los juegos que contengan el género especificado
  df_genre = df_genres[df_genres['genres'].apply(lambda x: genero in x if isinstance(x, list) else False)]

  # Filtrar los usuarios que poseen los juegos del género específico
  df_user_aggregated = df_item[df_item['item_id'].isin(df_genre['item_id'])]
    
  # Merge para concatenar el año de df_genre a df_user_aggregated basado en el item_id
  df_user_aggregated = df_user_aggregated.merge(df_genre[['item_id', 'year']], on='item_id', how='left')
    
  # Calcular la suma de las horas jugadas por cada usuario a los juegos del género específico
  user_hours_per_game = df_user_aggregated.groupby('user_id')['playtime_forever'].sum()
    
  # Obtener al usuario con más horas jugadas
  user_most_hours_user_id = user_hours_per_game.idxmax()
    
  # Filtrar las horas jugadas por el usuario con más horas jugadas
  user_most_hours_df = df_user_aggregated[df_user_aggregated['user_id'] == user_most_hours_user_id]
    
  # Calcular la cantidad de horas jugadas por año del usuario con más horas jugadas considerando el año de publicación del juego
  hours_per_year = user_most_hours_df.groupby('year')['playtime_forever'].sum().reset_index()
    
  # Formatear el resultado en el formato especificado
  result = {
        "Usuario con más horas jugadas para " + genero: user_most_hours_user_id,
        "Horas jugadas": [{"Año": int(row['year']), "Horas": int(row['playtime_forever'])} for index, row in hours_per_year.iterrows()]
  }
    
  return result

@app.get("/best_developer_year/{year}")
def best_developer_year(año: int):
    games_df = df_games.copy()
    reviews_df = df_review.copy()

    games_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    reviews_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    # Filtrar juegos por el año especificado
    games_filtered = games_df[games_df['year'] == año]
    # Merge juegos y reviews basado en 'id'
    merged_df = pd.merge(games_filtered, reviews_df, left_on='item_id', right_on='item_id')
    # Contar recomendaciones por desarrollador
    developer_counts = merged_df[merged_df['recommend'] == 1]['developer'].value_counts().reset_index()
    developer_counts.columns = ['Developer', 'Recommendations']
    # Ordenar por número de recomendaciones y obtener los 3 primeros
    sorted_developers = developer_counts.sort_values(by='Recommendations', ascending=False)
    top_developers = sorted_developers.head(3)
    # Formatear el resultado como una lista de diccionarios
    result = [{"Puesto {}: {}".format(i+1, row['Developer']): row['Recommendations']} for i, row in top_developers.iterrows()]
    return result


@app.get("/developer_reviews_analysis/{desarrolladora}")
def developer_reviews_analysis(desarrolladora: str):
    games_df = df_games.copy()
    reviews_df = df_review.copy()

    games_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    reviews_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    # Merge juegos y reviews basado en 'id'
    merged_df = pd.merge(games_df, reviews_df, left_on='item_id', right_on='item_id')
    # Filtrar por la desarrolladora especificada
    developer_filtered = merged_df[merged_df['developer'].str.strip().str.lower() == desarrolladora.lower()]
    # Contar registros de reseñas categorizadas como análisis positivo o negativo
    positive_count = len(developer_filtered[developer_filtered['sentiment_analysis'] == 2])
    negative_count = len(developer_filtered[developer_filtered['sentiment_analysis'] == 0])
    # Crear el diccionario de retorno
    result = {desarrolladora: {'Negative': negative_count, 'Positive': positive_count}}
    return result



@app.get('/recomendacion_juego/{juego_id}')
def recomendacion_juego(product_id: int):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_games['genres'])
    
    # Configurar el modelo "Vecino más cercano"
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    
    # Obtener el índice del juego actual
    juego_index = df_games[df_games['item_id'] == product_id].index[0]
    
    # Encontrar los índices de los 5 juegos más similares
    distances, indices = nn.kneighbors(tfidf_matrix[juego_index], n_neighbors=6)
    
    # Excluir el propio juego de los resultados
    similar_indices = indices.flatten()[1:]
    
    # Obtener los nombres de los juegos similares
    juegos_similares = df_games.iloc[similar_indices]['name'].values
    
    # Obtener el nombre del juego actual
    juego_actual = df_games.loc[df_games['item_id'] == product_id, 'name'].iloc[0]
    
    # Crear el resultado en el formato especificado
    resultado = {
        "Juego actual": juego_actual,
        "Juegos recomendados similares": list(juegos_similares)
    }
    
    return resultado

#print(developer('valve'))

#print(userdata('imsodonionringsrightnow'))

#print(UserForGenre('Action'))

#print(best_developer_year(2015))

#print(developer_reviews_analysis('valve'))

#print(recomendacion_juego(99910))