import pandas as pd
import os

ruta_actual = os.getcwd()

ruta_games = os.path.join(ruta_actual, 'datasets', 'games.parquet')
df_games = pd.read_parquet(ruta_games)

ruta_review = os.path.join(ruta_actual, 'ModeloML', 'prediccion_review.parquet')
df_review = pd.read_parquet(ruta_review)

ruta_item = os.path.join(ruta_actual, 'datasets', 'items.parquet')
df_item = pd.read_parquet(ruta_item)

df_games.loc[df_games['id'].astype(str).str.contains('None'), 'id'] = None
df_games.dropna(subset=['id'], inplace=True)
df_games['id'] = df_games['id'].astype(int)
df_games['developer'] = df_games['developer'].str.strip().str.lower()

df_item['user_id'] = df_item['user_id'].str.strip()
df_item.loc[df_item['item_id'].astype(str).str.contains('None'), 'item_id'] = None
df_item.dropna(subset=['item_id'], inplace=True)
df_item['item_id'] = df_item['item_id'].astype(int)

df_review['user_id'] = df_review['user_id'].str.strip()
df_review['item_id'] = df_review['item_id'].str.strip()
df_review.loc[df_review['item_id'].astype(str).str.contains('None'), 'item_id'] = None
df_review.dropna(subset=['item_id'], inplace=True)
df_review['item_id'] = df_review['item_id'].astype(int)


def developer(empresa:str, df=df_games):
    # Filtrar el DataFrame por la empresa desarrolladora especificada
    df_empresa = df[df['developer'] == empresa.lower()]
    # Agrupar por año
    grouped = df_empresa.groupby('anio')
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
    # Convertir resultados a DataFrame para una mejor visualización
    return resultados


def userdata(user_name, df_games=df_games, df_items=df_item, df_reviews=df_review) -> dict:
    #Identificación usuario
    user_name = str(user_name)
    #Obtener cantidad de items
    user_items = df_items[df_items['user_id'] == user_name] #conexión de items de un usuario por su id
    num_items = len(user_items['item_id'].unique())
    #Obtención de dinero gastado
    user_items_prices = user_items.merge(df_games, left_on='item_id', right_on='id', how='inner')
    total_gastado = user_items_prices['price'].sum()
    #Porcentaje de recomendación
    user_reviews = df_reviews[df_reviews['user_id'] == user_name] #conexión de reviews de un usuario por su id
    total_reviews = len(user_reviews)
    positive_reviews = user_reviews[user_reviews['recommend']==2]
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


def UserForGenre(genero: str, games_df=df_games, items_df=df_item):
    genre_games = games_df[games_df[genero] == 1]
    # Unir con items_df para obtener las horas jugadas y otros detalles
    genre_items = pd.merge(items_df, genre_games, left_on='item_id', right_on='id')
    # Filtrar las filas con playtime_forever menor o igual a 8760
    genre_items.loc[genre_items['playtime_forever'].astype(str).str.contains('None'), 'playtime_forever'] = None
    genre_items.dropna(subset=['playtime_forever'], inplace=True)
    genre_items['playtime_forever'] = genre_items['playtime_forever'].astype(int)
    filtered_items = genre_items[genre_items['playtime_forever'] <= 8760]
    # Agrupar por usuario y año de lanzamiento para calcular horas jugadas por año
    grouped_items = filtered_items.groupby(['user_id', 'anio'])['playtime_forever'].sum().reset_index()
    # Limitar las horas jugadas a un máximo de 8760 horas por año (aprox. la cantidad de horas en un año)
    grouped_items['playtime_forever'] = grouped_items['playtime_forever'].clip(upper=8760)
    # Encontrar el usuario con más horas jugadas para el género dado
    top_user = grouped_items.groupby('user_id')['playtime_forever'].sum().idxmax()
    top_user_hours = grouped_items[grouped_items['user_id'] == top_user].copy()
    # Agrupar por año para calcular las horas jugadas totales por año
    acumulacion_horas_anio = top_user_hours.groupby('anio')['playtime_forever'].sum().reset_index()
    # Construir la lista de horas jugadas por año
    horas_jugadas_por_anio = []
    for anio, horas in zip(acumulacion_horas_anio['anio'], acumulacion_horas_anio['playtime_forever']):
        horas_jugadas_por_anio.append({'anio': anio, 'horas': horas})
    # Crear el diccionario de resultados con las horas jugadas
    resultado = {
        'Usuario con más horas jugadas para género {}:'.format(genero): top_user,
        'Horas jugadas': horas_jugadas_por_anio
    }
    return resultado


def best_developer_year(año: int, games_df=df_games, reviews_df=df_review):
    games_df.sort_values(by='id', ascending=False, inplace=True, ignore_index=True)
    reviews_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    # Filtrar juegos por el año especificado
    games_filtered = games_df[games_df['anio'] == año]
    # Merge juegos y reviews basado en 'id'
    merged_df = pd.merge(games_filtered, reviews_df, left_on='id', right_on='item_id')
    # Contar recomendaciones por desarrollador
    developer_counts = merged_df[merged_df['recommend'] == 2]['developer'].value_counts().reset_index()
    developer_counts.columns = ['Developer', 'Recommendations']
    # Ordenar por número de recomendaciones y obtener los 3 primeros
    sorted_developers = developer_counts.sort_values(by='Recommendations', ascending=False)
    top_developers = sorted_developers.head(3)
    # Formatear el resultado como una lista de diccionarios
    result = [{"Puesto {}: {}".format(i+1, row['Developer']): row['Recommendations']} for i, row in top_developers.iterrows()]
    return result


def developer_reviews_analysis(desarrolladora: str, games_df=df_games, reviews_df=df_review):
    games_df.sort_values(by='id', ascending=False, inplace=True, ignore_index=True)
    reviews_df.sort_values(by='item_id', ascending=False, inplace=True, ignore_index=True)
    # Merge juegos y reviews basado en 'id'
    merged_df = pd.merge(games_df, reviews_df, left_on='id', right_on='item_id')
    # Filtrar por la desarrolladora especificada
    developer_filtered = merged_df[merged_df['developer'].str.strip().str.lower() == desarrolladora.lower()]
    # Contar registros de reseñas categorizadas como análisis positivo o negativo
    positive_count = len(developer_filtered[developer_filtered['sentiment_analysis'] == 2])
    negative_count = len(developer_filtered[developer_filtered['sentiment_analysis'] == 0])
    # Crear el diccionario de retorno
    result = {desarrolladora: {'Negative': negative_count, 'Positive': positive_count}}
    return result