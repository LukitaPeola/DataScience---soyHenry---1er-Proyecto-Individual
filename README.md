Primer proyecto individual de Ian Brandan para SoyHenry.

Bienvenidos/as a este primer proyecto enfocado al Data Analysis de Steam, el cual propone combinar MachineLearning con procesos como ETL, EDA y otros.

Para esto, he sido proporcionado con 3 archivos .json

-australian_user_reviews -australian_users_items -output_steam_games

Los cuales vienen algo corruptos y hay que encontrar la forma de ingresar a los datos. De esa manera se consiguen nuestros ETLs, donde se realiza una ardua limpieza de los datos y se exportan al final solamente los que vamos a necesitar. Seguido de eso, creé un notebook EDA donde me dedico a explorar los datos obtenidos y observo estadísticas de lo que contiene.

Luego, me enfoqué en la tarea de usar el framework FastAPI, creé las funciones de consulta solicitadas y realicé la conexión para que funcionaran de forma dinámica. Dichas funciones eran:

-def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. 
-def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items. 
-def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento. 
-def best_developer_year( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos).
-def developer_reviews_analysis( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo. (En este último, decidí usar el modelo del vecino más cercano para encontrar juegos similares).

Terminado por completo el trabajo se realiza el deploy en render y puede ser consultado en el siguiente link:

DEPLOY EN RENDER: https://datascience-soyhenry-1er-proyecto.onrender.com/docs

Video explicativo en Youtube: https://youtu.be/djoTMMjlDuU
