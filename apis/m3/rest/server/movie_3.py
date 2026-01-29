from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/movie_3/movie_3.sqlite')
cursor = conn.cursor()

# Endpoint to get the description of a film based on its title
@app.get("/v1/movie_3/film_description_by_title", operation_id="get_film_description", summary="Retrieves the detailed description of a specific film by its title. The operation searches for the film with the provided title and returns its description.")
async def get_film_description(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT description FROM film WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of films with a rental duration greater than a specified value
@app.get("/v1/movie_3/film_count_by_rental_duration", operation_id="get_film_count_by_rental_duration", summary="Retrieves the total number of films that have a rental duration exceeding the provided value in days.")
async def get_film_count_by_rental_duration(rental_duration: int = Query(..., description="Rental duration in days")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE rental_duration > ?", (rental_duration,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the titles of films based on release year and rental rate
@app.get("/v1/movie_3/film_titles_by_release_year_and_rental_rate", operation_id="get_film_titles_by_release_year_and_rental_rate", summary="Retrieves the titles of films that were released in a specific year and have a particular rental rate. The operation requires the release year and rental rate as input parameters to filter the results.")
async def get_film_titles_by_release_year_and_rental_rate(release_year: int = Query(..., description="Release year of the film"), rental_rate: float = Query(..., description="Rental rate of the film")):
    cursor.execute("SELECT title FROM film WHERE release_year = ? AND rental_rate = ?", (release_year, rental_rate))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the title of the longest film
@app.get("/v1/movie_3/longest_film_title", operation_id="get_longest_film_title", summary="Retrieves the title of the film with the longest duration. The operation returns the title of the film that has the highest length value in the database.")
async def get_longest_film_title():
    cursor.execute("SELECT title FROM film ORDER BY length DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the title of the film with the highest replacement cost among specified titles
@app.get("/v1/movie_3/highest_replacement_cost_film_title", operation_id="get_highest_replacement_cost_film_title", summary="Retrieves the title of the film with the highest replacement cost from a list of specified film titles.")
async def get_highest_replacement_cost_film_title(title1: str = Query(..., description="First film title"), title2: str = Query(..., description="Second film title")):
    cursor.execute("SELECT title FROM film WHERE title IN (?, ?) ORDER BY replacement_cost DESC LIMIT 1", (title1, title2))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of films with a specific rating and release year
@app.get("/v1/movie_3/film_count_by_rating_and_release_year", operation_id="get_film_count_by_rating_and_release_year", summary="Retrieves the total number of films that match a given rating and release year. The operation considers both the rating and the year of release to provide an accurate count of films that meet the specified criteria.")
async def get_film_count_by_rating_and_release_year(rating: str = Query(..., description="Rating of the film"), release_year: int = Query(..., description="Release year of the film")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE rating = ? AND release_year = ?", (rating, release_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of films with a specific rental rate and special feature
@app.get("/v1/movie_3/film_count_by_rental_rate_and_special_feature", operation_id="get_film_count_by_rental_rate_and_special_feature", summary="Retrieves the total number of films that have a specified rental rate and a particular special feature. The rental rate and special feature are provided as input parameters.")
async def get_film_count_by_rental_rate_and_special_feature(rental_rate: float = Query(..., description="Rental rate of the film"), special_feature: str = Query(..., description="Special feature of the film")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE rental_rate = ? AND special_features = ?", (rental_rate, special_feature))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the titles of films with more than 2 special features
@app.get("/v1/movie_3/film_titles_with_more_than_two_special_features", operation_id="get_film_titles_with_more_than_two_special_features", summary="Retrieves a list of film titles that have more than two special features. The list is ordered by the number of special features, with those having more than two appearing first.")
async def get_film_titles_with_more_than_two_special_features():
    cursor.execute("SELECT title FROM ( SELECT title, COUNT(special_features) AS num FROM film GROUP BY title ) AS T ORDER BY T.num > 2")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the email of a staff member based on their first and last name
@app.get("/v1/movie_3/staff_email_by_name", operation_id="get_staff_email_by_name", summary="Retrieves the email address of a staff member using their first and last names. The operation searches for a staff member with the provided names and returns their email if found.")
async def get_staff_email_by_name(first_name: str = Query(..., description="First name of the staff member"), last_name: str = Query(..., description="Last name of the staff member")):
    cursor.execute("SELECT email FROM staff WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"email": []}
    return {"email": result[0]}

# Endpoint to get the names of active staff members
@app.get("/v1/movie_3/active_staff_names", operation_id="get_active_staff_names", summary="Retrieves the first and last names of staff members who are currently active. The active status of the staff member is provided as an input parameter to filter the results.")
async def get_active_staff_names(active: int = Query(..., description="Active status of the staff member (1 for active, 0 for inactive)")):
    cursor.execute("SELECT first_name, last_name FROM staff WHERE active = ?", (active,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get distinct release years of films with the highest replacement cost
@app.get("/v1/movie_3/distinct_release_years_highest_replacement_cost", operation_id="get_distinct_release_years", summary="Retrieves the unique release years of films that have the highest replacement cost. This operation provides a list of years in which the most expensive films were released, offering insights into the distribution of high-cost productions over time.")
async def get_distinct_release_years():
    cursor.execute("SELECT DISTINCT release_year FROM film WHERE replacement_cost = ( SELECT MAX(replacement_cost) FROM film )")
    result = cursor.fetchall()
    if not result:
        return {"release_years": []}
    return {"release_years": [row[0] for row in result]}

# Endpoint to get titles of films with the highest replacement cost, limited by a specified number
@app.get("/v1/movie_3/titles_highest_replacement_cost", operation_id="get_titles_highest_replacement_cost", summary="Retrieves the titles of films with the highest replacement cost, up to a specified limit. The operation returns the titles of films that have the maximum replacement cost, providing a concise list of the most expensive films to replace. The limit parameter allows you to control the number of titles returned, enabling you to customize the size of the result set.")
async def get_titles_highest_replacement_cost(limit: int = Query(..., description="Number of titles to return")):
    cursor.execute("SELECT title FROM film WHERE replacement_cost = ( SELECT MAX(replacement_cost) FROM film ) LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the language name of a film by its title
@app.get("/v1/movie_3/language_name_by_film_title", operation_id="get_language_name_by_film_title", summary="Retrieves the name of the language used in a film, based on the provided film title. The operation searches for the film by its title and returns the corresponding language name.")
async def get_language_name_by_film_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.name FROM film AS T1 INNER JOIN language AS T2 ON T1.language_id = T2.language_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"language_name": []}
    return {"language_name": result[0]}

# Endpoint to get the count of films in a specific language
@app.get("/v1/movie_3/count_films_by_language", operation_id="get_count_films_by_language", summary="Retrieves the total number of films available in a specified language. The operation uses the provided language name to filter the films and calculate the count.")
async def get_count_films_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN language AS T2 ON T1.language_id = T2.language_id WHERE T2.name = ?", (language_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get film titles by actor's first and last name
@app.get("/v1/movie_3/film_titles_by_actor_name", operation_id="get_film_titles_by_actor_name", summary="Retrieves a list of film titles associated with the specified actor. The operation requires the actor's first and last names as input parameters to filter the results accordingly.")
async def get_film_titles_by_actor_name(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T2.title FROM film_actor AS T1 INNER JOIN film AS T2 ON T1.film_id = T2.film_id INNER JOIN actor AS T3 ON T1.actor_id = T3.actor_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of actors in a specific film
@app.get("/v1/movie_3/count_actors_by_film_title", operation_id="get_count_actors_by_film_title", summary="Retrieves the total number of actors associated with a specific film, based on the provided film title.")
async def get_count_actors_by_film_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT COUNT(T1.actor_id) FROM film_actor AS T1 INNER JOIN film AS T2 ON T1.film_id = T2.film_id WHERE T2.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get actor names by film title
@app.get("/v1/movie_3/actor_names_by_film_title", operation_id="get_actor_names_by_film_title", summary="Retrieves the first and last names of all actors who have appeared in a film with the specified title. The film title is provided as an input parameter.")
async def get_actor_names_by_film_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of films by actor name and release year
@app.get("/v1/movie_3/count_films_by_actor_name_and_release_year", operation_id="get_count_films_by_actor_name_and_release_year", summary="Retrieves the total number of films in which a specific actor has appeared in a given release year. The operation requires the actor's first and last names, as well as the release year of the films.")
async def get_count_films_by_actor_name_and_release_year(release_year: int = Query(..., description="Release year of the film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(T2.film_id) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.release_year = ? AND T1.first_name = ? AND T1.last_name = ?", (release_year, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the film title with the highest replacement cost for a specific actor
@app.get("/v1/movie_3/film_title_highest_replacement_cost_by_actor", operation_id="get_film_title_highest_replacement_cost_by_actor", summary="Retrieves the title of the film with the highest replacement cost associated with a specific actor. The actor is identified by their first and last names, which are provided as input parameters. The operation returns the title of the film that has the highest replacement cost among all films the actor has appeared in.")
async def get_film_title_highest_replacement_cost_by_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? ORDER BY T3.replacement_cost DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the actor with the highest replacement cost film
@app.get("/v1/movie_3/actor_highest_replacement_cost_film", operation_id="get_actor_highest_replacement_cost_film", summary="Retrieves the first name and last name of the actor who has starred in the film with the highest replacement cost. The replacement cost is a measure of the film's value, and the actor with the highest value film is returned.")
async def get_actor_highest_replacement_cost_film():
    cursor.execute("SELECT first_name, last_name FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id ORDER BY T3.replacement_cost DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"actor": []}
    return {"actor": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of films for an actor with a specific language
@app.get("/v1/movie_3/count_films_actor_language", operation_id="get_count_films_actor_language", summary="Retrieves the total number of films associated with a specific actor, filtered by a given language. The operation requires the actor's first and last names, as well as the desired language, to accurately determine the count.")
async def get_count_films_actor_language(language: str = Query(..., description="Language of the film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(T3.film_id) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id INNER JOIN language AS T4 ON T3.language_id = T4.language_id WHERE T4.name = ? AND T1.first_name = ? AND T1.last_name = ?", (language, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the longest film title for a specific actor
@app.get("/v1/movie_3/longest_film_title_actor", operation_id="get_longest_film_title_actor", summary="Retrieves the title of the longest film in which a specific actor has appeared. The actor is identified by their first and last names.")
async def get_longest_film_title_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? ORDER BY T3.length DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get film titles by category
@app.get("/v1/movie_3/film_titles_by_category", operation_id="get_film_titles_by_category", summary="Retrieves a list of film titles that belong to the specified category. The category is provided as an input parameter.")
async def get_film_titles_by_category(category: str = Query(..., description="Category of the film")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of films by category
@app.get("/v1/movie_3/count_films_by_category", operation_id="get_count_films_by_category", summary="Retrieves the total number of films belonging to a specified category. The category is provided as an input parameter.")
async def get_count_films_by_category(category: str = Query(..., description="Category of the film")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id WHERE T2.name = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get film titles by category and rental rate
@app.get("/v1/movie_3/film_titles_by_category_rental_rate", operation_id="get_film_titles_by_category_rental_rate", summary="Retrieves a list of film titles that belong to a specified category and have a particular rental rate. The category and rental rate are provided as input parameters.")
async def get_film_titles_by_category_rental_rate(category: str = Query(..., description="Category of the film"), rental_rate: float = Query(..., description="Rental rate of the film")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ? AND T1.rental_rate = ?", (category, rental_rate))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of rentals for a specific customer
@app.get("/v1/movie_3/count_rentals_customer", operation_id="get_count_rentals_customer", summary="Retrieves the total number of rentals associated with a customer, identified by their first and last names.")
async def get_count_rentals_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T2.rental_id) FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get film titles rented by a specific customer
@app.get("/v1/movie_3/film_titles_rented_customer", operation_id="get_film_titles_rented_customer", summary="Retrieves the titles of films rented by a customer identified by their first and last names. The operation uses the provided customer names to filter the relevant rental records and returns the corresponding film titles.")
async def get_film_titles_rented_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T4.title FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of customers who rented films released in a specific year
@app.get("/v1/movie_3/count_customers_rented_films_year", operation_id="get_count_customers_rented_films_year", summary="Retrieves the number of customers who rented films released in a specific year, filtered by the customer's first and last name.")
async def get_count_customers_rented_films_year(release_year: int = Query(..., description="Release year of the film"), first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T4.release_year = ? AND T1.first_name = ? AND T1.last_name = ?", (release_year, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most expensive film title rented by a specific customer
@app.get("/v1/movie_3/most_expensive_film_rented_customer", operation_id="get_most_expensive_film_rented_customer", summary="Retrieves the title of the most expensive film rented by a customer, based on the replacement cost. The operation requires the first and last name of the customer to identify the relevant rental records.")
async def get_most_expensive_film_rented_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T4.title FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T1.first_name = ? AND T1.last_name = ? ORDER BY T4.replacement_cost DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the customer who rented the most expensive film
@app.get("/v1/movie_3/customer_most_expensive_film", operation_id="get_customer_most_expensive_film", summary="Retrieves the first name and last name of the customer who rented the film with the highest replacement cost. This operation uses the replacement cost of films to determine the most expensive rental and identifies the associated customer.")
async def get_customer_most_expensive_film():
    cursor.execute("SELECT T1.first_name, T1.last_name FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id ORDER BY T4.replacement_cost DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of customers based on first name, last name, return month, and return year
@app.get("/v1/movie_3/customer_count_by_name_return_date", operation_id="get_customer_count", summary="Get the count of customers with a specific first name, last name, return month, and return year")
async def get_customer_count(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), return_month: str = Query(..., description="Return month in 'MM' format"), return_year: str = Query(..., description="Return year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ? AND STRFTIME('%m',T2.return_date) = ? AND STRFTIME('%Y', T2.return_date) = ?", (first_name, last_name, return_month, return_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer with the highest number of rentals
@app.get("/v1/movie_3/top_customer_by_rentals", operation_id="get_top_customer", summary="Retrieves the customer who has rented the most movies. The response includes the first and last name of the top customer.")
async def get_top_customer():
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T1.first_name, T1.last_name, COUNT(T2.rental_id) AS num FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id GROUP BY T1.first_name, T1.last_name ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of active customers who rented a specific film
@app.get("/v1/movie_3/active_customer_count_by_film", operation_id="get_active_customer_count", summary="Retrieves the number of active customers who have rented a specific film. The operation uses the provided active status and film title to filter the customer and film data, respectively. The result is a count of active customers who have rented the specified film.")
async def get_active_customer_count(active: int = Query(..., description="Active status of the customer (1 for active)"), film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T1.active = ? AND T4.title = ?", (active, film_title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most rented film
@app.get("/v1/movie_3/most_rented_film", operation_id="get_most_rented_film", summary="Retrieves the title of the film that has been rented the most times. This operation calculates the rental count for each film and returns the title of the film with the highest count.")
async def get_most_rented_film():
    cursor.execute("SELECT T.title FROM ( SELECT T1.title, COUNT(T3.rental_id) AS num FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id INNER JOIN rental AS T3 ON T2.inventory_id = T3.inventory_id GROUP BY T1.title ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"film": []}
    return {"film": result[0]}

# Endpoint to get the customer with the highest number of rentals among specified names
@app.get("/v1/movie_3/top_customer_by_rentals_among_names", operation_id="get_top_customer_among_names", summary="Retrieves the customer with the highest number of rentals from a specified pair of first and last names. The operation compares the rental counts of two customers and returns the one with the most rentals. The input parameters include the first and last names of both customers.")
async def get_top_customer_among_names(first_name_1: str = Query(..., description="First name of the first customer"), last_name_1: str = Query(..., description="Last name of the first customer"), first_name_2: str = Query(..., description="First name of the second customer"), last_name_2: str = Query(..., description="Last name of the second customer")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T1.first_name, T1.last_name, COUNT(T1.customer_id) AS num FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE (T1.first_name = ? AND T1.last_name = ?) OR (T1.first_name = ? AND T1.last_name = ?) GROUP BY T1.first_name, T1.last_name ) AS T ORDER BY T.num DESC LIMIT 1", (first_name_1, last_name_1, first_name_2, last_name_2))
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the film with the highest rental rate per duration for a specific actor
@app.get("/v1/movie_3/top_film_by_rental_rate_per_duration", operation_id="get_top_film_by_rental_rate_per_duration", summary="Retrieves the title of the film with the highest rental rate per duration for a specific actor. The actor is identified by their first and last names. The rental rate per duration is calculated as the rental rate divided by the rental duration.")
async def get_top_film_by_rental_rate_per_duration(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? ORDER BY T3.rental_rate / T3.rental_duration DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"film": []}
    return {"film": result[0]}

# Endpoint to get the average replacement cost of films in a specific category
@app.get("/v1/movie_3/average_replacement_cost_by_category", operation_id="get_average_replacement_cost", summary="Retrieves the average replacement cost of films belonging to a specified category. The category is identified by its name, which is provided as an input parameter.")
async def get_average_replacement_cost(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT AVG(T3.replacement_cost) FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id INNER JOIN film AS T3 ON T1.film_id = T3.film_id WHERE T2.name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_replacement_cost": []}
    return {"average_replacement_cost": result[0]}

# Endpoint to get the percentage of films in a specific category rented by a specific customer
@app.get("/v1/movie_3/percentage_films_by_category_and_customer", operation_id="get_percentage_films_by_category_and_customer", summary="Retrieves the percentage of films in a specified category that have been rented by a customer identified by their first and last names. The calculation is based on the total number of films in the category and the customer's rental history.")
async def get_percentage_films_by_category_and_customer(category_name: str = Query(..., description="Name of the category"), first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.film_id) FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id INNER JOIN inventory AS T4 ON T1.film_id = T4.film_id INNER JOIN customer AS T5 ON T4.store_id = T5.store_id INNER JOIN rental AS T6 ON T4.inventory_id = T6.inventory_id WHERE T5.first_name = ? AND T5.last_name = ?", (category_name, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average length of films for a specific actor
@app.get("/v1/movie_3/average_film_length_by_actor", operation_id="get_average_film_length", summary="Retrieves the average length of films in which a specific actor has appeared. The calculation is based on the actor's first and last names provided as input parameters.")
async def get_average_film_length(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT AVG(T3.length) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"average_length": []}
    return {"average_length": result[0]}

# Endpoint to get the email of a customer based on first name and last name
@app.get("/v1/movie_3/customer_email_by_name", operation_id="get_customer_email", summary="Retrieves the email address of a customer using their first and last names. This operation requires both the first and last names to be provided as input parameters to accurately identify the customer and return their email address.")
async def get_customer_email(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT email FROM customer WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"email": []}
    return {"email": result[0]}

# Endpoint to get the count of customers based on their active status
@app.get("/v1/movie_3/customer_count_by_active_status", operation_id="get_customer_count_by_active_status", summary="Retrieves the total count of customers categorized by their active status. The active status is a binary flag indicating whether a customer is currently active (1) or inactive (0).")
async def get_customer_count_by_active_status(active: int = Query(..., description="Active status of the customer (0 for inactive, 1 for active)")):
    cursor.execute("SELECT COUNT(customer_id) FROM customer WHERE active = ?", (active,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of a customer based on their email
@app.get("/v1/movie_3/customer_name_by_email", operation_id="get_customer_name_by_email", summary="Retrieves the first and last name of a customer using their email address. The operation returns the customer's name if a valid email is provided.")
async def get_customer_name_by_email(email: str = Query(..., description="Email of the customer")):
    cursor.execute("SELECT first_name, last_name FROM customer WHERE email = ?", (email,))
    result = cursor.fetchone()
    if not result:
        return {"first_name": [], "last_name": []}
    return {"first_name": result[0], "last_name": result[1]}

# Endpoint to get the postal code of an address based on the address ID
@app.get("/v1/movie_3/postal_code_by_address_id", operation_id="get_postal_code_by_address_id", summary="Retrieves the postal code associated with a specific address, identified by its unique address ID.")
async def get_postal_code_by_address_id(address_id: int = Query(..., description="ID of the address")):
    cursor.execute("SELECT postal_code FROM address WHERE address_id = ?", (address_id,))
    result = cursor.fetchone()
    if not result:
        return {"postal_code": []}
    return {"postal_code": result[0]}

# Endpoint to get the count of addresses in a specific district
@app.get("/v1/movie_3/address_count_by_district", operation_id="get_address_count_by_district", summary="Retrieves the total number of addresses located in a specified district.")
async def get_address_count_by_district(district: str = Query(..., description="District name")):
    cursor.execute("SELECT COUNT(address_id) FROM address WHERE district = ?", (district,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of an address based on the address ID
@app.get("/v1/movie_3/phone_by_address_id", operation_id="get_phone_by_address_id", summary="Retrieves the phone number associated with a specific address, identified by its unique address ID.")
async def get_phone_by_address_id(address_id: int = Query(..., description="ID of the address")):
    cursor.execute("SELECT phone FROM address WHERE address_id = ?", (address_id,))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the count of films based on their length
@app.get("/v1/movie_3/film_count_by_length", operation_id="get_film_count_by_length", summary="Retrieves the total number of films that match the specified length in minutes.")
async def get_film_count_by_length(length: int = Query(..., description="Length of the film in minutes")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE length = ?", (length,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the special features of a film based on its title
@app.get("/v1/movie_3/film_special_features_by_title", operation_id="get_film_special_features_by_title", summary="Retrieves the special features associated with a specific film, identified by its title.")
async def get_film_special_features_by_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT special_features FROM film WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"special_features": []}
    return {"special_features": result[0]}

# Endpoint to get the address details of a store based on the store ID
@app.get("/v1/movie_3/store_address_details_by_store_id", operation_id="get_store_address_details_by_store_id", summary="Retrieves the address details, including address, address2, and district, for a specific store identified by its unique store_id. The store_id is a required input parameter.")
async def get_store_address_details_by_store_id(store_id: int = Query(..., description="ID of the store")):
    cursor.execute("SELECT T1.address, T1.address2, T1.district FROM address AS T1 INNER JOIN store AS T2 ON T1.address_id = T2.address_id WHERE T2.store_id = ?", (store_id,))
    result = cursor.fetchone()
    if not result:
        return {"address": [], "address2": [], "district": []}
    return {"address": result[0], "address2": result[1], "district": result[2]}

# Endpoint to get the country of a city based on the city name
@app.get("/v1/movie_3/country_by_city_name", operation_id="get_country_by_city_name", summary="Retrieves the country associated with a specified city. The operation uses the provided city name to search for a match in the city table and returns the corresponding country from the country table.")
async def get_country_by_city_name(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.country FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of actors in films based on release year, rental duration, rental rate, and length
@app.get("/v1/movie_3/count_actors_in_films", operation_id="get_count_actors_in_films", summary="Get the count of actors in films based on release year, rental duration, rental rate, and length")
async def get_count_actors_in_films(release_year: int = Query(..., description="Release year of the film"), rental_duration: int = Query(..., description="Rental duration of the film"), rental_rate: float = Query(..., description="Rental rate of the film"), length: int = Query(..., description="Length of the film")):
    cursor.execute("SELECT COUNT(T1.actor_id) FROM film_actor AS T1 INNER JOIN film AS T2 ON T1.film_id = T2.film_id WHERE T2.release_year = ? AND T2.rental_duration = ? AND T2.rental_rate = ? AND T2.length = ?", (release_year, rental_duration, rental_rate, length))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the rating of films based on actor's name, film length, and replacement cost
@app.get("/v1/movie_3/film_rating_by_actor", operation_id="get_film_rating_by_actor", summary="Retrieves the rating of films associated with a specific actor, based on the actor's full name, film length, and replacement cost. This operation filters films by the provided actor's first and last name, film length, and replacement cost, and returns the corresponding film ratings.")
async def get_film_rating_by_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor"), length: int = Query(..., description="Length of the film"), replacement_cost: float = Query(..., description="Replacement cost of the film")):
    cursor.execute("SELECT T3.rating FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? AND T3.length = ? AND T3.replacement_cost = ?", (first_name, last_name, length, replacement_cost))
    result = cursor.fetchall()
    if not result:
        return {"ratings": []}
    return {"ratings": [row[0] for row in result]}

# Endpoint to get the count of films an actor has acted in
@app.get("/v1/movie_3/count_films_by_actor", operation_id="get_count_films_by_actor", summary="Retrieves the total number of films in which a specific actor has appeared, based on the provided first and last name of the actor.")
async def get_count_films_by_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the return date of rentals based on customer name and rental date
@app.get("/v1/movie_3/rental_return_date", operation_id="get_rental_return_date", summary="Retrieves the return date of rentals for a specific customer based on their first and last name and the rental date. The rental date should be provided in 'YYYY-MM-DD HH:MM:SS' format.")
async def get_rental_return_date(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), rental_date: str = Query(..., description="Rental date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T2.return_date FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ? AND T2.rental_date = ?", (first_name, last_name, rental_date))
    result = cursor.fetchall()
    if not result:
        return {"return_dates": []}
    return {"return_dates": [row[0] for row in result]}

# Endpoint to get the names of staff members based on store ID
@app.get("/v1/movie_3/staff_names_by_store", operation_id="get_staff_names_by_store", summary="Retrieves the first and last names of staff members associated with a specific store. The operation requires a store ID as input to filter the staff members accordingly.")
async def get_staff_names_by_store(store_id: int = Query(..., description="Store ID")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM staff AS T1 INNER JOIN store AS T2 ON T1.store_id = T2.store_id WHERE T2.store_id = ?", (store_id,))
    result = cursor.fetchall()
    if not result:
        return {"staff_names": []}
    return {"staff_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the address of a staff member based on their name
@app.get("/v1/movie_3/staff_address", operation_id="get_staff_address", summary="Retrieves the full address of a staff member using their first and last names. The address includes both the primary and secondary address lines.")
async def get_staff_address(first_name: str = Query(..., description="First name of the staff member"), last_name: str = Query(..., description="Last name of the staff member")):
    cursor.execute("SELECT T1.address, T1.address2 FROM address AS T1 INNER JOIN staff AS T2 ON T1.address_id = T2.address_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": {"address": result[0], "address2": result[1]}}

# Endpoint to get the count of addresses in a specific city
@app.get("/v1/movie_3/count_addresses_in_city", operation_id="get_count_addresses_in_city", summary="Retrieves the total number of addresses located in a specified city. The operation calculates this count by joining the 'address' and 'city' tables on the 'city_id' field and filtering for the provided city name.")
async def get_count_addresses_in_city(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT COUNT(T1.address_id) FROM address AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the address of a customer based on their name
@app.get("/v1/movie_3/customer_address", operation_id="get_customer_address", summary="Retrieves the address of a customer using their first and last names. This operation fetches the address details from the address table, which is linked to the customer table via the address_id field. The customer's first and last names are used to filter the results.")
async def get_customer_address(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T1.address FROM address AS T1 INNER JOIN customer AS T2 ON T1.address_id = T2.address_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the email of staff members based on their address
@app.get("/v1/movie_3/staff_email_by_address", operation_id="get_staff_email_by_address", summary="Retrieves the email of staff members associated with a specific address. The operation uses the provided address to look up the corresponding staff member's email in the database.")
async def get_staff_email_by_address(address: str = Query(..., description="Address of the staff member")):
    cursor.execute("SELECT T2.email FROM address AS T1 INNER JOIN staff AS T2 ON T1.address_id = T2.address_id WHERE T1.address = ?", (address,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the payment amount for a specific rental date and customer ID
@app.get("/v1/movie_3/payment_amount_by_rental_date_customer_id", operation_id="get_payment_amount", summary="Retrieves the payment amount associated with a specific rental date and customer ID. The operation filters payments based on the provided rental date and customer ID, returning the corresponding payment amount.")
async def get_payment_amount(rental_date: str = Query(..., description="Rental date in 'YYYY-MM-DD HH:MM:SS' format"), customer_id: int = Query(..., description="Customer ID")):
    cursor.execute("SELECT T1.amount FROM payment AS T1 INNER JOIN rental AS T2 ON T1.rental_id = T2.rental_id WHERE T2.rental_date = ? AND T2.customer_id = ?", (rental_date, customer_id))
    result = cursor.fetchone()
    if not result:
        return {"amount": []}
    return {"amount": result[0]}

# Endpoint to get the category name for a specific film title
@app.get("/v1/movie_3/category_name_by_film_title", operation_id="get_category_name", summary="Retrieves the category name associated with a given film title. This operation fetches the category name from the database by matching the provided film title with the corresponding film_id and category_id.")
async def get_category_name(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T3.name FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T1.title = ?", (film_title,))
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get the category with the highest number of films
@app.get("/v1/movie_3/top_category_by_film_count", operation_id="get_top_category", summary="Retrieves the category with the most films. This operation returns the name of the category that has the highest number of associated films in the database. The data is determined by counting the number of films per category and then selecting the category with the highest count.")
async def get_top_category():
    cursor.execute("SELECT T.name FROM ( SELECT T2.name, COUNT(T1.film_id) AS num FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id GROUP BY T2.name ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get the film title for a specific inventory ID
@app.get("/v1/movie_3/film_title_by_inventory_id", operation_id="get_film_title", summary="Retrieves the title of a film associated with the provided inventory ID. This operation fetches the film title from the film table, which is linked to the inventory table via the film_id field. The inventory_id parameter is used to identify the specific inventory record and retrieve the corresponding film title.")
async def get_film_title(inventory_id: int = Query(..., description="Inventory ID")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T2.inventory_id = ?", (inventory_id,))
    result = cursor.fetchone()
    if not result:
        return {"film_title": []}
    return {"film_title": result[0]}

# Endpoint to get the percentage difference in payment amounts between two stores
@app.get("/v1/movie_3/payment_percentage_difference_by_stores", operation_id="get_payment_percentage_difference", summary="Retrieve the percentage difference in total payment amounts between two specified stores. This operation compares the total payment amounts of two stores and calculates the percentage difference. The input parameters are used to identify the two stores for comparison.")
async def get_payment_percentage_difference(store_id_1: int = Query(..., description="First store ID"), store_id_2: int = Query(..., description="Second store ID")):
    cursor.execute("SELECT CAST((SUM(IIF(T2.store_id = ?, T1.amount, 0)) - SUM(IIF(T2.store_id = ?, T1.amount, 0))) AS REAL) * 100 / SUM(IIF(T2.store_id = ?, T1.amount, 0)) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id INNER JOIN store AS T3 ON T2.store_id = T3.store_id", (store_id_1, store_id_2, store_id_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the ratio of cities between two countries
@app.get("/v1/movie_3/city_ratio_by_countries", operation_id="get_city_ratio", summary="Retrieves the ratio of cities in the first country to the second country. The operation calculates the proportion of cities in the first country relative to the second country by comparing their respective counts.")
async def get_city_ratio(country_1: str = Query(..., description="First country name"), country_2: str = Query(..., description="Second country name")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.country = ?, 1, 0)) AS REAL) / SUM(IIF(T1.country = ?, 1, 0)) FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id", (country_1, country_2))
    result = cursor.fetchone()
    if not result:
        return {"city_ratio": []}
    return {"city_ratio": result[0]}

# Endpoint to get the percentage of films acted by one actor compared to another
@app.get("/v1/movie_3/actor_film_percentage", operation_id="get_actor_film_percentage", summary="Retrieves the percentage of films in which the first actor has appeared compared to the second actor. The calculation is based on the total number of films each actor has been credited for.")
async def get_actor_film_percentage(first_name_1: str = Query(..., description="First name of the first actor"), last_name_1: str = Query(..., description="Last name of the first actor"), first_name_2: str = Query(..., description="First name of the second actor"), last_name_2: str = Query(..., description="Last name of the second actor")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.first_name = ? AND T2.last_name = ?, 1, 0)) AS REAL) * 100 / SUM(IIF(T2.first_name = ? AND T2.last_name = ?, 1, 0)) FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id", (first_name_1, last_name_1, first_name_2, last_name_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of actors in a specific film
@app.get("/v1/movie_3/actor_count_by_film", operation_id="get_actor_count", summary="Retrieves the total number of actors associated with a particular film, identified by its unique film ID.")
async def get_actor_count(film_id: int = Query(..., description="Film ID")):
    cursor.execute("SELECT COUNT(actor_id) FROM film_actor WHERE film_id = ?", (film_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of payments in a specific month and year
@app.get("/v1/movie_3/payment_count_by_month_year", operation_id="get_payment_count_by_month_year", summary="Retrieves the total number of payments made in a specific month and year. The month and year should be provided in the 'YYYY-MM' format.")
async def get_payment_count_by_month_year(month_year: str = Query(..., description="Month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(customer_id) FROM payment WHERE SUBSTR(payment_date, 1, 7) LIKE ?", (month_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get titles of films longer than a specified length
@app.get("/v1/movie_3/film_titles_by_length", operation_id="get_film_titles_by_length", summary="Retrieves the titles of films that exceed a specified duration. The duration is provided in minutes.")
async def get_film_titles_by_length(length: int = Query(..., description="Length of the film in minutes")):
    cursor.execute("SELECT title FROM film WHERE length > ?", (length,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the total amount paid for rentals within a range of rental IDs
@app.get("/v1/movie_3/total_payment_by_rental_id_range", operation_id="get_total_payment_by_rental_id_range", summary="Retrieves the cumulative payment amount for rentals that fall within the specified range of rental IDs. The range is defined by the provided starting and ending rental IDs.")
async def get_total_payment_by_rental_id_range(start_rental_id: int = Query(..., description="Starting rental ID"), end_rental_id: int = Query(..., description="Ending rental ID")):
    cursor.execute("SELECT SUM(amount) FROM payment WHERE rental_id BETWEEN ? AND ?", (start_rental_id, end_rental_id))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the manager staff ID of a store by store ID
@app.get("/v1/movie_3/store_manager_by_store_id", operation_id="get_store_manager_by_store_id", summary="Retrieves the unique identifier of the manager staff assigned to a specific store, based on the provided store ID.")
async def get_store_manager_by_store_id(store_id: int = Query(..., description="Store ID")):
    cursor.execute("SELECT manager_staff_id FROM store WHERE store_id = ?", (store_id,))
    result = cursor.fetchone()
    if not result:
        return {"manager_staff_id": []}
    return {"manager_staff_id": result[0]}

# Endpoint to get the count of rentals on a specific date
@app.get("/v1/movie_3/rental_count_by_date", operation_id="get_rental_count_by_date", summary="Retrieves the total number of rentals that occurred on a specified date. The date must be provided in the 'YYYY-MM-DD' format.")
async def get_rental_count_by_date(rental_date: str = Query(..., description="Rental date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(rental_id) FROM rental WHERE rental_date = ?", (rental_date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get film titles available in a specific store
@app.get("/v1/movie_3/film_titles_by_store_id", operation_id="get_film_titles_by_store_id", summary="Retrieves the titles of films that are available in a store specified by the provided store ID. The operation returns a list of film titles that are currently in stock at the given store.")
async def get_film_titles_by_store_id(store_id: int = Query(..., description="Store ID")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T2.store_id = ?", (store_id,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get customers who rented films for a specific number of days
@app.get("/v1/movie_3/customers_by_rental_days", operation_id="get_customers_by_rental_days", summary="Retrieves the first and last names of customers who rented films for a specified number of days. The operation calculates the number of days each customer rented films and returns those who match the input parameter.")
async def get_customers_by_rental_days(num_days: int = Query(..., description="Number of rental days")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM customer AS T1 INNER JOIN ( SELECT customer_id, COUNT(*) AS num_days FROM ( SELECT *, date(days, '-' || rn || ' day') AS results FROM ( SELECT customer_id, days, row_number() OVER (PARTITION BY customer_id ORDER BY days) AS rn FROM ( SELECT DISTINCT customer_id, date(rental_date) AS days FROM rental ) ) ) GROUP BY customer_id, results HAVING num_days = ? ) AS T2 ON T1.customer_id = T2.customer_id", (num_days,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the special features of the top film in a specific category
@app.get("/v1/movie_3/top_film_special_features_by_category", operation_id="get_top_film_special_features_by_category", summary="Retrieves the special features of the top-ranked film in a specified category. The category is determined by the provided category name. The special features are ordered in descending order, with the top feature being returned.")
async def get_top_film_special_features_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T1.special_features FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ? ORDER BY T1.special_features DESC LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"special_features": []}
    return {"special_features": result[0]}

# Endpoint to get the actor with the most film appearances
@app.get("/v1/movie_3/actor_most_film_appearances", operation_id="get_actor_most_film_appearances", summary="Retrieves the actor who has appeared in the most films. The actor's first and last names are returned.")
async def get_actor_most_film_appearances():
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T2.first_name, T2.last_name, COUNT(T1.film_id) AS num FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id GROUP BY T2.first_name, T2.last_name ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"actor": []}
    return {"actor": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of films in a specific category with a specific rental duration
@app.get("/v1/movie_3/count_films_category_rental_duration", operation_id="get_count_films_category_rental_duration", summary="Retrieves the total number of films in a specified category that have a given rental duration. The rental duration is provided in days, and the category is identified by its name.")
async def get_count_films_category_rental_duration(rental_duration: int = Query(..., description="Rental duration in days"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T1.rental_duration = ? AND T3.name = ?", (rental_duration, category_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the staff member with the most inactive customers
@app.get("/v1/movie_3/staff_most_inactive_customers", operation_id="get_staff_most_inactive_customers", summary="Retrieves the staff member who has the highest number of inactive customers. This operation identifies the staff member associated with the store that has the most customers marked as inactive. The response includes the first and last names of the staff member.")
async def get_staff_most_inactive_customers():
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T3.first_name, T3.last_name, COUNT(T1.customer_id) AS num FROM customer AS T1 INNER JOIN store AS T2 ON T1.store_id = T2.store_id INNER JOIN staff AS T3 ON T2.store_id = T3.store_id WHERE T1.active = 0 GROUP BY T3.first_name, T3.last_name ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"staff": []}
    return {"staff": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the highest rental rate per day for a specific category
@app.get("/v1/movie_3/highest_rental_rate_per_day_category", operation_id="get_highest_rental_rate_per_day_category", summary="Retrieves the highest daily rental rate for a specified movie category. The category is identified by its name, and the rate is calculated by dividing the rental rate by the rental duration. The result is sorted in descending order, and the top rate is returned.")
async def get_highest_rental_rate_per_day_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T1.rental_rate FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ? ORDER BY T1.rental_rate / T1.rental_duration DESC LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"rental_rate": []}
    return {"rental_rate": result[0]}

# Endpoint to get the address details of a specific store
@app.get("/v1/movie_3/store_address_details", operation_id="get_store_address_details", summary="Retrieves the full address details of a specific store, including the street address, secondary address, and district. The store is identified by its unique store_id.")
async def get_store_address_details(store_id: int = Query(..., description="Store ID")):
    cursor.execute("SELECT T3.address, T3.address2, T3.district FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id INNER JOIN address AS T3 ON T2.city_id = T3.city_id INNER JOIN store AS T4 ON T3.address_id = T4.address_id WHERE T4.store_id = ?", (store_id,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": {"address": result[0], "address2": result[1], "district": result[2]}}

# Endpoint to get the count of customers in a specific city
@app.get("/v1/movie_3/count_customers_city", operation_id="get_count_customers_city", summary="Retrieves the total number of customers residing in a specified city. The city is identified by its name.")
async def get_count_customers_city(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(T3.customer_id) FROM city AS T1 INNER JOIN address AS T2 ON T1.city_id = T2.city_id INNER JOIN customer AS T3 ON T2.address_id = T3.address_id WHERE T1.city = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cities in a specific country
@app.get("/v1/movie_3/count_cities_country", operation_id="get_count_cities_country", summary="Retrieves the total number of cities in a specified country. The operation calculates this count by joining the 'country' and 'city' tables based on their shared 'country_id' and filtering for the provided country name.")
async def get_count_cities_country(country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT COUNT(T2.city) FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id WHERE T1.country = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customers in a specific country
@app.get("/v1/movie_3/customers_country", operation_id="get_customers_country", summary="Retrieves the first and last names of customers residing in a specified country. The country is identified by its name.")
async def get_customers_country(country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT T4.first_name, T4.last_name FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id INNER JOIN address AS T3 ON T2.city_id = T3.city_id INNER JOIN customer AS T4 ON T3.address_id = T4.address_id WHERE T1.country = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of films in a specific category with a rental rate less than a specified value
@app.get("/v1/movie_3/count_films_category_rental_rate", operation_id="get_count_films_category_rental_rate", summary="Retrieves the number of films in a specified category that have a rental rate below a given value. The category is identified by its name, and the rental rate is provided as a maximum threshold.")
async def get_count_films_category_rental_rate(rental_rate: float = Query(..., description="Maximum rental rate"), category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id INNER JOIN film AS T3 ON T1.film_id = T3.film_id WHERE T3.rental_rate < ? AND T2.name = ?", (rental_rate, category_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer with the most rentals
@app.get("/v1/movie_3/customer_most_rentals", operation_id="get_customer_most_rentals", summary="Retrieves the first name and last name of the customer who has rented the most movies. The data is obtained by counting the number of rentals per customer and sorting the results in descending order. The customer with the highest rental count is returned.")
async def get_customer_most_rentals():
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T2.first_name, T2.last_name, COUNT(T1.rental_id) AS num FROM rental AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id GROUP BY T2.first_name, T2.last_name ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of rentals for a specific film title
@app.get("/v1/movie_3/rental_count_by_film_title", operation_id="get_rental_count_by_film_title", summary="Retrieves the total number of rentals for a specific film, identified by its title.")
async def get_rental_count_by_film_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT COUNT(T3.rental_id) FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id INNER JOIN rental AS T3 ON T2.inventory_id = T3.inventory_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the actor with the most films of a specific rating
@app.get("/v1/movie_3/top_actor_by_rating", operation_id="get_top_actor_by_rating", summary="Retrieves the actor who has starred in the most films with a specified rating. The rating is provided as an input parameter.")
async def get_top_actor_by_rating(rating: str = Query(..., description="Rating of the films")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T1.first_name, T1.last_name, COUNT(T2.film_id) AS num FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.rating = ? GROUP BY T1.first_name, T1.last_name ) AS T ORDER BY T.num DESC LIMIT 1", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"actor": []}
    return {"actor": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the top actors by the number of films they have acted in
@app.get("/v1/movie_3/top_actors_by_film_count", operation_id="get_top_actors_by_film_count", summary="Retrieve a list of the top actors, ranked by the number of films they have appeared in. The list is limited to the specified number of actors, as determined by the input parameter.")
async def get_top_actors_by_film_count(limit: int = Query(..., description="Number of top actors to return")):
    cursor.execute("SELECT T.first_name, T.last_name, num FROM ( SELECT T1.first_name, T1.last_name, COUNT(T2.film_id) AS num FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id GROUP BY T1.first_name, T1.last_name ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": [{"first_name": row[0], "last_name": row[1], "num_films": row[2]} for row in result]}

# Endpoint to get actor IDs by last name
@app.get("/v1/movie_3/actor_ids_by_last_name", operation_id="get_actor_ids_by_last_name", summary="Retrieves the unique identifiers of all actors whose last name matches the provided input. This operation is useful for finding specific actors based on their last name.")
async def get_actor_ids_by_last_name(last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT actor_id FROM actor WHERE last_name = ?", (last_name,))
    result = cursor.fetchall()
    if not result:
        return {"actor_ids": []}
    return {"actor_ids": [row[0] for row in result]}

# Endpoint to get film titles with the minimum replacement cost
@app.get("/v1/movie_3/film_titles_min_replacement_cost", operation_id="get_film_titles_min_replacement_cost", summary="Retrieves the titles of films that have the lowest replacement cost. This operation identifies the minimum replacement cost among all films and returns the titles of films that match this cost.")
async def get_film_titles_min_replacement_cost():
    cursor.execute("SELECT title FROM film WHERE replacement_cost = ( SELECT MIN(replacement_cost) FROM film )")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get details of the longest films
@app.get("/v1/movie_3/longest_films_details", operation_id="get_longest_films_details", summary="Retrieves detailed information about the longest films, including their titles, descriptions, and special features. The operation returns up to the specified number of longest films, sorted by length in descending order.")
async def get_longest_films_details(limit: int = Query(..., description="Number of longest films to return")):
    cursor.execute("SELECT title, description, special_features FROM film WHERE length = ( SELECT MAX(length) FROM film ) LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"films": []}
    return {"films": [{"title": row[0], "description": row[1], "special_features": row[2]} for row in result]}

# Endpoint to get the count of distinct rentals within a date range
@app.get("/v1/movie_3/rental_count_by_date_range", operation_id="get_rental_count_by_date_range", summary="Retrieves the total number of unique rentals that occurred within the specified date range. The date range is defined by the provided start and end dates, both in 'YYYY-MM-DD' format.")
async def get_rental_count_by_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(DISTINCT rental_id) FROM rental WHERE date(rental_date) BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average payment amount grouped by customer ID
@app.get("/v1/movie_3/average_payment_by_customer", operation_id="get_average_payment_by_customer", summary="Retrieves the average payment amount for each unique customer ID. This operation calculates the mean payment value for every distinct customer, providing insights into spending patterns.")
async def get_average_payment_by_customer():
    cursor.execute("SELECT AVG(amount) FROM payment GROUP BY customer_id")
    result = cursor.fetchall()
    if not result:
        return {"average_payments": []}
    return {"average_payments": [{"customer_id": idx + 1, "average_amount": row[0]} for idx, row in enumerate(result)]}

# Endpoint to get staff details by store ID
@app.get("/v1/movie_3/staff_details_by_store_id", operation_id="get_staff_details_by_store_id", summary="Retrieves the first name, last name, and email of all staff members associated with the specified store ID.")
async def get_staff_details_by_store_id(store_id: int = Query(..., description="Store ID")):
    cursor.execute("SELECT first_name, last_name, email FROM staff WHERE store_id = ?", (store_id,))
    result = cursor.fetchall()
    if not result:
        return {"staff": []}
    return {"staff": [{"first_name": row[0], "last_name": row[1], "email": row[2]} for row in result]}

# Endpoint to get the percentage of inactive customers
@app.get("/v1/movie_3/percentage_inactive_customers", operation_id="get_percentage_inactive_customers", summary="Retrieves the percentage of inactive customers by calculating the ratio of inactive customers to the total number of customers. The operation considers a customer as inactive if the 'active' field in the customer table is set to 0.")
async def get_percentage_inactive_customers():
    cursor.execute("SELECT CAST(SUM(IIF(active = 0, 1, 0)) AS REAL) * 100 / COUNT(customer_id) FROM customer")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the description and title of a film based on film ID
@app.get("/v1/movie_3/film_description_title", operation_id="get_film_description_title", summary="Retrieves the title and description of a specific film using its unique identifier.")
async def get_film_description_title(film_id: int = Query(..., description="Film ID")):
    cursor.execute("SELECT description, title FROM film_text WHERE film_id = ?", (film_id,))
    result = cursor.fetchall()
    if not result:
        return {"films": []}
    return {"films": result}

# Endpoint to get the total payment amount for a specific month and year
@app.get("/v1/movie_3/total_payment_amount", operation_id="get_total_payment_amount", summary="Retrieves the total payment amount for a given month and year. The input parameter specifies the desired month and year in 'YYYY-MM' format. The operation calculates the sum of all payment amounts for the specified period.")
async def get_total_payment_amount(year_month: str = Query(..., description="Year and month in 'YYYY-MM' format")):
    cursor.execute("SELECT SUM(amount) FROM payment WHERE SUBSTR(payment_date, 1, 7) = ?", (year_month,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the titles of films an actor has acted in
@app.get("/v1/movie_3/actor_film_titles", operation_id="get_actor_film_titles", summary="Retrieves the titles of films in which a specific actor has appeared, based on their first and last name. The operation uses the provided actor's first and last name to search for matching records in the database and returns the corresponding film titles.")
async def get_actor_film_titles(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"film_titles": []}
    return {"film_titles": result}

# Endpoint to get the actors of a specific film
@app.get("/v1/movie_3/film_actors", operation_id="get_film_actors", summary="Retrieves the first and last names of all actors associated with a specific film, identified by its title.")
async def get_film_actors(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T3.first_name, T3.last_name FROM film_actor AS T1 INNER JOIN film AS T2 ON T1.film_id = T2.film_id INNER JOIN actor AS T3 ON T1.actor_id = T3.actor_id WHERE T2.title = ?", (film_title,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": result}

# Endpoint to get the count of films in a specific category and rating
@app.get("/v1/movie_3/film_count_category_rating", operation_id="get_film_count_category_rating", summary="Retrieves the total number of films that belong to a specified category and have a certain rating. The category is identified by its name, and the rating is a string value representing the film's rating.")
async def get_film_count_category_rating(category_name: str = Query(..., description="Name of the category"), rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ? AND T1.rating = ?", (category_name, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the categories of films an actor has acted in
@app.get("/v1/movie_3/actor_film_categories", operation_id="get_actor_film_categories", summary="Retrieve the categories of films in which a specific actor has appeared. The actor is identified by their first and last name. The operation returns a list of categories that the actor's films belong to.")
async def get_actor_film_categories(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T5.name FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id INNER JOIN film_category AS T4 ON T2.film_id = T4.film_id INNER JOIN category AS T5 ON T4.category_id = T5.category_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": result}

# Endpoint to get the titles of films in a specific category with a limit
@app.get("/v1/movie_3/film_titles_category_limit", operation_id="get_film_titles_category_limit", summary="Retrieves a specified number of film titles from a particular category. The operation filters films based on the provided category name and returns the corresponding titles up to the defined limit.")
async def get_film_titles_category_limit(category_name: str = Query(..., description="Name of the category"), limit: int = Query(..., description="Limit of the number of titles to return")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ? LIMIT ?", (category_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"film_titles": []}
    return {"film_titles": result}

# Endpoint to get the language and cost per rental duration of a specific film
@app.get("/v1/movie_3/film_language_cost", operation_id="get_film_language_cost", summary="Retrieves the name of the language and the cost per rental duration for a specific film. The cost is calculated by dividing the replacement cost by the rental duration. The film is identified by its title.")
async def get_film_language_cost(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.name, T1.replacement_cost / T1.rental_duration AS cost FROM film AS T1 INNER JOIN language AS T2 ON T1.language_id = T2.language_id WHERE T1.title = ?", (film_title,))
    result = cursor.fetchall()
    if not result:
        return {"language_cost": []}
    return {"language_cost": result}

# Endpoint to get the titles of films rented on a specific date
@app.get("/v1/movie_3/film_titles_rented_on_date", operation_id="get_film_titles_rented_on_date", summary="Retrieves the titles of films that were rented on a specific date. The date should be provided in 'YYYY-MM-DD' format. The operation fetches the data from the film, inventory, and rental tables, and returns the titles of films rented on the given date.")
async def get_film_titles_rented_on_date(rental_date: str = Query(..., description="Rental date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id INNER JOIN rental AS T3 ON T2.inventory_id = T3.inventory_id WHERE SUBSTR(T3.rental_date, 1, 10) = ?", (rental_date,))
    result = cursor.fetchall()
    if not result:
        return {"film_titles": []}
    return {"film_titles": result}

# Endpoint to get the titles of films rented by a specific customer in a specific month and year
@app.get("/v1/movie_3/customer_film_titles_month_year", operation_id="get_customer_film_titles_month_year", summary="Retrieves the titles of films rented by a specific customer during a particular month and year. The operation requires the customer's first and last names, as well as the year and month of the rental period in 'YYYY' and 'MM' format, respectively.")
async def get_customer_film_titles_month_year(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), year: str = Query(..., description="Year in 'YYYY' format"), month: str = Query(..., description="Month in 'MM' format")):
    cursor.execute("SELECT T4.title FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T1.first_name = ? AND T1.last_name = ? AND STRFTIME('%Y', T2.rental_date) = ? AND STRFTIME('%m', T2.rental_date) = ?", (first_name, last_name, year, month))
    result = cursor.fetchall()
    if not result:
        return {"film_titles": []}
    return {"film_titles": result}

# Endpoint to get actor details for a specific film title
@app.get("/v1/movie_3/actor_details_by_film_title", operation_id="get_actor_details_by_film_title", summary="Retrieves detailed information about the actors who have appeared in a specific film, including their first and last names and the inventory IDs associated with the film. To use this endpoint, provide the exact title of the film as an input parameter.")
async def get_actor_details_by_film_title(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T4.inventory_id, T1.first_name, T1.last_name FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id INNER JOIN inventory AS T4 ON T2.film_id = T4.film_id WHERE T3.title = ?", (film_title,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": result}

# Endpoint to get film titles and categories rented by a specific customer in a specific year and month
@app.get("/v1/movie_3/film_titles_categories_by_customer_year_month", operation_id="get_film_titles_categories_by_customer_year_month", summary="Retrieves the titles and categories of films rented by a specific customer during a particular year and month. The operation requires the customer's first and last names, as well as the year and month in question, to filter the results accordingly.")
async def get_film_titles_categories_by_customer_year_month(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), year: str = Query(..., description="Year in 'YYYY' format"), month: str = Query(..., description="Month in 'MM' format")):
    cursor.execute("SELECT T3.title, T2.name FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id INNER JOIN film AS T3 ON T1.film_id = T3.film_id INNER JOIN inventory AS T4 ON T3.film_id = T4.film_id INNER JOIN customer AS T5 ON T4.store_id = T5.store_id INNER JOIN rental AS T6 ON T4.inventory_id = T6.inventory_id WHERE T5.first_name = ? AND T5.last_name = ? AND STRFTIME('%Y',T3.rental_rate) = ? AND STRFTIME('%m',T3.rental_rate) = ?", (first_name, last_name, year, month))
    result = cursor.fetchall()
    if not result:
        return {"films": []}
    return {"films": result}

# Endpoint to get the count of rentals by a specific customer
@app.get("/v1/movie_3/rental_count_by_customer", operation_id="get_rental_count_by_customer", summary="Retrieves the total number of rentals made by a customer identified by their first and last names. The operation uses the provided first and last names to filter the customer records and calculate the rental count.")
async def get_rental_count_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T1.rental_id) FROM rental AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer details who rented a specific film title
@app.get("/v1/movie_3/customer_details_by_film_title", operation_id="get_customer_details_by_film_title", summary="Retrieves the first and last names, as well as the city of residence, of customers who have rented a film with the specified title. This operation provides a comprehensive view of customer details associated with a particular film title.")
async def get_customer_details_by_film_title(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T4.first_name, T4.last_name, T6.city FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id INNER JOIN rental AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN customer AS T4 ON T3.customer_id = T4.customer_id INNER JOIN address AS T5 ON T4.address_id = T5.address_id INNER JOIN city AS T6 ON T5.city_id = T6.city_id WHERE T1.title = ?", (film_title,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the percentage of customers from a specific country
@app.get("/v1/movie_3/customer_percentage_by_country", operation_id="get_customer_percentage_by_country", summary="Retrieves the percentage of customers residing in a specified country. This operation calculates the proportion of customers from the given country by comparing the total count of customers from that country to the overall customer count. The result is expressed as a percentage.")
async def get_customer_percentage_by_country(country: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.country = ?, 1, 0)) AS REAL) * 100 / COUNT(T4.customer_id) FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id INNER JOIN address AS T3 ON T2.city_id = T3.city_id INNER JOIN customer AS T4 ON T3.address_id = T4.address_id", (country,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage difference in film appearances between two actors
@app.get("/v1/movie_3/actor_appearance_percentage_difference", operation_id="get_actor_appearance_percentage_difference", summary="Retrieves the percentage difference in the number of film appearances between two specified actors. The calculation is based on the total number of films in which the first actor has appeared, with the result expressed as a percentage.")
async def get_actor_appearance_percentage_difference(first_name_1: str = Query(..., description="First name of the first actor"), last_name_1: str = Query(..., description="Last name of the first actor"), first_name_2: str = Query(..., description="First name of the second actor"), last_name_2: str = Query(..., description="Last name of the second actor")):
    cursor.execute("SELECT CAST((SUM(IIF(T1.first_name = ? AND T1.last_name = ?, 1, 0)) - SUM(IIF(T1.first_name = ? AND T1.last_name = ?, 1, 0))) AS REAL) * 100 / SUM(IIF(T1.first_name = ? AND T1.last_name = ?, 1, 0)) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id", (first_name_1, last_name_1, first_name_2, last_name_2, first_name_2, last_name_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get customer details by name
@app.get("/v1/movie_3/customer_details_by_name", operation_id="get_customer_details_by_name", summary="Retrieves the email, address, city, and country of a customer using their first and last names. The operation performs a search in the customer database, joining relevant tables to gather comprehensive customer details.")
async def get_customer_details_by_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T1.email, T2.address, T3.city, T4.country FROM customer AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id INNER JOIN city AS T3 ON T2.city_id = T3.city_id INNER JOIN country AS T4 ON T3.country_id = T4.country_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get customer names associated with a specific staff member
@app.get("/v1/movie_3/customer_names_by_staff", operation_id="get_customer_names_by_staff", summary="Retrieves the first and last names of customers who share the same address as a specified staff member. The operation accepts the first and last names of the staff member as input parameters, along with a limit parameter to restrict the number of results returned.")
async def get_customer_names_by_staff(first_name: str = Query(..., description="First name of the staff member"), last_name: str = Query(..., description="Last name of the staff member"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.first_name, T3.last_name FROM staff AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id INNER JOIN customer AS T3 ON T2.address_id = T3.address_id WHERE T1.first_name = ? AND T1.last_name = ? LIMIT ?", (first_name, last_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the total payment amount by a specific customer
@app.get("/v1/movie_3/total_payment_by_customer", operation_id="get_total_payment_by_customer", summary="Retrieves the total payment amount made by a customer identified by their first and last names. The operation calculates the sum of all payment amounts associated with the specified customer.")
async def get_total_payment_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT SUM(T2.amount) FROM customer AS T1 INNER JOIN payment AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"total_payment": []}
    return {"total_payment": result[0]}

# Endpoint to get customer details who have made payments above a certain threshold
@app.get("/v1/movie_3/customer_details_above_payment_threshold", operation_id="get_customer_details_above_payment_threshold", summary="Retrieves the first name, last name, and email of customers who have made payments above a certain threshold. The threshold is calculated as a percentage of the average payment amount. The endpoint returns distinct customer details, ensuring no duplicates.")
async def get_customer_details_above_payment_threshold(threshold_percentage: float = Query(..., description="Threshold percentage of the average payment amount")):
    cursor.execute("SELECT DISTINCT T2.first_name, T2.last_name, T2.email FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id INNER JOIN address AS T3 ON T2.address_id = T3.address_id WHERE T1.amount > ( SELECT AVG(amount) FROM payment ) * ?", (threshold_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the count of films based on rental rate
@app.get("/v1/movie_3/film_count_by_rental_rate", operation_id="get_film_count_by_rental_rate", summary="Retrieves the total number of films that have a specified rental rate. The rental rate is provided as an input parameter.")
async def get_film_count_by_rental_rate(rental_rate: float = Query(..., description="Rental rate of the film")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE rental_rate = ?", (rental_rate,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on last name and customer ID
@app.get("/v1/movie_3/customer_count_by_last_name_and_id", operation_id="get_customer_count_by_last_name_and_id", summary="Retrieves the number of customers with a given last name and a customer ID less than the provided value. This operation is useful for obtaining a count of customers who meet specific criteria, enabling targeted analysis or segmentation.")
async def get_customer_count_by_last_name_and_id(last_name: str = Query(..., description="Last name of the customer"), customer_id: int = Query(..., description="Customer ID less than this value")):
    cursor.execute("SELECT COUNT(customer_id) FROM customer WHERE last_name = ? AND customer_id < ?", (last_name, customer_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get actor last names based on film description
@app.get("/v1/movie_3/actor_last_names_by_film_description", operation_id="get_actor_last_names_by_film_description", summary="Retrieve the last names of actors who have appeared in films that match a given description. The description parameter is used to filter the films and identify the relevant actors.")
async def get_actor_last_names_by_film_description(description: str = Query(..., description="Description of the film")):
    cursor.execute("SELECT T1.last_name FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the highest replacement cost film title for a specific actor
@app.get("/v1/movie_3/highest_replacement_cost_film_by_actor", operation_id="get_highest_replacement_cost_film_by_actor", summary="Retrieves the title of the film with the highest replacement cost associated with a specific actor. The operation requires the first and last name of the actor as input parameters to identify the relevant film.")
async def get_highest_replacement_cost_film_by_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? ORDER BY replacement_cost DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the highest rental rate film title for a specific store
@app.get("/v1/movie_3/highest_rental_rate_film_by_store", operation_id="get_highest_rental_rate_film_by_store", summary="Retrieves the title of the film with the highest rental rate for a given store. The store is identified by its unique ID. The operation returns the title of the film that has the highest rental rate among all films available in the specified store.")
async def get_highest_rental_rate_film_by_store(store_id: int = Query(..., description="Store ID")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T2.store_id = ? ORDER BY rental_rate DESC LIMIT 1", (store_id,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get film titles based on actor name and replacement cost
@app.get("/v1/movie_3/film_titles_by_actor_and_replacement_cost", operation_id="get_film_titles_by_actor_and_replacement_cost", summary="Retrieve the titles of films featuring a specific actor and having a particular replacement cost. The operation requires the first and last name of the actor, as well as the replacement cost of the film.")
async def get_film_titles_by_actor_and_replacement_cost(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor"), replacement_cost: float = Query(..., description="Replacement cost of the film")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? AND T3.replacement_cost = ?", (first_name, last_name, replacement_cost))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get inventory IDs based on film title
@app.get("/v1/movie_3/inventory_ids_by_film_title", operation_id="get_inventory_ids_by_film_title", summary="Retrieves the inventory IDs associated with a specific film title. The operation searches for the film by its title and returns the corresponding inventory IDs from the inventory table.")
async def get_inventory_ids_by_film_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.inventory_id FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"inventory_ids": []}
    return {"inventory_ids": [row[0] for row in result]}

# Endpoint to get the count of actors based on film length and actor name
@app.get("/v1/movie_3/actor_count_by_film_length_and_name", operation_id="get_actor_count_by_film_length_and_name", summary="Retrieve the number of actors who have appeared in films of a certain duration and share a specific full name. The response is based on the provided film length and actor's first and last names.")
async def get_actor_count_by_film_length_and_name(length: int = Query(..., description="Length of the film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(T1.actor_id) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.length = ? AND T1.first_name = ? AND T1.last_name = ?", (length, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of films based on inventory ID range and rating
@app.get("/v1/movie_3/film_count_by_inventory_id_range_and_rating", operation_id="get_film_count_by_inventory_id_range_and_rating", summary="Retrieves the total number of films that have inventory IDs within the specified range and a particular rating. The range is defined by the minimum and maximum inventory IDs, and the rating is a specific value.")
async def get_film_count_by_inventory_id_range_and_rating(min_inventory_id: int = Query(..., description="Minimum inventory ID"), max_inventory_id: int = Query(..., description="Maximum inventory ID"), rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T2.inventory_id BETWEEN ? AND ? AND T1.rating = ?", (min_inventory_id, max_inventory_id, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of actors based on rental rate and actor name
@app.get("/v1/movie_3/actor_count_by_rental_rate_and_name", operation_id="get_actor_count_by_rental_rate_and_name", summary="Retrieves the number of actors who have appeared in films with a specified rental rate and share a given first and last name. This operation is useful for understanding the distribution of actors across films with different rental rates.")
async def get_actor_count_by_rental_rate_and_name(rental_rate: float = Query(..., description="Rental rate of the film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(T1.actor_id) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.rental_rate = ? AND T1.first_name = ? AND T1.last_name = ?", (rental_rate, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get inventory IDs for films of a specific length range and actor name
@app.get("/v1/movie_3/inventory_ids_by_film_length_and_actor", operation_id="get_inventory_ids_by_film_length_and_actor", summary="Retrieves the inventory IDs of films that fall within a specified length range and are associated with a particular actor. The operation considers the first and last names of the actor, as well as the minimum and maximum length of the films.")
async def get_inventory_ids_by_film_length_and_actor(min_length: int = Query(..., description="Minimum film length"), max_length: int = Query(..., description="Maximum film length"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T4.inventory_id FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id INNER JOIN inventory AS T4 ON T3.film_id = T4.film_id WHERE T3.length BETWEEN ? AND ? AND T1.first_name = ? AND T1.last_name = ?", (min_length, max_length, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"inventory_ids": []}
    return {"inventory_ids": [row[0] for row in result]}

# Endpoint to get the store ID and inventory ID of the longest film
@app.get("/v1/movie_3/longest_film_store_inventory", operation_id="get_longest_film_store_inventory", summary="Retrieves the store and inventory details of the longest film available in the inventory. The operation returns the store ID and inventory ID of the film with the maximum length.")
async def get_longest_film_store_inventory():
    cursor.execute("SELECT T2.store_id, T2.inventory_id FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id ORDER BY T1.length DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"store_id": [], "inventory_id": []}
    return {"store_id": result[0], "inventory_id": result[1]}

# Endpoint to get film titles for films of a specific length range and actor name
@app.get("/v1/movie_3/film_titles_by_length_and_actor", operation_id="get_film_titles_by_length_and_actor", summary="Retrieves the titles of films within a specified length range that were acted in by a specific actor. The operation accepts the minimum and maximum length of the films, as well as the first and last name of the actor, to filter the results accordingly.")
async def get_film_titles_by_length_and_actor(min_length: int = Query(..., description="Minimum film length"), max_length: int = Query(..., description="Maximum film length"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.length BETWEEN ? AND ? AND T1.first_name = ? AND T1.last_name = ?", (min_length, max_length, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get inventory IDs for films with a specific actor name and rental rate
@app.get("/v1/movie_3/inventory_ids_by_actor_and_rental_rate", operation_id="get_inventory_ids_by_actor_and_rental_rate", summary="Retrieves the inventory IDs of films featuring a specific actor and having a certain rental rate. The operation requires the first and last name of the actor, as well as the rental rate of the film.")
async def get_inventory_ids_by_actor_and_rental_rate(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor"), rental_rate: float = Query(..., description="Rental rate of the film")):
    cursor.execute("SELECT T4.inventory_id FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id INNER JOIN inventory AS T4 ON T3.film_id = T4.film_id WHERE T1.first_name = ? AND T1.last_name = ? AND T3.rental_rate = ?", (first_name, last_name, rental_rate))
    result = cursor.fetchall()
    if not result:
        return {"inventory_ids": []}
    return {"inventory_ids": [row[0] for row in result]}

# Endpoint to get store IDs for films with rental rates above 60% of the average
@app.get("/v1/movie_3/store_ids_above_average_rental_rate", operation_id="get_store_ids_above_average_rental_rate", summary="Retrieves the store IDs of films that have a rental rate higher than 60% of the average rental rate. This operation calculates the average rental rate across all films and identifies those with rental rates surpassing 60% of this average. The resulting store IDs correspond to the stores carrying these films.")
async def get_store_ids_above_average_rental_rate():
    cursor.execute("SELECT T2.store_id FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T1.rental_rate > ( SELECT AVG(T1.rental_rate) * 0.6 FROM film AS T1 )")
    result = cursor.fetchall()
    if not result:
        return {"store_ids": []}
    return {"store_ids": [row[0] for row in result]}

# Endpoint to get the proportion of films with a specific rating for a given actor
@app.get("/v1/movie_3/proportion_films_by_rating_and_actor", operation_id="get_proportion_films_by_rating_and_actor", summary="Retrieve the proportion of films with a specific rating for a given actor. This operation calculates the ratio of films with the provided rating to the total number of films in which the specified actor has appeared. The actor is identified by their first and last names.")
async def get_proportion_films_by_rating_and_actor(rating: str = Query(..., description="Rating of the film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.rating = ?, 1, 0)) AS REAL) / COUNT(T3.film_id) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ?", (rating, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the address with the minimum city ID in a specific district
@app.get("/v1/movie_3/address_min_city_id_by_district", operation_id="get_address_min_city_id_by_district", summary="Retrieves the address with the lowest city ID in the specified district. The district parameter is used to filter the results.")
async def get_address_min_city_id_by_district(district: str = Query(..., description="District name")):
    cursor.execute("SELECT address FROM address WHERE district = ? AND city_id = ( SELECT MIN(city_id) FROM address WHERE district = ? )", (district, district))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get customer details created in a specific year and with a specific active status
@app.get("/v1/movie_3/customer_details_by_year_and_status", operation_id="get_customer_details_by_year_and_status", summary="Retrieves the first name, last name, and email of customers created in a specified year and with a specified active status. The year should be provided in 'YYYY' format, and the active status should be indicated as either 0 or 1.")
async def get_customer_details_by_year_and_status(year: str = Query(..., description="Year in 'YYYY' format"), active: int = Query(..., description="Active status (0 or 1)")):
    cursor.execute("SELECT first_name, last_name, email FROM customer WHERE STRFTIME('%Y',create_date) = ? AND active = ?", (year, active))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first_name": row[0], "last_name": row[1], "email": row[2]} for row in result]}

# Endpoint to get the percentage of films with a specific rating
@app.get("/v1/movie_3/percentage_films_by_rating", operation_id="get_percentage_films_by_rating", summary="Retrieves the percentage of films that have a specified rating. The rating is provided as an input parameter, and the operation calculates the proportion of films with that rating relative to the total number of films in the database.")
async def get_percentage_films_by_rating(rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT CAST(SUM(IIF(rating = ?, 1, 0)) AS REAL) * 100 / COUNT(film_id) FROM film", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top 10 film titles by rental rate per rental duration
@app.get("/v1/movie_3/top_films_by_rental_rate_per_duration", operation_id="get_top_films_by_rental_rate_per_duration", summary="Retrieves the top 10 films with the highest rental rate per rental duration. The films are ranked based on the ratio of their rental rate to rental duration, providing a measure of their relative value for rental.")
async def get_top_films_by_rental_rate_per_duration():
    cursor.execute("SELECT title FROM film ORDER BY rental_rate / rental_duration DESC LIMIT 10")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the average payment amount for a specific customer
@app.get("/v1/movie_3/average_payment_amount", operation_id="get_average_payment_amount", summary="Retrieves the average payment amount for a specific customer. The operation requires a customer ID as input and calculates the average payment amount associated with that customer.")
async def get_average_payment_amount(customer_id: int = Query(..., description="Customer ID")):
    cursor.execute("SELECT AVG(amount) FROM payment WHERE customer_id = ?", (customer_id,))
    result = cursor.fetchone()
    if not result:
        return {"average_amount": []}
    return {"average_amount": result[0]}

# Endpoint to get the count of rentals where the rental period is longer than the average rental period
@app.get("/v1/movie_3/count_long_rentals", operation_id="get_count_long_rentals", summary="Retrieves the number of rentals that exceed the average rental duration. This operation calculates the average rental period and identifies rentals that surpass this duration, providing a count of such instances.")
async def get_count_long_rentals():
    cursor.execute("SELECT COUNT(customer_id) FROM rental WHERE return_date - rental_date > ( SELECT AVG(return_date - rental_date) FROM rental )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of films in a specific category
@app.get("/v1/movie_3/percentage_films_in_category", operation_id="get_percentage_films_in_category", summary="Retrieves the percentage of films that belong to a specific category. The category is identified by its name, which is provided as an input parameter. The calculation is based on the total number of films and the count of films in the specified category.")
async def get_percentage_films_in_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.name = ?, 1, 0)) AS REAL) * 100 / COUNT(T2.category_id) FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the actor with the most films in a specific category
@app.get("/v1/movie_3/top_actor_in_category", operation_id="get_top_actor_in_category", summary="Retrieves the actor with the highest number of films in a specified category. The category is determined by the provided category ID.")
async def get_top_actor_in_category(category_id: int = Query(..., description="Category ID")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T1.first_name, T1.last_name, COUNT(T2.film_id) AS num FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film_category AS T3 ON T2.film_id = T3.film_id WHERE T3.category_id = ? GROUP BY T1.first_name, T1.last_name ) AS T ORDER BY T.num DESC LIMIT 1", (category_id,))
    result = cursor.fetchone()
    if not result:
        return {"actor": []}
    return {"actor": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the difference in average number of customers between two countries
@app.get("/v1/movie_3/customer_country_difference", operation_id="get_customer_country_difference", summary="Retrieves the difference in average number of customers between two specified countries. The calculation is based on the customers' addresses and their associated city and country information.")
async def get_customer_country_difference(country1: str = Query(..., description="First country"), country2: str = Query(..., description="Second country")):
    cursor.execute("SELECT AVG(IIF(T4.country = ?, 1, 0)) - AVG(IIF(T4.country = ?, 1, 0)) AS diff FROM customer AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id INNER JOIN city AS T3 ON T2.city_id = T3.city_id INNER JOIN country AS T4 ON T3.country_id = T4.country_id", (country1, country2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the percentage of films in a specific category for a specific actor
@app.get("/v1/movie_3/actor_category_percentage", operation_id="get_actor_category_percentage", summary="Retrieves the percentage of films in a specific category that a particular actor has starred in. The calculation is based on the provided actor's first and last name, as well as the category name.")
async def get_actor_category_percentage(category_name: str = Query(..., description="Category name"), first_name: str = Query(..., description="Actor's first name"), last_name: str = Query(..., description="Actor's last name")):
    cursor.execute("SELECT CAST(SUM(IIF(T4.name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.actor_id) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film_category AS T3 ON T2.film_id = T3.film_id INNER JOIN category AS T4 ON T3.category_id = T4.category_id WHERE T1.first_name = ? AND T1.last_name = ?", (category_name, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total payment amount for a specific film title
@app.get("/v1/movie_3/total_payment_for_film", operation_id="get_total_payment_for_film", summary="Retrieves the total payment amount for a specific film title by aggregating the payment amounts associated with the film's rentals.")
async def get_total_payment_for_film(film_title: str = Query(..., description="Film title")):
    cursor.execute("SELECT SUM(T1.amount) FROM payment AS T1 INNER JOIN rental AS T2 ON T1.rental_id = T2.rental_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T4.title = ?", (film_title,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get customers who have rented more than a specified number of films
@app.get("/v1/movie_3/customers_with_rentals_above_threshold", operation_id="get_customers_with_rentals_above_threshold", summary="Retrieves a list of customers who have rented more than the specified minimum number of films. The operation filters customers based on their rental history and returns their first and last names.")
async def get_customers_with_rentals_above_threshold(rental_threshold: int = Query(..., description="Minimum number of rentals")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T1.first_name, T1.last_name, COUNT(T1.customer_id) AS num FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id INNER JOIN film_category AS T5 ON T4.film_id = T5.film_id GROUP BY T1.first_name, T1.last_name ) AS T WHERE T.num > ?", (rental_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the average actor ID for films in a specific category
@app.get("/v1/movie_3/average_actor_id_in_category", operation_id="get_average_actor_id_in_category", summary="Retrieves the average actor ID for films belonging to a specified category. The category is identified by its name, which is provided as an input parameter.")
async def get_average_actor_id_in_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT AVG(T1.actor_id) FROM film_actor AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id INNER JOIN actor AS T4 ON T4.actor_id = T1.actor_id WHERE T3.name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_actor_id": []}
    return {"average_actor_id": result[0]}

# Endpoint to get the most rented film title in a specific category
@app.get("/v1/movie_3/most_rented_film_in_category", operation_id="get_most_rented_film_in_category", summary="Retrieves the title of the film that has been rented the most in a specified category. The category is determined by the provided category name. The operation calculates the total number of rentals for each film within the category and returns the film with the highest rental count.")
async def get_most_rented_film_in_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T.title FROM ( SELECT T4.title, COUNT(T4.title) AS num FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id INNER JOIN film_category AS T5 ON T4.film_id = T5.film_id INNER JOIN category AS T6 ON T5.category_id = T6.category_id WHERE T6.name = ? GROUP BY T4.title ) AS T ORDER BY T.num DESC LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"film_title": []}
    return {"film_title": result[0]}

# Endpoint to get the percentage of customers who have paid more than the average payment amount
@app.get("/v1/movie_3/percentage_customers_above_avg_payment", operation_id="get_percentage_customers_above_avg_payment", summary="Retrieves the percentage of customers who have made payments above the average payment amount. This operation calculates the total number of customers who have paid more than the average payment amount and divides it by the total number of customers to provide a percentage.")
async def get_percentage_customers_above_avg_payment():
    cursor.execute("SELECT CAST(( SELECT COUNT(T1.customer_id) FROM customer AS T1 INNER JOIN payment AS T2 ON T1.customer_id = T2.customer_id WHERE T2.amount > ( SELECT AVG(amount) FROM payment ) ) AS REAL) * 100 / ( SELECT COUNT(customer_id) FROM customer )")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct customer names who have rented more films of one category over another
@app.get("/v1/movie_3/distinct_customer_names_by_category_preference", operation_id="get_distinct_customer_names_by_category_preference", summary="Retrieve a list of unique customer names who have rented more films from the first category than the second category. This operation compares the rental history of customers across two specified categories and returns the names of those who have a preference for the first category.")
async def get_distinct_customer_names_by_category_preference(category1: str = Query(..., description="First category name"), category2: str = Query(..., description="Second category name")):
    cursor.execute("SELECT DISTINCT IIF(SUM(IIF(T5.name = ?, 1, 0)) - SUM(IIF(T5.name = ?, 1, 0)) > 0, T1.first_name, 0) FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film_category AS T4 ON T4.film_id = T3.film_id INNER JOIN category AS T5 ON T4.category_id = T5.category_id GROUP BY T1.customer_id", (category1, category2))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get film titles based on rating
@app.get("/v1/movie_3/film_titles_by_rating", operation_id="get_film_titles_by_rating", summary="Retrieves a list of film titles that match the specified rating. The rating parameter is used to filter the results.")
async def get_film_titles_by_rating(rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT title FROM film WHERE rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of actors based on last name
@app.get("/v1/movie_3/actor_count_by_last_name", operation_id="get_actor_count_by_last_name", summary="Retrieves the total number of actors with the specified last name. The operation filters the actor records based on the provided last name and returns the count.")
async def get_actor_count_by_last_name(last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(actor_id) FROM actor WHERE last_name = ?", (last_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total payment amount for a specific month and year
@app.get("/v1/movie_3/total_payment_by_month_year", operation_id="get_total_payment_by_month_year", summary="Retrieves the total payment amount for a given month and year. The input parameter specifies the month and year in 'YYYY-MM%' format, which is used to filter the payment data and calculate the sum of the payment amounts.")
async def get_total_payment_by_month_year(month_year: str = Query(..., description="Month and year in 'YYYY-MM%' format")):
    cursor.execute("SELECT SUM(amount) FROM payment WHERE payment_date LIKE ?", (month_year,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the country based on a specific address
@app.get("/v1/movie_3/country_by_address", operation_id="get_country_by_address", summary="Retrieves the country associated with a given address. This operation uses the provided address to look up the corresponding country in the database. The address is matched with a city, which is then linked to its respective country.")
async def get_country_by_address(address: str = Query(..., description="Address")):
    cursor.execute("SELECT T1.country FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id INNER JOIN address AS T3 ON T2.city_id = T3.city_id WHERE T3.address = ?", (address,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the most common language of films released in a specific year
@app.get("/v1/movie_3/most_common_language_by_year", operation_id="get_most_common_language_by_year", summary="Retrieve the language that is most frequently used in films released in a specific year. The operation filters films by the provided release year and calculates the frequency of each language. The language with the highest frequency is then returned.")
async def get_most_common_language_by_year(release_year: str = Query(..., description="Release year in 'YYYY' format")):
    cursor.execute("SELECT T.language_id FROM ( SELECT T1.language_id, COUNT(T1.language_id) AS num FROM film AS T1 INNER JOIN language AS T2 ON T1.language_id = T2.language_id WHERE STRFTIME('%Y',T1.release_year) = ? GROUP BY T1.language_id ) AS T ORDER BY T.num DESC LIMIT 1", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"language_id": []}
    return {"language_id": result[0]}

# Endpoint to get the count of rentals by a specific customer within a date range
@app.get("/v1/movie_3/rental_count_by_customer_date_range", operation_id="get_rental_count_by_customer_date_range", summary="Retrieves the total number of rentals made by a specific customer within a given date range. The customer is identified by their first and last names, and the date range is defined by the start and end dates provided in 'YYYY-MM-DD' format.")
async def get_rental_count_by_customer_date_range(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.rental_id) FROM rental AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ? AND date(T1.rental_date) BETWEEN ? AND ?", (first_name, last_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers rented by a specific staff member
@app.get("/v1/movie_3/customer_count_by_staff", operation_id="get_customer_count_by_staff", summary="Retrieves the total number of customers who have rented from a staff member identified by their first and last names. The count is determined by matching the staff member's first and last names with the corresponding records in the rental and staff tables.")
async def get_customer_count_by_staff(first_name: str = Query(..., description="First name of the staff member"), last_name: str = Query(..., description="Last name of the staff member")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM rental AS T1 INNER JOIN staff AS T2 ON T1.staff_id = T2.staff_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total payment amount for rentals on a specific date
@app.get("/v1/movie_3/total_payment_by_date", operation_id="get_total_payment_by_date", summary="Get the total payment amount for rentals on a specific date (format: 'YYYY-MM-DD')")
async def get_total_payment_by_date(rental_date: str = Query(..., description="Rental date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(T2.amount) FROM rental AS T1 INNER JOIN payment AS T2 ON T1.rental_id = T2.rental_id WHERE date(T1.rental_date) = ?", (rental_date,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get customer first names based on the first two digits of their postal code
@app.get("/v1/movie_3/customer_first_names_by_postal_code", operation_id="get_customer_first_names_by_postal_code", summary="Retrieves the first names of customers who reside in areas with the specified postal code prefix. The postal code prefix is a two-digit string that represents the first two characters of the postal code.")
async def get_customer_first_names_by_postal_code(postal_code_prefix: str = Query(..., description="First two digits of the postal code")):
    cursor.execute("SELECT T1.first_name FROM customer AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id WHERE SUBSTR(T2.postal_code, 1, 2) = ?", (postal_code_prefix,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get rental dates for a specific film title
@app.get("/v1/movie_3/rental_dates_by_film_title", operation_id="get_rental_dates_by_film_title", summary="Retrieves the rental dates for a specific film. The operation requires the title of the film as input and returns a list of dates when the film was rented.")
async def get_rental_dates_by_film_title(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T1.rental_date FROM rental AS T1 INNER JOIN inventory AS T2 ON T1.inventory_id = T2.inventory_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.title = ?", (film_title,))
    result = cursor.fetchall()
    if not result:
        return {"rental_dates": []}
    return {"rental_dates": [row[0] for row in result]}

# Endpoint to get the count of actors in films of a specific category
@app.get("/v1/movie_3/actor_count_by_category", operation_id="get_actor_count_by_category", summary="Retrieve the total number of actors who have appeared in films belonging to a specified category. The category is identified by its name.")
async def get_actor_count_by_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT COUNT(T1.actor_id) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id INNER JOIN film_category AS T4 ON T3.film_id = T4.film_id INNER JOIN category AS T5 ON T4.category_id = T5.category_id WHERE T5.name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the actor with the most films in a specific category
@app.get("/v1/movie_3/top_actor_by_category", operation_id="get_top_actor_by_category", summary="Retrieves the actor with the highest number of films in a specified category. The category is determined by the provided category name.")
async def get_top_actor_by_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T4.first_name, T4.last_name, COUNT(T2.actor_id) AS num FROM film_category AS T1 INNER JOIN film_actor AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T1.category_id = T3.category_id INNER JOIN actor AS T4 ON T2.actor_id = T4.actor_id WHERE T3.name = ? GROUP BY T4.first_name, T4.last_name ) AS T ORDER BY T.num DESC LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"actor": []}
    return {"actor": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of customers managed by staff members not named a specific first name
@app.get("/v1/movie_3/customer_count_excluding_manager", operation_id="get_customer_count_excluding_manager", summary="Retrieves the total number of customers managed by staff members whose first name is not the specified one. This operation excludes customers managed by a staff member with the given first name.")
async def get_customer_count_excluding_manager(first_name: str = Query(..., description="First name of the manager to exclude")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM customer AS T1 INNER JOIN store AS T2 ON T1.store_id = T2.store_id INNER JOIN staff AS T3 ON T2.manager_staff_id = T3.staff_id WHERE T3.first_name != ?", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of inactive customers
@app.get("/v1/movie_3/inactive_customer_count", operation_id="get_inactive_customer_count", summary="Retrieves the total count of customers who are currently inactive. The inactivity status is determined by the provided active parameter, where 0 indicates an inactive customer.")
async def get_inactive_customer_count(active: int = Query(..., description="Active status of the customer (0 for inactive)")):
    cursor.execute("SELECT COUNT(T2.customer_id) FROM rental AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.active = ?", (active,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the title of the shortest film
@app.get("/v1/movie_3/shortest_film_title", operation_id="get_shortest_film_title", summary="Retrieves the title of the shortest film available in the database. The operation considers films from all categories and sorts them by length to determine the shortest one.")
async def get_shortest_film_title():
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id ORDER BY T1.length LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the country of a customer by first and last name
@app.get("/v1/movie_3/customer_country_by_name", operation_id="get_customer_country_by_name", summary="Retrieves the country of a customer based on their first and last name. This operation uses the provided names to search across multiple tables, including customer, store, address, city, and country, to determine the customer's country of residence.")
async def get_customer_country_by_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T5.country FROM customer AS T1 INNER JOIN store AS T2 ON T1.store_id = T2.store_id INNER JOIN address AS T3 ON T2.address_id = T3.address_id INNER JOIN city AS T4 ON T3.city_id = T4.city_id INNER JOIN country AS T5 ON T4.country_id = T5.country_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the average payment amount for a specific film category
@app.get("/v1/movie_3/average_payment_by_category", operation_id="get_average_payment_by_category", summary="Retrieves the average payment amount for a specified film category. This operation calculates the average payment by aggregating the payment amounts associated with the specified category across all relevant films, inventory, rentals, and payments. The category name is required as an input parameter.")
async def get_average_payment_by_category(category_name: str = Query(..., description="Name of the film category")):
    cursor.execute("SELECT AVG(T5.amount) FROM category AS T1 INNER JOIN film_category AS T2 ON T1.category_id = T2.category_id INNER JOIN inventory AS T3 ON T2.film_id = T3.film_id INNER JOIN rental AS T4 ON T3.inventory_id = T4.inventory_id INNER JOIN payment AS T5 ON T4.rental_id = T5.rental_id WHERE T1.name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_amount": []}
    return {"average_amount": result[0]}

# Endpoint to get the average payment amount for a specific customer by first and last name
@app.get("/v1/movie_3/average_payment_by_customer_name", operation_id="get_average_payment_by_customer_name", summary="Retrieves the average payment amount made by a customer identified by their first and last name. The calculation is based on the customer's payment history.")
async def get_average_payment_by_customer_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT AVG(T2.amount) FROM customer AS T1 INNER JOIN payment AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"average_amount": []}
    return {"average_amount": result[0]}

# Endpoint to get the percentage of films in a specific category with a length less than a specified value
@app.get("/v1/movie_3/percentage_short_films_by_category", operation_id="get_percentage_short_films_by_category", summary="Retrieves the percentage of films in a specified category that are shorter than a given length. The calculation is based on the total count of films in the category.")
async def get_percentage_short_films_by_category(length: int = Query(..., description="Maximum length of the film"), category_name: str = Query(..., description="Name of the film category")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.length < ? AND T3.name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.film_id) FROM film_category AS T1 INNER JOIN film AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T1.category_id = T3.category_id", (length, category_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the details of an actor by first name
@app.get("/v1/movie_3/actor_details_by_first_name", operation_id="get_actor_details_by_first_name", summary="Retrieves the full name of an actor based on their first name. The operation returns the first and last names of the actor whose first name matches the provided input. The input parameter specifies the first name to search for.")
async def get_actor_details_by_first_name(first_name: str = Query(..., description="First name of the actor")):
    cursor.execute("SELECT first_name, last_name FROM actor WHERE first_name = ?", (first_name,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": result}

# Endpoint to get the address IDs in a specific district
@app.get("/v1/movie_3/address_ids_by_district", operation_id="get_address_ids_by_district", summary="Retrieves a list of unique address identifiers (address_ids) located within the specified district. The district parameter is used to filter the results.")
async def get_address_ids_by_district(district: str = Query(..., description="Name of the district")):
    cursor.execute("SELECT address_id FROM address WHERE district = ?", (district,))
    result = cursor.fetchall()
    if not result:
        return {"address_ids": []}
    return {"address_ids": result}

# Endpoint to get distinct categories with a limit
@app.get("/v1/movie_3/distinct_categories_with_limit", operation_id="get_distinct_categories_with_limit", summary="Retrieves a limited number of distinct categories, including their names, unique identifiers, and last update timestamps. The limit parameter determines the maximum number of categories to return.")
async def get_distinct_categories_with_limit(limit: int = Query(..., description="Limit the number of categories returned")):
    cursor.execute("SELECT DISTINCT name, category_id, last_update FROM category LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": result}

# Endpoint to get customer names based on their active status and limit
@app.get("/v1/movie_3/customer_names_by_active_status", operation_id="get_customer_names", summary="Retrieves the first and last names of customers based on their active status, with the option to limit the number of results returned. The active status parameter determines whether to include customers who are currently active or inactive. The limit parameter allows you to specify the maximum number of customer names to return.")
async def get_customer_names(active: int = Query(..., description="Active status of the customer (0 or 1)"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT first_name, last_name FROM customer WHERE active = ? LIMIT ?", (active, limit))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the rental rate per rental duration for a specific film title
@app.get("/v1/movie_3/rental_rate_per_duration", operation_id="get_rental_rate_per_duration", summary="Retrieves the rental rate per unit of rental duration for a specific film. The operation calculates the rate by dividing the rental rate by the rental duration, providing a clear cost-per-time metric for the given film title.")
async def get_rental_rate_per_duration(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT rental_rate / rental_duration AS result FROM film WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"result": []}
    return {"result": result[0]}

# Endpoint to get the city based on a specific address
@app.get("/v1/movie_3/city_by_address", operation_id="get_city_by_address", summary="Retrieves the city associated with a given address. This operation uses the provided address to search for a matching city in the database. The result is the name of the city corresponding to the input address.")
async def get_city_by_address(address: str = Query(..., description="Address to look up the city")):
    cursor.execute("SELECT T1.city FROM city AS T1 INNER JOIN address AS T2 ON T2.city_id = T1.city_id WHERE T2.address = ?", (address,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get cities based on a specific country
@app.get("/v1/movie_3/cities_by_country", operation_id="get_cities_by_country", summary="Retrieves a list of cities located within a specified country. The operation filters the city data based on the provided country parameter, returning only the cities that match the given country.")
async def get_cities_by_country(country: str = Query(..., description="Country to look up the cities")):
    cursor.execute("SELECT T2.city FROM country AS T1 INNER JOIN city AS T2 ON T1.country_id = T2.country_id WHERE T1.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": result}

# Endpoint to get the difference in counts between two categories
@app.get("/v1/movie_3/category_count_difference", operation_id="get_category_count_difference", summary="Retrieves the difference in the number of movies between two specified categories. The operation compares the count of movies in the first category with the count in the second category and returns the difference.")
async def get_category_count_difference(category1: str = Query(..., description="First category name"), category2: str = Query(..., description="Second category name")):
    cursor.execute("SELECT SUM(IIF(T2.name = ?, 1, 0)) - SUM(IIF(T2.name = ?, 1, 0)) AS diff FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id", (category1, category2))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get the district based on customer's first and last name
@app.get("/v1/movie_3/district_by_customer_name", operation_id="get_district_by_customer_name", summary="Retrieves the district associated with a customer, based on the provided first and last name. This operation returns the district where the customer resides, as determined by matching the input names with the customer's first and last names in the database.")
async def get_district_by_customer_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T2.district FROM customer AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"district": []}
    return {"district": result[0]}

# Endpoint to get customer names based on address and active status
@app.get("/v1/movie_3/customer_names_by_address_and_active_status", operation_id="get_customer_names_by_address_and_active_status", summary="Retrieves the first and last names of customers who reside at a specified address and have a particular active status. The active status can be either 0 or 1.")
async def get_customer_names_by_address_and_active_status(address: str = Query(..., description="Address of the customer"), active: int = Query(..., description="Active status of the customer (0 or 1)")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM customer AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id WHERE T2.address = ? AND T1.active = ?", (address, active))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the count of films based on language, length, and replacement cost
@app.get("/v1/movie_3/film_count_by_language_length_cost", operation_id="get_film_count_by_language_length_cost", summary="Retrieves the number of films that meet the specified language, minimum length, and maximum replacement cost criteria. The response provides a count of films that match the input parameters, offering insights into the distribution of films based on language, length, and cost.")
async def get_film_count_by_language_length_cost(language: str = Query(..., description="Language of the film"), min_length: int = Query(..., description="Minimum length of the film"), max_replacement_cost: float = Query(..., description="Maximum replacement cost of the film")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN language AS T2 ON T1.language_id = T2.language_id WHERE T2.name = ? AND T1.length > ? AND T1.replacement_cost < ?", (language, min_length, max_replacement_cost))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of films in a specific category
@app.get("/v1/movie_3/percentage_of_films_in_category", operation_id="get_percentage_of_films_in_category", summary="Retrieves the percentage of films that belong to a specified category. The calculation is based on the total number of films and the count of films in the given category. The category is identified by its name.")
async def get_percentage_of_films_in_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.film_id) FROM film_category AS T1 INNER JOIN category AS T2 ON T1.category_id = T2.category_id", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of films based on language and rating
@app.get("/v1/movie_3/count_films_by_language_and_rating", operation_id="get_count_films_by_language_and_rating", summary="Retrieves the total number of films that match a specified language and rating. The language is identified by its name, and the rating is a value between 1 and 5.")
async def get_count_films_by_language_and_rating(language_name: str = Query(..., description="Name of the language"), rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN language AS T2 ON T1.language_id = T2.language_id WHERE T2.name = ? AND T1.rating = ?", (language_name, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of actors based on first name
@app.get("/v1/movie_3/count_actors_by_first_name", operation_id="get_count_actors_by_first_name", summary="Retrieves the total number of actors with a specified first name from the database.")
async def get_count_actors_by_first_name(first_name: str = Query(..., description="First name of the actor")):
    cursor.execute("SELECT COUNT(actor_id) FROM actor WHERE first_name = ?", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common customer first name
@app.get("/v1/movie_3/most_common_customer_first_name", operation_id="get_most_common_customer_first_name", summary="Retrieves the most frequently occurring first name among customers, up to a specified limit. This operation provides insight into the most common first names among customers, which can be useful for various analytical purposes.")
async def get_most_common_customer_first_name(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT first_name FROM customer GROUP BY first_name ORDER BY COUNT(first_name) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"first_name": []}
    return {"first_name": result[0]}

# Endpoint to get film ratings based on special features
@app.get("/v1/movie_3/film_ratings_by_special_features", operation_id="get_film_ratings_by_special_features", summary="Retrieves the ratings of films that match the specified special features. The special features parameter accepts a wildcard (%%) to broaden the search. The endpoint returns a list of ratings for films that meet the criteria.")
async def get_film_ratings_by_special_features(special_features: str = Query(..., description="Special features of the film (use %% for wildcard)")):
    cursor.execute("SELECT rating FROM film WHERE special_features LIKE ?", (special_features,))
    result = cursor.fetchall()
    if not result:
        return {"ratings": []}
    return {"ratings": [row[0] for row in result]}

# Endpoint to get the customer with the most rentals
@app.get("/v1/movie_3/customer_with_most_rentals", operation_id="get_customer_with_most_rentals", summary="Retrieves the customer who has rented the most movies, up to a specified limit. The operation returns the count of rentals for the top customer(s) in descending order.")
async def get_customer_with_most_rentals(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT COUNT(rental_id) FROM rental GROUP BY customer_id ORDER BY COUNT(rental_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get film titles based on description
@app.get("/v1/movie_3/film_titles_by_description", operation_id="get_film_titles_by_description", summary="Retrieves a list of film titles that match the provided description. The description parameter supports wildcard characters (%%) to broaden the search results.")
async def get_film_titles_by_description(description: str = Query(..., description="Description of the film (use %% for wildcard)")):
    cursor.execute("SELECT title FROM film_text WHERE description LIKE ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of rentals by a specific customer
@app.get("/v1/movie_3/count_rentals_by_customer", operation_id="get_count_rentals_by_customer", summary="Retrieves the total number of rentals made by a customer identified by their first and last names. The operation uses the provided first and last names to filter the customer records and calculate the count of associated rentals.")
async def get_count_rentals_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top manager by the number of films in inventory
@app.get("/v1/movie_3/top_manager_by_film_count", operation_id="get_top_manager_by_film_count", summary="Retrieve the top manager with the highest number of films in inventory, with the option to limit the number of results returned.")
async def get_top_manager_by_film_count(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T3.first_name, T3.last_name, COUNT(T1.film_id) AS num FROM inventory AS T1 INNER JOIN store AS T2 ON T1.store_id = T2.store_id INNER JOIN staff AS T3 ON T2.manager_staff_id = T3.staff_id GROUP BY T3.first_name, T3.last_name ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"managers": []}
    return {"managers": result}

# Endpoint to get addresses of customers based on their active status
@app.get("/v1/movie_3/customer_addresses_by_active_status", operation_id="get_customer_addresses_by_active_status", summary="Retrieves the addresses of customers based on their active status. The active status is used to filter the results, with 0 indicating inactive customers and 1 indicating active customers.")
async def get_customer_addresses_by_active_status(active: int = Query(..., description="Active status of the customer (0 for inactive, 1 for active)")):
    cursor.execute("SELECT T2.address FROM customer AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id WHERE T1.active = ?", (active,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the top customer by total payment amount
@app.get("/v1/movie_3/top_customer_by_payment_amount", operation_id="get_top_customer_by_payment_amount", summary="Retrieve the top customer(s) who have made the highest total payment amount. The number of customers returned can be limited by specifying the 'limit' parameter.")
async def get_top_customer_by_payment_amount(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T1.first_name, T1.last_name, SUM(T2.amount) AS num FROM customer AS T1 INNER JOIN payment AS T2 ON T1.customer_id = T2.customer_id GROUP BY T1.first_name, T1.last_name ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the top films by rental count
@app.get("/v1/movie_3/top_films_by_rental_count", operation_id="get_top_films_by_rental_count", summary="Retrieves a list of the top-rented films, sorted by rental count in descending order. The number of results can be limited by specifying the 'limit' parameter.")
async def get_top_films_by_rental_count(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T.title FROM ( SELECT T3.title, COUNT(T2.inventory_id) AS num FROM rental AS T1 INNER JOIN inventory AS T2 ON T1.inventory_id = T2.inventory_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id GROUP BY T3.title ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"films": []}
    return {"films": result}

# Endpoint to get all store addresses
@app.get("/v1/movie_3/store_addresses", operation_id="get_store_addresses", summary="Retrieves the addresses of all stores. This operation fetches the addresses from the address table that are associated with the stores in the store table.")
async def get_store_addresses():
    cursor.execute("SELECT T2.address FROM store AS T1 INNER JOIN address AS T2 ON T1.address_id = T2.address_id")
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the top city by the number of customers
@app.get("/v1/movie_3/top_city_by_customer_count", operation_id="get_top_city_by_customer_count", summary="Retrieves the city with the highest number of customers, based on the provided limit. The operation calculates the number of customers per city and returns the top city with the most customers, up to the specified limit.")
async def get_top_city_by_customer_count(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T.city FROM ( SELECT T1.city, COUNT(T3.customer_id) AS num FROM city AS T1 INNER JOIN address AS T2 ON T2.city_id = T1.city_id INNER JOIN customer AS T3 ON T2.address_id = T3.address_id GROUP BY T1.city ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": result}

# Endpoint to get the actor with the highest number of films
@app.get("/v1/movie_3/top_actor_by_film_count", operation_id="get_top_actor_by_film_count", summary="Retrieves the actor with the highest number of films, based on the provided limit. The actor's first and last names are returned.")
async def get_top_actor_by_film_count(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T.first_name, T.last_name FROM ( SELECT T2.first_name, T2.last_name, SUM(T1.film_id) AS num FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id GROUP BY T2.first_name, T2.last_name ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": result}

# Endpoint to get actor details by actor ID
@app.get("/v1/movie_3/actor_details_by_id", operation_id="get_actor_details_by_id", summary="Retrieves the first and last name of an actor based on the provided actor ID.")
async def get_actor_details_by_id(actor_id: int = Query(..., description="ID of the actor")):
    cursor.execute("SELECT first_name, last_name FROM actor WHERE actor_id = ?", (actor_id,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": result}

# Endpoint to get the count of films in a specific category
@app.get("/v1/movie_3/count_films_in_category", operation_id="get_count_films_in_category", summary="Retrieves the total number of films that belong to a specified category. The category is identified by its unique ID.")
async def get_count_films_in_category(category_id: int = Query(..., description="ID of the category")):
    cursor.execute("SELECT COUNT(film_id) FROM film_category WHERE category_id = ?", (category_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the actor with the highest rental rate film
@app.get("/v1/movie_3/top_actor_by_rental_rate", operation_id="get_top_actor_by_rental_rate", summary="Retrieves the actor with the highest rental rate film, based on the provided limit. This operation returns the first and last name of the actor with the highest rental rate film, as determined by the rental rate of the films they have starred in. The limit parameter allows you to specify the maximum number of results to return.")
async def get_top_actor_by_rental_rate(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T3.film_id = T2.film_id ORDER BY T3.rental_rate DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": result}

# Endpoint to get film descriptions for a specific actor
@app.get("/v1/movie_3/film_descriptions_by_actor", operation_id="get_film_descriptions_by_actor", summary="Retrieves detailed descriptions of all films in which a specific actor has appeared. The operation requires the actor's first and last names as input parameters to accurately identify and return the relevant film descriptions.")
async def get_film_descriptions_by_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.description FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T3.film_id = T2.film_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get customers who made payments above a certain amount
@app.get("/v1/movie_3/customers_by_payment_amount", operation_id="get_customers_by_payment_amount", summary="Retrieves the first and last names of customers who have made payments exceeding the specified minimum amount. The minimum payment amount is provided as an input parameter.")
async def get_customers_by_payment_amount(amount: float = Query(..., description="Minimum payment amount")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T1.amount > ?", (amount,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the address of a customer by name and postal code
@app.get("/v1/movie_3/customer_address_by_name_and_postal_code", operation_id="get_customer_address_by_name_and_postal_code", summary="Retrieves the address of a customer based on the provided first name and postal code. The operation searches for a customer with the given first name and postal code, and returns the corresponding address.")
async def get_customer_address_by_name_and_postal_code(first_name: str = Query(..., description="First name of the customer"), postal_code: int = Query(..., description="Postal code of the customer")):
    cursor.execute("SELECT T1.address FROM address AS T1 INNER JOIN customer AS T2 ON T1.address_id = T2.address_id WHERE T2.first_name = ? AND T1.postal_code = ?", (first_name, postal_code))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the count of customers in a specific city
@app.get("/v1/movie_3/count_customers_in_city", operation_id="get_count_customers_in_city", summary="Retrieves the total number of customers residing in a specified city. The city is identified by its name, which is provided as an input parameter.")
async def get_count_customers_in_city(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT COUNT(T1.city_id) FROM city AS T1 INNER JOIN address AS T2 ON T1.city_id = T2.city_id INNER JOIN customer AS T3 ON T2.address_id = T3.address_id WHERE T1.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names based on address
@app.get("/v1/movie_3/customer_names_by_address", operation_id="get_customer_names_by_address", summary="Retrieves the first and last names of customers associated with a given address. The address is provided as an input parameter.")
async def get_customer_names_by_address(address: str = Query(..., description="Address of the customer")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM address AS T1 INNER JOIN customer AS T2 ON T1.address_id = T2.address_id WHERE T1.address = ?", (address,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get film titles based on category and length
@app.get("/v1/movie_3/film_titles_by_category_and_length", operation_id="get_film_titles_by_category_and_length", summary="Retrieve the titles of films that belong to a specified category and have a duration longer than a given minimum length. The category is identified by its name, and the minimum length is provided in minutes.")
async def get_film_titles_by_category_and_length(category_name: str = Query(..., description="Category name of the film"), min_length: int = Query(..., description="Minimum length of the film")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T3.category_id = T2.category_id WHERE T3.`name` = ? AND T1.length > ?", (category_name, min_length))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the top film title based on category and rental rate
@app.get("/v1/movie_3/top_film_title_by_category", operation_id="get_top_film_title_by_category", summary="Retrieves the title of the top-rated film in a specified category, sorted by rental rate in descending order. The category is determined by the provided category name.")
async def get_top_film_title_by_category(category_name: str = Query(..., description="Category name of the film")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.`name` = ? ORDER BY T1.rental_rate LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get film descriptions based on category
@app.get("/v1/movie_3/film_descriptions_by_category", operation_id="get_film_descriptions_by_category", summary="Retrieve the descriptions of films that belong to a specified category. The category is identified by its name, which is provided as an input parameter.")
async def get_film_descriptions_by_category(category_name: str = Query(..., description="Category name of the film")):
    cursor.execute("SELECT T1.description FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.`name` = ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get the total payment amount based on district
@app.get("/v1/movie_3/total_payment_by_district", operation_id="get_total_payment_by_district", summary="Retrieves the total payment amount for a specific district. This operation calculates the sum of all payments made by customers residing in the provided district. The district parameter is used to filter the customers and their corresponding payments.")
async def get_total_payment_by_district(district: str = Query(..., description="District of the customer")):
    cursor.execute("SELECT SUM(T1.amount) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id INNER JOIN address AS T3 ON T3.address_id = T2.address_id WHERE T3.district = ?", (district,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the percentage of total payment by a specific customer
@app.get("/v1/movie_3/payment_percentage_by_customer", operation_id="get_payment_percentage_by_customer", summary="Retrieves the percentage of total payments made by a customer identified by their first and last names. This operation calculates the sum of payments made by the specified customer and divides it by the total sum of all payments, then multiplies the result by 100 to express it as a percentage.")
async def get_payment_percentage_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.first_name = ? AND T2.last_name = ?, T1.amount, 0)) AS REAL) * 100 / SUM(T1.amount) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of films in a specific category with a minimum length
@app.get("/v1/movie_3/film_percentage_by_category_and_length", operation_id="get_film_percentage_by_category_and_length", summary="Retrieves the percentage of films in a specified category that have a length greater than a given minimum. The category is identified by its name, and the minimum length is provided in minutes.")
async def get_film_percentage_by_category_and_length(category_name: str = Query(..., description="Category name of the film"), min_length: int = Query(..., description="Minimum length of the film")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.`name` = ?, 1, 0)) * 100 / COUNT(T1.film_id) AS REAL) FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T3.category_id = T2.category_id WHERE T1.length > ?", (category_name, min_length))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of films released in a specific year
@app.get("/v1/movie_3/film_count_by_release_year", operation_id="get_film_count_by_release_year", summary="Retrieves the total number of films released in a specified year. The operation filters the films based on the provided release year and returns the count.")
async def get_film_count_by_release_year(release_year: int = Query(..., description="Release year of the film")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE release_year = ?", (release_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get film titles within a range of film IDs
@app.get("/v1/movie_3/film_titles_by_id_range", operation_id="get_film_titles_by_id_range", summary="Retrieves the titles of films with IDs falling within the specified range. The range is defined by a minimum and maximum film ID, both of which are inclusive.")
async def get_film_titles_by_id_range(min_id: int = Query(..., description="Minimum film ID"), max_id: int = Query(..., description="Maximum film ID")):
    cursor.execute("SELECT title FROM film WHERE film_id BETWEEN ? AND ?", (min_id, max_id))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get film IDs with the maximum rental duration
@app.get("/v1/movie_3/max_rental_duration_films", operation_id="get_max_rental_duration_films", summary="Retrieves the identifiers of films that have the longest rental duration available in the catalog.")
async def get_max_rental_duration_films():
    cursor.execute("SELECT film_id FROM film WHERE rental_duration = ( SELECT MAX(rental_duration) FROM film )")
    result = cursor.fetchall()
    if not result:
        return {"film_ids": []}
    return {"film_ids": [row[0] for row in result]}

# Endpoint to get film titles with the maximum rental rate
@app.get("/v1/movie_3/max_rental_rate_films", operation_id="get_max_rental_rate_films", summary="Retrieves the titles of films that have the highest rental rate. This operation returns a list of film titles that share the maximum rental rate in the database.")
async def get_max_rental_rate_films():
    cursor.execute("SELECT title FROM film WHERE rental_rate = ( SELECT MAX(rental_rate) FROM film )")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the language name of a film by its title
@app.get("/v1/movie_3/film_language_by_title", operation_id="get_film_language_by_title", summary="Retrieves the name of the language associated with a specific film, based on the provided film title.")
async def get_film_language_by_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.`name` FROM film AS T1 INNER JOIN `language` AS T2 ON T1.language_id = T2.language_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get distinct last update dates of films in a specific language and release year
@app.get("/v1/movie_3/distinct_last_update_by_language_and_year", operation_id="get_distinct_last_update_by_language_and_year", summary="Retrieve the unique dates when films in a specific language and release year were last updated. This operation allows you to identify the latest update timestamps for films based on their language and year of release. The input parameters determine the language and release year for which the distinct last update dates are retrieved.")
async def get_distinct_last_update_by_language_and_year(language: str = Query(..., description="Language of the film"), release_year: int = Query(..., description="Release year of the film")):
    cursor.execute("SELECT DISTINCT T1.last_update FROM film AS T1 INNER JOIN `language` AS T2 ON T1.language_id = T2.language_id WHERE T2.`name` = ? AND T1.release_year = ?", (language, release_year))
    result = cursor.fetchall()
    if not result:
        return {"last_updates": []}
    return {"last_updates": [row[0] for row in result]}

# Endpoint to get the count of films in a specific language with specific special features
@app.get("/v1/movie_3/count_films_by_language_and_special_features", operation_id="get_count_films_by_language_and_special_features", summary="Retrieves the total number of films in a specified language that have certain special features. The language and special features are provided as input parameters.")
async def get_count_films_by_language_and_special_features(language: str = Query(..., description="Language of the film"), special_features: str = Query(..., description="Special features of the film")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN `language` AS T2 ON T1.language_id = T2.language_id WHERE T2.`name` = ? AND T1.special_features = ?", (language, special_features))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of films in a specific category with a specific rating
@app.get("/v1/movie_3/count_films_by_category_and_rating", operation_id="get_count_films_by_category_and_rating", summary="Retrieves the total number of films that belong to a specified category and have a certain rating. The category and rating are provided as input parameters.")
async def get_count_films_by_category_and_rating(category: str = Query(..., description="Category of the film"), rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT COUNT(T1.title) FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ? AND T1.rating = ?", (category, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get film IDs in a specific category
@app.get("/v1/movie_3/film_ids_by_category", operation_id="get_film_ids_by_category", summary="Retrieves a list of film IDs that belong to a specified category. The category is provided as an input parameter, allowing the user to filter the results based on their desired category.")
async def get_film_ids_by_category(category: str = Query(..., description="Category of the film")):
    cursor.execute("SELECT T1.film_id FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T3.category_id = T2.category_id WHERE T3.name = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"film_ids": []}
    return {"film_ids": [row[0] for row in result]}

# Endpoint to get the longest film title in a specific category
@app.get("/v1/movie_3/longest_film_title_by_category", operation_id="get_longest_film_title_by_category", summary="Retrieves the longest film title from the specified category. The operation filters films by the provided category and sorts them by length in descending order to identify the longest title.")
async def get_longest_film_title_by_category(category: str = Query(..., description="Category of the film")):
    cursor.execute("SELECT T1.title FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T3.category_id = T2.category_id WHERE T3.name = ? ORDER BY T1.length DESC LIMIT 1", (category,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the category name of a film by its title
@app.get("/v1/movie_3/film_category_by_title", operation_id="get_film_category_by_title", summary="Retrieves the category name associated with a specific film, based on the provided film title.")
async def get_film_category_by_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T3.name FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T3.category_id = T2.category_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get rental IDs by customer's first and last name
@app.get("/v1/movie_3/rental_ids_by_customer_name", operation_id="get_rental_ids_by_customer_name", summary="Retrieves a list of rental IDs associated with a customer, identified by their first and last name. The operation searches for the customer in the database using the provided first and last names, and returns the corresponding rental IDs.")
async def get_rental_ids_by_customer_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T2.rental_id FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"rental_ids": []}
    return {"rental_ids": [row[0] for row in result]}

# Endpoint to get distinct customer names by staff ID
@app.get("/v1/movie_3/customer_names_by_staff_id", operation_id="get_customer_names_by_staff_id", summary="Retrieves a list of unique customer names associated with the specified staff ID. This operation returns the first and last names of customers who have rentals linked to the given staff member.")
async def get_customer_names_by_staff_id(staff_id: int = Query(..., description="Staff ID")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T2.staff_id = ?", (staff_id,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get distinct customer emails by staff ID
@app.get("/v1/movie_3/customer_emails_by_staff_id", operation_id="get_customer_emails_by_staff_id", summary="Retrieves a list of unique customer email addresses associated with the specified staff member. This operation filters customer emails based on the provided staff ID, ensuring that only emails linked to rentals handled by that staff member are returned.")
async def get_customer_emails_by_staff_id(staff_id: int = Query(..., description="Staff ID")):
    cursor.execute("SELECT DISTINCT T1.email FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T2.staff_id = ?", (staff_id,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get actor IDs by film title
@app.get("/v1/movie_3/actor_ids_by_film_title", operation_id="get_actor_ids_by_film_title", summary="Retrieves the IDs of actors who have appeared in a film with the specified title. The input parameter is the title of the film.")
async def get_actor_ids_by_film_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.actor_id FROM film AS T1 INNER JOIN film_actor AS T2 ON T1.film_id = T2.film_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"actor_ids": []}
    return {"actor_ids": [row[0] for row in result]}

# Endpoint to get inventory IDs by customer's first and last name
@app.get("/v1/movie_3/inventory_ids_by_customer_name", operation_id="get_inventory_ids_by_customer_name", summary="Retrieves the inventory IDs associated with a customer, identified by their first and last name. This operation returns a list of inventory IDs that the specified customer has rented. The customer's first and last name are used as input parameters to filter the results.")
async def get_inventory_ids_by_customer_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T2.inventory_id FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"inventory_ids": []}
    return {"inventory_ids": [row[0] for row in result]}

# Endpoint to get the total rental rate for films in a specific category
@app.get("/v1/movie_3/total_rental_rate_by_category", operation_id="get_total_rental_rate_by_category", summary="Retrieves the cumulative rental rate for all films belonging to a specified category. The category is identified by its name.")
async def get_total_rental_rate_by_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT SUM(T1.rental_rate) FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_rental_rate": []}
    return {"total_rental_rate": result[0]}

# Endpoint to get the average rental rate for films in a specific category
@app.get("/v1/movie_3/average_rental_rate_by_category", operation_id="get_average_rental_rate_by_category", summary="Retrieves the average rental rate for films belonging to a specified category. The category is identified by its name, which is provided as an input parameter.")
async def get_average_rental_rate_by_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT AVG(T1.rental_rate) FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T3.name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_rental_rate": []}
    return {"average_rental_rate": result[0]}

# Endpoint to get the percentage of films in a specific category and language
@app.get("/v1/movie_3/percentage_films_category_language", operation_id="get_percentage_films_category_language", summary="Retrieves the percentage of films that belong to a specific category and are in a certain language. The calculation is based on the total number of films in the database.")
async def get_percentage_films_category_language(category_name: str = Query(..., description="Name of the category"), language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.film_id) FROM film_category AS T1 INNER JOIN film AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T1.category_id = T3.category_id INNER JOIN language AS T4 ON T2.language_id = T4.language_id WHERE T4.name = ?", (category_name, language_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of films with a specific rating and rental duration
@app.get("/v1/movie_3/count_films_rating_rental_duration", operation_id="get_count_films_rating_rental_duration", summary="Retrieves the total number of films that match a given rating and have a rental duration less than a specified value.")
async def get_count_films_rating_rental_duration(rating: str = Query(..., description="Rating of the film"), rental_duration: int = Query(..., description="Rental duration of the film")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE rating = ? AND rental_duration < ?", (rating, rental_duration))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the titles of films with specific replacement cost, rating, and length
@app.get("/v1/movie_3/film_titles_replacement_cost_rating_length", operation_id="get_film_titles_replacement_cost_rating_length", summary="Retrieve the titles of films that match the provided replacement cost, rating, and length. This operation allows you to filter films based on these criteria, providing a targeted list of film titles.")
async def get_film_titles_replacement_cost_rating_length(replacement_cost: float = Query(..., description="Replacement cost of the film"), rating: str = Query(..., description="Rating of the film"), length: int = Query(..., description="Length of the film")):
    cursor.execute("SELECT title FROM film WHERE replacement_cost = ? AND rating = ? AND length = ?", (replacement_cost, rating, length))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the emails of active customers who rented films within a specific date range
@app.get("/v1/movie_3/customer_emails_rental_date_range", operation_id="get_customer_emails_rental_date_range", summary="Retrieves the email addresses of customers who rented films during a specified date range and are currently active. The start and end dates of the rental period are provided in 'YYYY-MM-DD HH:MM:SS' format. Only customers with an active status of 1 are included in the results.")
async def get_customer_emails_rental_date_range(start_date: str = Query(..., description="Start date of the rental period in 'YYYY-MM-DD HH:MM:SS' format"), end_date: str = Query(..., description="End date of the rental period in 'YYYY-MM-DD HH:MM:SS' format"), active: int = Query(..., description="Active status of the customer (1 for active)")):
    cursor.execute("SELECT T2.email FROM rental AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T1.rental_date BETWEEN ? AND ? AND T2.active = ?", (start_date, end_date, active))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the total payment amount for a specific customer
@app.get("/v1/movie_3/total_payment_customer", operation_id="get_total_payment_customer", summary="Retrieves the total payment amount made by a customer, identified by their first and last names. This operation calculates the sum of all payment amounts associated with the customer's rentals.")
async def get_total_payment_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT SUM(T3.amount) FROM rental AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id INNER JOIN payment AS T3 ON T1.rental_id = T3.rental_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the count of customers who made payments within a specific date range
@app.get("/v1/movie_3/count_customers_payment_date_range", operation_id="get_count_customers_payment_date_range", summary="Retrieves the total number of customers who have made payments within a specified date range. The date range is defined by the start and end dates provided as input parameters. The count is determined by cross-referencing payment and customer data.")
async def get_count_customers_payment_date_range(start_date: str = Query(..., description="Start date of the payment period in 'YYYY-MM-DD HH:MM:SS' format"), end_date: str = Query(..., description="End date of the payment period in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T1.payment_date BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the actors of a specific film
@app.get("/v1/movie_3/actors_of_film", operation_id="get_actors_of_film", summary="Retrieves the first and last names of all actors who have appeared in a specific film. The film is identified by its title, which is provided as an input parameter.")
async def get_actors_of_film(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T1.film_id = T3.film_id WHERE T3.title = ?", (film_title,))
    result = cursor.fetchall()
    if not result:
        return {"actors": []}
    return {"actors": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the film titles of a specific actor
@app.get("/v1/movie_3/film_titles_of_actor", operation_id="get_film_titles_of_actor", summary="Retrieves the titles of all films in which a specific actor has appeared. The operation requires the first and last name of the actor to accurately identify and return the corresponding film titles.")
async def get_film_titles_of_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T1.film_id = T3.film_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the film details with rental duration and rate product greater than a specific value
@app.get("/v1/movie_3/film_details_rental_duration_rate", operation_id="get_film_details_rental_duration_rate", summary="Retrieves film details, including title, category name, and special features, for films with a rental duration and rate product greater than the provided value. This operation is useful for identifying films with specific rental cost characteristics.")
async def get_film_details_rental_duration_rate(product_value: float = Query(..., description="Product of rental duration and rental rate")):
    cursor.execute("SELECT T1.title, T3.name, T1.special_features FROM film AS T1 INNER JOIN film_category AS T2 ON T1.film_id = T2.film_id INNER JOIN category AS T3 ON T2.category_id = T3.category_id WHERE T1.rental_duration * T1.rental_rate > ?", (product_value,))
    result = cursor.fetchall()
    if not result:
        return {"film_details": []}
    return {"film_details": [{"title": row[0], "category": row[1], "special_features": row[2]} for row in result]}

# Endpoint to get the most recently rented film by a customer
@app.get("/v1/movie_3/most_recent_rented_film_by_customer", operation_id="get_most_recent_rented_film_by_customer", summary="Retrieves the title of the most recent film rented by a customer identified by their first and last names. The operation searches across customer, rental, inventory, and film tables to find the relevant film title.")
async def get_most_recent_rented_film_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T4.title FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T1.first_name = ? AND T1.last_name = ? ORDER BY T2.rental_date DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"film_title": []}
    return {"film_title": result[0]}

# Endpoint to get the count of films in a specific category
@app.get("/v1/movie_3/film_count_by_category", operation_id="get_film_count_by_category", summary="Retrieves the total number of films associated with a specified category. The category is identified by its name, which is provided as an input parameter.")
async def get_film_count_by_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT COUNT(T2.film_id) FROM category AS T1 INNER JOIN film_category AS T2 ON T1.category_id = T2.category_id WHERE T1.name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get store details and rental rate for a specific film
@app.get("/v1/movie_3/store_details_and_rental_rate_by_film_title", operation_id="get_store_details_and_rental_rate_by_film_title", summary="Retrieves the store details and rental rate for a specific film. The operation returns the store ID, address, and rental rate associated with the provided film title. The input parameter is the title of the film.")
async def get_store_details_and_rental_rate_by_film_title(film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.store_id, T1.address, T4.rental_rate FROM address AS T1 INNER JOIN store AS T2 ON T1.address_id = T2.address_id INNER JOIN inventory AS T3 ON T2.store_id = T3.store_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T4.title = ?", (film_title,))
    result = cursor.fetchall()
    if not result:
        return {"store_details": []}
    return {"store_details": result}

# Endpoint to get the rental duration for a specific customer and film
@app.get("/v1/movie_3/rental_duration_by_customer_and_film", operation_id="get_rental_duration_by_customer_and_film", summary="Retrieves the duration of a rental for a specific film and customer. The operation calculates the difference between the rental and return dates for the specified customer and film, providing insights into the rental period.")
async def get_rental_duration_by_customer_and_film(first_name: str = Query(..., description="First name of the customer"), film_title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.rental_date - T2.return_date FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id INNER JOIN inventory AS T3 ON T2.inventory_id = T3.inventory_id INNER JOIN film AS T4 ON T3.film_id = T4.film_id WHERE T1.first_name = ? AND T4.title = ?", (first_name, film_title))
    result = cursor.fetchone()
    if not result:
        return {"rental_duration": []}
    return {"rental_duration": result[0]}

# Endpoint to get the count of films acted by a specific actor
@app.get("/v1/movie_3/film_count_by_actor", operation_id="get_film_count_by_actor", summary="Retrieves the total number of films in which a specific actor has appeared. The actor is identified by their first and last names.")
async def get_film_count_by_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id AND T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the title of the shortest film with the highest rental duration and rate
@app.get("/v1/movie_3/shortest_film_with_highest_rental_duration_rate", operation_id="get_shortest_film_with_highest_rental_duration_rate", summary="Retrieves the title of the shortest film that has the highest rental duration and rate. The film's length is compared with the minimum length of all films, and the result is ordered by the product of rental duration and rate in descending order. Only the top result is returned.")
async def get_shortest_film_with_highest_rental_duration_rate():
    cursor.execute("SELECT title FROM film WHERE length = ( SELECT MIN(length) FROM film ) ORDER BY rental_duration * rental_rate DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"film_title": []}
    return {"film_title": result[0]}

# Endpoint to get the total payment amount for a specific customer in a specific month
@app.get("/v1/movie_3/total_payment_by_customer_and_month", operation_id="get_total_payment_by_customer_and_month", summary="Retrieves the total payment amount made by a specific customer during a particular month. The operation requires the customer's first and last names, as well as the month and year in 'YYYY-MM' format. The result is a summation of all payment amounts associated with the specified customer and month.")
async def get_total_payment_by_customer_and_month(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), month_year: str = Query(..., description="Month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT SUM(T1.amount) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ? AND SUBSTR(T1.payment_date, 1, 7) = ?", (first_name, last_name, month_year))
    result = cursor.fetchone()
    if not result:
        return {"total_payment": []}
    return {"total_payment": result[0]}

# Endpoint to get the average replacement cost for films with a specific rental rate
@app.get("/v1/movie_3/average_replacement_cost_by_rental_rate", operation_id="get_average_replacement_cost_by_rental_rate", summary="Retrieves the average replacement cost for films that have a specified rental rate. The rental rate is provided as an input parameter, allowing the calculation to be tailored to a particular rate.")
async def get_average_replacement_cost_by_rental_rate(rental_rate: float = Query(..., description="Rental rate of the film")):
    cursor.execute("SELECT AVG(replacement_cost) FROM film WHERE rental_rate = ?", (rental_rate,))
    result = cursor.fetchone()
    if not result:
        return {"average_replacement_cost": []}
    return {"average_replacement_cost": result[0]}

# Endpoint to get the average rental rate of films based on rating
@app.get("/v1/movie_3/average_rental_rate_by_rating", operation_id="get_average_rental_rate", summary="Retrieves the average rental rate for films that have a specified rating. The rating parameter is used to filter the films and calculate the average rental rate.")
async def get_average_rental_rate(rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT AVG(rental_rate) FROM film WHERE rating = ?", (rating,))
    result = cursor.fetchone()
    if not result:
        return {"average_rental_rate": []}
    return {"average_rental_rate": result[0]}

# Endpoint to get the rental duration of a film by title
@app.get("/v1/movie_3/rental_duration_by_title", operation_id="get_rental_duration", summary="Retrieves the rental duration of a film based on its title. The title is provided as an input parameter.")
async def get_rental_duration(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT rental_duration FROM film WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"rental_duration": []}
    return {"rental_duration": result[0]}

# Endpoint to get the count of distinct categories
@app.get("/v1/movie_3/count_distinct_categories", operation_id="get_count_distinct_categories", summary="Retrieves the total number of unique categories available in the database.")
async def get_count_distinct_categories():
    cursor.execute("SELECT COUNT(DISTINCT category_id) FROM category")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of rentals and percentage of rentals in July for a specific customer in a specific year
@app.get("/v1/movie_3/rental_count_and_percentage_july", operation_id="get_rental_count_and_percentage_july", summary="Retrieves the total number of rentals and the percentage of rentals made in July for a specific customer in a given year. The calculation is based on the customer's first and last name, and the year of the rentals.")
async def get_rental_count_and_percentage_july(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.rental_id), CAST(SUM(IIF(STRFTIME('%m',T2.rental_date) = '7', 1, 0)) AS REAL) * 100 / COUNT(T2.rental_id) FROM customer AS T1 INNER JOIN rental AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ? AND STRFTIME('%Y',T2.rental_date) = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"rental_count": [], "percentage_july": []}
    return {"rental_count": result[0], "percentage_july": result[1]}

# Endpoint to get film titles with rental duration greater than a specified value and rented by more than a specified number of customers
@app.get("/v1/movie_3/film_titles_by_rental_duration_and_customer_count", operation_id="get_film_titles_by_rental_duration_and_customer_count", summary="Retrieves the titles of films that have been rented for a duration exceeding the specified minimum and by more than the specified number of customers. This operation provides insights into popular films based on rental duration and customer count.")
async def get_film_titles_by_rental_duration_and_customer_count(rental_duration: int = Query(..., description="Minimum rental duration"), customer_count: int = Query(..., description="Minimum number of customers who rented the film")):
    cursor.execute("SELECT T.title FROM ( SELECT T1.title, COUNT(T3.customer_id) AS num FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id INNER JOIN rental AS T3 ON T2.inventory_id = T3.inventory_id WHERE T1.rental_duration > ? GROUP BY T1.title ) AS T WHERE T.num > ?", (rental_duration, customer_count))
    result = cursor.fetchall()
    if not result:
        return {"film_titles": []}
    return {"film_titles": [row[0] for row in result]}

# Endpoint to get film titles based on actor's name and film rating
@app.get("/v1/movie_3/film_titles_by_actor_name_and_rating", operation_id="get_film_titles_by_actor_name_and_rating", summary="Retrieves the titles of films in which a specific actor, identified by their first and last name, has appeared and that have a particular rating. The operation filters the films based on the provided actor's name and rating, and returns the corresponding film titles.")
async def get_film_titles_by_actor_name_and_rating(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor"), rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? AND T3.rating = ?", (first_name, last_name, rating))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the most rented film title
@app.get("/v1/movie_3/most_rented_film_title", operation_id="get_most_rented_film_title", summary="Retrieves the title of the film that has been rented the most frequently. This operation calculates the rental count for each film and returns the title of the film with the highest count.")
async def get_most_rented_film_title():
    cursor.execute("SELECT T.title FROM ( SELECT T3.title, COUNT(T1.customer_id) AS num FROM rental AS T1 INNER JOIN inventory AS T2 ON T1.inventory_id = T2.inventory_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id GROUP BY T3.title ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get film IDs based on actor's first and last name
@app.get("/v1/movie_3/film_ids_by_actor_name", operation_id="get_film_ids_by_actor_name", summary="Retrieves a list of film IDs associated with the specified actor. The actor is identified by their first and last names, which are used to search the actor database. The returned film IDs correspond to movies in which the actor has appeared.")
async def get_film_ids_by_actor_name(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T1.film_id FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"film_ids": []}
    return {"film_ids": [row[0] for row in result]}

# Endpoint to get actor last names based on film rating
@app.get("/v1/movie_3/actor_last_names_by_film_rating", operation_id="get_actor_last_names_by_film_rating", summary="Retrieves the last names of all actors who have appeared in films with the specified rating. The rating is a required input parameter.")
async def get_actor_last_names_by_film_rating(rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT T1.last_name FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T3.rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the average rental rate of films based on actor's first and last name
@app.get("/v1/movie_3/average_rental_rate_by_actor_name", operation_id="get_average_rental_rate_by_actor_name", summary="Retrieves the average rental rate of films in which a specific actor, identified by their first and last name, has appeared. The calculation is based on the rental rates of all films associated with the actor.")
async def get_average_rental_rate_by_actor_name(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT AVG(T3.rental_rate) FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"average_rental_rate": []}
    return {"average_rental_rate": result[0]}

# Endpoint to get the count of films based on length range
@app.get("/v1/movie_3/film_count_by_length_range", operation_id="get_film_count_by_length_range", summary="Retrieves the total number of films that have lengths falling within the specified range. The range is defined by a minimum and maximum length, which are provided as input parameters.")
async def get_film_count_by_length_range(min_length: int = Query(..., description="Minimum length of the film"), max_length: int = Query(..., description="Maximum length of the film")):
    cursor.execute("SELECT COUNT(film_id) FROM film WHERE length BETWEEN ? AND ?", (min_length, max_length))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of active customers based on first name
@app.get("/v1/movie_3/active_customer_count_by_first_name", operation_id="get_active_customer_count_by_first_name", summary="Retrieves the number of active customers with a given first name. The operation requires the first name and the active status as input parameters. The active status indicates whether the customer is currently active (1) or inactive (0).")
async def get_active_customer_count_by_first_name(first_name: str = Query(..., description="First name of the customer"), active: int = Query(..., description="Active status of the customer (1 for active, 0 for inactive)")):
    cursor.execute("SELECT COUNT(customer_id) FROM customer WHERE first_name = ? AND active = ?", (first_name, active))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of films in a specific store with a specific rating
@app.get("/v1/movie_3/film_count_by_store_and_rating", operation_id="get_film_count_by_store_and_rating", summary="Retrieves the total number of films in a specified store that have a particular rating. The operation requires the store's unique identifier and the desired rating as input parameters.")
async def get_film_count_by_store_and_rating(store_id: int = Query(..., description="ID of the store"), rating: str = Query(..., description="Rating of the film")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T2.store_id = ? AND T1.rating = ?", (store_id, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get store IDs based on actor's name and film length
@app.get("/v1/movie_3/store_ids_by_actor_and_film_length", operation_id="get_store_ids_by_actor_and_film_length", summary="Retrieve the store IDs of films that feature a specific actor and have a duration less than a given value. The operation requires the maximum length of the film, the first name of the actor, and the last name of the actor as input parameters. The result is a list of store IDs where these films are available.")
async def get_store_ids_by_actor_and_film_length(max_length: int = Query(..., description="Maximum length of the film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T4.store_id FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id INNER JOIN inventory AS T4 ON T3.film_id = T4.film_id WHERE T3.length < ? AND T1.first_name = ? AND T1.last_name = ?", (max_length, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"store_ids": []}
    return {"store_ids": [row[0] for row in result]}

# Endpoint to get the highest rental rate film title for a specific actor
@app.get("/v1/movie_3/highest_rental_rate_film_by_actor", operation_id="get_highest_rental_rate_film_by_actor", summary="Retrieves the title of the film with the highest rental rate associated with a specific actor. The operation requires the first and last name of the actor as input parameters to identify the relevant film.")
async def get_highest_rental_rate_film_by_actor(first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT T3.title FROM actor AS T1 INNER JOIN film_actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T2.film_id = T3.film_id WHERE T1.first_name = ? AND T1.last_name = ? ORDER BY T3.rental_rate DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get store IDs based on film title
@app.get("/v1/movie_3/store_ids_by_film_title", operation_id="get_store_ids_by_film_title", summary="Retrieves the store IDs associated with a given film title. The operation searches for the film by its title and returns the corresponding store IDs where the film is available.")
async def get_store_ids_by_film_title(title: str = Query(..., description="Title of the film")):
    cursor.execute("SELECT T2.store_id FROM film AS T1 INNER JOIN inventory AS T2 ON T1.film_id = T2.film_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"store_ids": []}
    return {"store_ids": [row[0] for row in result]}

# Endpoint to get the count of films based on rental rate and actor's name
@app.get("/v1/movie_3/film_count_by_rental_rate_and_actor", operation_id="get_film_count_by_rental_rate_and_actor", summary="Retrieves the number of films with a specified rental rate that feature a particular actor, identified by their first and last names.")
async def get_film_count_by_rental_rate_and_actor(rental_rate: float = Query(..., description="Rental rate of the film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT COUNT(T1.film_id) FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T1.film_id = T3.film_id WHERE T3.rental_rate = ? AND T2.first_name = ? AND T2.last_name = ?", (rental_rate, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in film counts for two specific films featuring a specific actor
@app.get("/v1/movie_3/film_count_diff_by_actor", operation_id="get_film_count_diff_by_actor", summary="Retrieve the difference in inventory counts of two specified films in which a particular actor has appeared. The operation requires the first and last names of the actor, as well as the IDs of the two films.")
async def get_film_count_diff_by_actor(film_id_1: int = Query(..., description="ID of the first film"), film_id_2: int = Query(..., description="ID of the second film"), first_name: str = Query(..., description="First name of the actor"), last_name: str = Query(..., description="Last name of the actor")):
    cursor.execute("SELECT SUM(IIF(T4.film_id = ?, 1, 0)) - SUM(IIF(T4.film_id = ?, 1, 0)) AS diff FROM film_actor AS T1 INNER JOIN actor AS T2 ON T1.actor_id = T2.actor_id INNER JOIN film AS T3 ON T1.film_id = T3.film_id INNER JOIN inventory AS T4 ON T3.film_id = T4.film_id WHERE T2.first_name = ? AND T2.last_name = ?", (film_id_1, film_id_2, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get the postal code based on the address
@app.get("/v1/movie_3/postal_code_by_address", operation_id="get_postal_code", summary="Retrieves the postal code associated with a specific address. The operation uses the provided address to search for a matching entry in the database and returns the corresponding postal code.")
async def get_postal_code(address: str = Query(..., description="Address to look up the postal code")):
    cursor.execute("SELECT postal_code FROM address WHERE address = ?", (address,))
    result = cursor.fetchone()
    if not result:
        return {"postal_code": []}
    return {"postal_code": result[0]}

# Endpoint to get the count of active customers in a specific city
@app.get("/v1/movie_3/active_customer_count_by_city", operation_id="get_active_customer_count_by_city", summary="Retrieves the count of active customers residing in a specified city. The operation filters customers based on their active status and city of residence.")
async def get_active_customer_count_by_city(active: int = Query(..., description="Active status of the customer (1 for active)"), city: str = Query(..., description="City to filter customers")):
    cursor.execute("SELECT COUNT(T2.customer_id) FROM address AS T1 INNER JOIN customer AS T2 ON T1.address_id = T2.address_id INNER JOIN city AS T3 ON T1.city_id = T3.city_id WHERE T2.active = ? AND T3.city = ?", (active, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of customers in a specific country
@app.get("/v1/movie_3/customer_names_by_country", operation_id="get_customer_names_by_country", summary="Retrieves the first and last names of customers residing in a specified country. The operation filters customers based on the provided country parameter, ensuring that only customers from the designated country are returned.")
async def get_customer_names_by_country(country: str = Query(..., description="Country to filter customers")):
    cursor.execute("SELECT T4.first_name, T4.last_name FROM address AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN country AS T3 ON T2.country_id = T3.country_id INNER JOIN customer AS T4 ON T1.address_id = T4.address_id WHERE T3.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the country of a customer by their first and last name
@app.get("/v1/movie_3/customer_country", operation_id="get_customer_country", summary="Retrieves the country of a customer based on their first and last names. This operation uses the provided names to search for the customer in the database and returns the associated country.")
async def get_customer_country(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T3.country FROM address AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN country AS T3 ON T2.country_id = T3.country_id INNER JOIN customer AS T4 ON T1.address_id = T4.address_id WHERE T4.first_name = ? AND T4.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the highest payment amount by a customer
@app.get("/v1/movie_3/highest_payment_amount", operation_id="get_highest_payment_amount", summary="Retrieves the highest payment amount made by a customer identified by their first and last names. The operation filters payments by the specified customer and returns the maximum payment amount.")
async def get_highest_payment_amount(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T1.amount FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ? ORDER BY T1.amount DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"amount": []}
    return {"amount": result[0]}

# Endpoint to get the count of payments made by a customer
@app.get("/v1/movie_3/payment_count_by_customer", operation_id="get_payment_count_by_customer", summary="Retrieves the total number of payments made by a customer identified by their first and last names. The endpoint uses the provided first and last names to filter the customer records and calculate the payment count.")
async def get_payment_count_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total payment amount by a customer
@app.get("/v1/movie_3/total_payment_amount_by_customer", operation_id="get_total_payment_amount_by_customer", summary="Retrieves the total payment amount made by a customer, identified by their first and last names. The operation calculates the sum of all payments associated with the specified customer.")
async def get_total_payment_amount_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT SUM(T1.amount) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the count of payments made by a customer in a specific year and month
@app.get("/v1/movie_3/payment_count_by_customer_year_month", operation_id="get_payment_count_by_customer_year_month", summary="Get the count of payments made by a customer in a specific year and month")
async def get_payment_count_by_customer_year_month(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), year: str = Query(..., description="Year in 'YYYY' format"), month: str = Query(..., description="Month in 'MM' format")):
    cursor.execute("SELECT COUNT(T1.customer_id) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ? AND STRFTIME('%Y',T1.payment_date) = ? AND STRFTIME('%m', T1.payment_date) = ?", (first_name, last_name, year, month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of the customer with the highest payment amount
@app.get("/v1/movie_3/customer_with_highest_payment", operation_id="get_customer_with_highest_payment", summary="Retrieves the full name of the customer who has made the highest payment. This operation fetches the first and last name of the customer with the maximum payment amount from the database.")
async def get_customer_with_highest_payment():
    cursor.execute("SELECT T2.first_name, T2.last_name FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id ORDER BY T1.amount DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the total payment amount for customers in a specific country
@app.get("/v1/movie_3/total_payment_by_country", operation_id="get_total_payment_by_country", summary="Retrieves the total payment amount made by customers residing in a specified country. The operation calculates the sum of all payments made by customers who live in the provided country.")
async def get_total_payment_by_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT SUM(T5.amount) FROM address AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN country AS T3 ON T2.country_id = T3.country_id INNER JOIN customer AS T4 ON T1.address_id = T4.address_id INNER JOIN payment AS T5 ON T4.customer_id = T5.customer_id WHERE T3.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"total_payment": []}
    return {"total_payment": result[0]}

# Endpoint to get the count of payments made by a customer with a specific first name, last name, and amount greater than a specified value
@app.get("/v1/movie_3/payment_count_by_customer_name_and_amount", operation_id="get_payment_count_by_customer_name_and_amount", summary="Retrieve the total number of payments made by a customer with a given first name, last name, and payment amount exceeding a specified minimum value. This operation provides a count of qualifying payments, facilitating analysis of customer spending patterns.")
async def get_payment_count_by_customer_name_and_amount(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), min_amount: float = Query(..., description="Minimum payment amount")):
    cursor.execute("SELECT COUNT(T1.amount) FROM payment AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ? AND T1.amount > ?", (first_name, last_name, min_amount))
    result = cursor.fetchone()
    if not result:
        return {"payment_count": []}
    return {"payment_count": result[0]}

# Endpoint to get the average payment amount for customers in a specific country
@app.get("/v1/movie_3/average_payment_by_country", operation_id="get_average_payment_by_country", summary="Retrieves the average payment amount for customers residing in a specified country. This operation calculates the average payment by aggregating the payment amounts of customers who live in the provided country. The country is identified by its name.")
async def get_average_payment_by_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT AVG(T5.amount) FROM address AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN country AS T3 ON T2.country_id = T3.country_id INNER JOIN customer AS T4 ON T1.address_id = T4.address_id INNER JOIN payment AS T5 ON T4.customer_id = T5.customer_id WHERE T3.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"average_payment": []}
    return {"average_payment": result[0]}

api_calls = [
    "/v1/movie_3/film_description_by_title?title=ACADEMY%20DINOSAUR",
    "/v1/movie_3/film_count_by_rental_duration?rental_duration=6",
    "/v1/movie_3/film_titles_by_release_year_and_rental_rate?release_year=2006&rental_rate=2.99",
    "/v1/movie_3/longest_film_title",
    "/v1/movie_3/highest_replacement_cost_film_title?title1=ACE%20GOLDFINGER&title2=ACADEMY%20DINOSAUR",
    "/v1/movie_3/film_count_by_rating_and_release_year?rating=NC-17&release_year=2006",
    "/v1/movie_3/film_count_by_rental_rate_and_special_feature?rental_rate=2.99&special_feature=Deleted%20Scenes",
    "/v1/movie_3/film_titles_with_more_than_two_special_features",
    "/v1/movie_3/staff_email_by_name?first_name=Jon&last_name=Stephens",
    "/v1/movie_3/active_staff_names?active=1",
    "/v1/movie_3/distinct_release_years_highest_replacement_cost",
    "/v1/movie_3/titles_highest_replacement_cost?limit=3",
    "/v1/movie_3/language_name_by_film_title?title=ACADEMY%20DINOSAUR",
    "/v1/movie_3/count_films_by_language?language_name=English",
    "/v1/movie_3/film_titles_by_actor_name?first_name=PENELOPE&last_name=GUINESS",
    "/v1/movie_3/count_actors_by_film_title?title=ACADEMY%20DINOSAUR",
    "/v1/movie_3/actor_names_by_film_title?title=ACADEMY%20DINOSAUR",
    "/v1/movie_3/count_films_by_actor_name_and_release_year?release_year=2006&first_name=PENELOPE&last_name=GUINESS",
    "/v1/movie_3/film_title_highest_replacement_cost_by_actor?first_name=PENELOPE&last_name=GUINESS",
    "/v1/movie_3/actor_highest_replacement_cost_film",
    "/v1/movie_3/count_films_actor_language?language=English&first_name=PENELOPE&last_name=GUINESS",
    "/v1/movie_3/longest_film_title_actor?first_name=PENELOPE&last_name=GUINESS",
    "/v1/movie_3/film_titles_by_category?category=Horror",
    "/v1/movie_3/count_films_by_category?category=Horror",
    "/v1/movie_3/film_titles_by_category_rental_rate?category=Horror&rental_rate=2.99",
    "/v1/movie_3/count_rentals_customer?first_name=RUTH&last_name=MARTINEZ",
    "/v1/movie_3/film_titles_rented_customer?first_name=RUTH&last_name=MARTINEZ",
    "/v1/movie_3/count_customers_rented_films_year?release_year=2006&first_name=RUTH&last_name=MARTINEZ",
    "/v1/movie_3/most_expensive_film_rented_customer?first_name=RUTH&last_name=MARTINEZ",
    "/v1/movie_3/customer_most_expensive_film",
    "/v1/movie_3/customer_count_by_name_return_date?first_name=RUTH&last_name=MARTINEZ&return_month=08&return_year=2005",
    "/v1/movie_3/top_customer_by_rentals",
    "/v1/movie_3/active_customer_count_by_film?active=1&film_title=ACADEMY%20DINOSAUR",
    "/v1/movie_3/most_rented_film",
    "/v1/movie_3/top_customer_by_rentals_among_names?first_name_1=RUTH&last_name_1=MARTINEZ&first_name_2=LINDA&last_name_2=WILLIAMS",
    "/v1/movie_3/top_film_by_rental_rate_per_duration?first_name=PENELOPE&last_name=GUINESS",
    "/v1/movie_3/average_replacement_cost_by_category?category_name=Horror",
    "/v1/movie_3/percentage_films_by_category_and_customer?category_name=Music&first_name=RUTH&last_name=MARTINEZ",
    "/v1/movie_3/average_film_length_by_actor?first_name=PENELOPE&last_name=GUINESS",
    "/v1/movie_3/customer_email_by_name?first_name=DIANE&last_name=COLLINS",
    "/v1/movie_3/customer_count_by_active_status?active=0",
    "/v1/movie_3/customer_name_by_email?email=JEREMY.HURTADO@sakilacustomer.org",
    "/v1/movie_3/postal_code_by_address_id?address_id=65",
    "/v1/movie_3/address_count_by_district?district=Nordrhein-Westfalen",
    "/v1/movie_3/phone_by_address_id?address_id=72",
    "/v1/movie_3/film_count_by_length?length=178",
    "/v1/movie_3/film_special_features_by_title?title=UPRISING%20UPTOWN",
    "/v1/movie_3/store_address_details_by_store_id?store_id=2",
    "/v1/movie_3/country_by_city_name?city=Clarksville",
    "/v1/movie_3/count_actors_in_films?release_year=2006&rental_duration=7&rental_rate=4.99&length=98",
    "/v1/movie_3/film_rating_by_actor?first_name=DAN&last_name=HARRIS&length=77&replacement_cost=9.99",
    "/v1/movie_3/count_films_by_actor?first_name=DARYL&last_name=WAHLBERG",
    "/v1/movie_3/rental_return_date?first_name=SHERRI&last_name=RHODES&rental_date=2005-07-28%2012:27:27",
    "/v1/movie_3/staff_names_by_store?store_id=1",
    "/v1/movie_3/staff_address?first_name=Jon&last_name=Stephens",
    "/v1/movie_3/count_addresses_in_city?city=Woodridge",
    "/v1/movie_3/customer_address?first_name=HEATHER&last_name=MORRIS",
    "/v1/movie_3/staff_email_by_address?address=1411%20Lillydale%20Drive",
    "/v1/movie_3/payment_amount_by_rental_date_customer_id?rental_date=2005-07-28%2012:27:27&customer_id=297",
    "/v1/movie_3/category_name_by_film_title?film_title=WORKING%20MICROCOSMOS",
    "/v1/movie_3/top_category_by_film_count",
    "/v1/movie_3/film_title_by_inventory_id?inventory_id=3479",
    "/v1/movie_3/payment_percentage_difference_by_stores?store_id_1=2&store_id_2=1",
    "/v1/movie_3/city_ratio_by_countries?country_1=India&country_2=Italy",
    "/v1/movie_3/actor_film_percentage?first_name_1=GINA&last_name_1=DEGENERES&first_name_2=PENELOPE&last_name_2=GUINESS",
    "/v1/movie_3/actor_count_by_film?film_id=508",
    "/v1/movie_3/payment_count_by_month_year?month_year=2005-08",
    "/v1/movie_3/film_titles_by_length?length=180",
    "/v1/movie_3/total_payment_by_rental_id_range?start_rental_id=1&end_rental_id=10",
    "/v1/movie_3/store_manager_by_store_id?store_id=2",
    "/v1/movie_3/rental_count_by_date?rental_date=2005-05-27",
    "/v1/movie_3/film_titles_by_store_id?store_id=2",
    "/v1/movie_3/customers_by_rental_days?num_days=7",
    "/v1/movie_3/top_film_special_features_by_category?category_name=sci-fi",
    "/v1/movie_3/actor_most_film_appearances",
    "/v1/movie_3/count_films_category_rental_duration?rental_duration=7&category_name=Comedy",
    "/v1/movie_3/staff_most_inactive_customers",
    "/v1/movie_3/highest_rental_rate_per_day_category?category_name=Children",
    "/v1/movie_3/store_address_details?store_id=1",
    "/v1/movie_3/count_customers_city?city_name=Lethbridge",
    "/v1/movie_3/count_cities_country?country_name=United%20States",
    "/v1/movie_3/customers_country?country_name=India",
    "/v1/movie_3/count_films_category_rental_rate?rental_rate=1&category_name=Classics",
    "/v1/movie_3/customer_most_rentals",
    "/v1/movie_3/rental_count_by_film_title?title=Blanket%20Beverly",
    "/v1/movie_3/top_actor_by_rating?rating=R",
    "/v1/movie_3/top_actors_by_film_count?limit=5",
    "/v1/movie_3/actor_ids_by_last_name?last_name=KILMER",
    "/v1/movie_3/film_titles_min_replacement_cost",
    "/v1/movie_3/longest_films_details?limit=5",
    "/v1/movie_3/rental_count_by_date_range?start_date=2005-05-26&end_date=2005-05-30",
    "/v1/movie_3/average_payment_by_customer",
    "/v1/movie_3/staff_details_by_store_id?store_id=2",
    "/v1/movie_3/percentage_inactive_customers",
    "/v1/movie_3/film_description_title?film_id=996",
    "/v1/movie_3/total_payment_amount?year_month=2005-08",
    "/v1/movie_3/actor_film_titles?first_name=Emily&last_name=Dee",
    "/v1/movie_3/film_actors?film_title=CHOCOLATE%20DUCK",
    "/v1/movie_3/film_count_category_rating?category_name=Horror&rating=PG-13",
    "/v1/movie_3/actor_film_categories?first_name=Judy&last_name=Dean",
    "/v1/movie_3/film_titles_category_limit?category_name=Documentary&limit=5",
    "/v1/movie_3/film_language_cost?film_title=UNTOUCHABLES%20SUNRISE",
    "/v1/movie_3/film_titles_rented_on_date?rental_date=2005-05-24",
    "/v1/movie_3/customer_film_titles_month_year?first_name=BRIAN&last_name=WYMAN&year=2005&month=7",
    "/v1/movie_3/actor_details_by_film_title?film_title=STREETCAR%20INTENTIONS",
    "/v1/movie_3/film_titles_categories_by_customer_year_month?first_name=Natalie&last_name=Meyer&year=2006&month=02",
    "/v1/movie_3/rental_count_by_customer?first_name=Eleanor&last_name=Hunt",
    "/v1/movie_3/customer_details_by_film_title?film_title=DREAM%20PICKUP",
    "/v1/movie_3/customer_percentage_by_country?country=India",
    "/v1/movie_3/actor_appearance_percentage_difference?first_name_1=ANGELA&last_name_1=WITHERSPOON&first_name_2=MARY&last_name_2=KEITEL",
    "/v1/movie_3/customer_details_by_name?first_name=Lillie&last_name=Kim",
    "/v1/movie_3/customer_names_by_staff?first_name=Mike&last_name=Hillyer&limit=5",
    "/v1/movie_3/total_payment_by_customer?first_name=Diane&last_name=Collins",
    "/v1/movie_3/customer_details_above_payment_threshold?threshold_percentage=0.7",
    "/v1/movie_3/film_count_by_rental_rate?rental_rate=0.99",
    "/v1/movie_3/customer_count_by_last_name_and_id?last_name=Thomas&customer_id=100",
    "/v1/movie_3/actor_last_names_by_film_description?description=A%20Thoughtful%20Drama%20of%20a%20Composer%20And%20a%20Feminist%20who%20must%20Meet%20a%20Secret%20Agent%20in%20The%20Canadian%20Rockies",
    "/v1/movie_3/highest_replacement_cost_film_by_actor?first_name=Liza&last_name=Bergman",
    "/v1/movie_3/highest_rental_rate_film_by_store?store_id=2",
    "/v1/movie_3/film_titles_by_actor_and_replacement_cost?first_name=Angelina&last_name=Astaire&replacement_cost=27.99",
    "/v1/movie_3/inventory_ids_by_film_title?title=African%20Egg",
    "/v1/movie_3/actor_count_by_film_length_and_name?length=113&first_name=Kirk&last_name=Jovovich",
    "/v1/movie_3/film_count_by_inventory_id_range_and_rating?min_inventory_id=20&max_inventory_id=60&rating=G",
    "/v1/movie_3/actor_count_by_rental_rate_and_name?rental_rate=4.99&first_name=Bob&last_name=Fawcett",
    "/v1/movie_3/inventory_ids_by_film_length_and_actor?min_length=110&max_length=150&first_name=Russell&last_name=Close",
    "/v1/movie_3/longest_film_store_inventory",
    "/v1/movie_3/film_titles_by_length_and_actor?min_length=110&max_length=150&first_name=Russell&last_name=Close",
    "/v1/movie_3/inventory_ids_by_actor_and_rental_rate?first_name=Lucille&last_name=Dee&rental_rate=4.99",
    "/v1/movie_3/store_ids_above_average_rental_rate",
    "/v1/movie_3/proportion_films_by_rating_and_actor?rating=G&first_name=Elvis&last_name=Marx",
    "/v1/movie_3/address_min_city_id_by_district?district=Texas",
    "/v1/movie_3/customer_details_by_year_and_status?year=2006&active=0",
    "/v1/movie_3/percentage_films_by_rating?rating=PG-13",
    "/v1/movie_3/top_films_by_rental_rate_per_duration",
    "/v1/movie_3/average_payment_amount?customer_id=15",
    "/v1/movie_3/count_long_rentals",
    "/v1/movie_3/percentage_films_in_category?category_name=horror",
    "/v1/movie_3/top_actor_in_category?category_id=7",
    "/v1/movie_3/customer_country_difference?country1=Australia&country2=Canada",
    "/v1/movie_3/actor_category_percentage?category_name=Action&first_name=Reese&last_name=Kilmer",
    "/v1/movie_3/total_payment_for_film?film_title=CLOCKWORK%20PARADICE",
    "/v1/movie_3/customers_with_rentals_above_threshold?rental_threshold=5",
    "/v1/movie_3/average_actor_id_in_category?category_name=comedy",
    "/v1/movie_3/most_rented_film_in_category?category_name=Children",
    "/v1/movie_3/percentage_customers_above_avg_payment",
    "/v1/movie_3/distinct_customer_names_by_category_preference?category1=Family&category2=Sci-Fi",
    "/v1/movie_3/film_titles_by_rating?rating=NC-17",
    "/v1/movie_3/actor_count_by_last_name?last_name=Kilmer",
    "/v1/movie_3/total_payment_by_month_year?month_year=2005-08%",
    "/v1/movie_3/country_by_address?address=1386%20Nakhon%20Sawan%20Boulevard",
    "/v1/movie_3/most_common_language_by_year?release_year=2006",
    "/v1/movie_3/rental_count_by_customer_date_range?first_name=ELLA&last_name=ELLA&start_date=2005-06-01&end_date=2005-06-30",
    "/v1/movie_3/customer_count_by_staff?first_name=Jon&last_name=Stephens",
    "/v1/movie_3/total_payment_by_date?rental_date=2005-07-29",
    "/v1/movie_3/customer_first_names_by_postal_code?postal_code_prefix=76",
    "/v1/movie_3/rental_dates_by_film_title?film_title=BLOOD%20ARGONAUTS",
    "/v1/movie_3/actor_count_by_category?category_name=Music",
    "/v1/movie_3/top_actor_by_category?category_name=Comedy",
    "/v1/movie_3/customer_count_excluding_manager?first_name=Mike",
    "/v1/movie_3/inactive_customer_count?active=0",
    "/v1/movie_3/shortest_film_title",
    "/v1/movie_3/customer_country_by_name?first_name=HECTOR&last_name=POINDEXTER",
    "/v1/movie_3/average_payment_by_category?category_name=Horror",
    "/v1/movie_3/average_payment_by_customer_name?first_name=CHRISTY&last_name=VARGAS",
    "/v1/movie_3/percentage_short_films_by_category?length=100&category_name=Drama",
    "/v1/movie_3/actor_details_by_first_name?first_name=Johnny",
    "/v1/movie_3/address_ids_by_district?district=Gansu",
    "/v1/movie_3/distinct_categories_with_limit?limit=3",
    "/v1/movie_3/customer_names_by_active_status?active=0&limit=3",
    "/v1/movie_3/rental_rate_per_duration?title=AIRPLANE%20SIERRA",
    "/v1/movie_3/city_by_address?address=1623%20Kingstown%20Drive",
    "/v1/movie_3/cities_by_country?country=Algeria",
    "/v1/movie_3/category_count_difference?category1=Children&category2=Action",
    "/v1/movie_3/district_by_customer_name?first_name=Maria&last_name=Miller",
    "/v1/movie_3/customer_names_by_address_and_active_status?address=1795%20Santiago%20de%20Compostela%20Way&active=1",
    "/v1/movie_3/film_count_by_language_length_cost?language=English&min_length=50&max_replacement_cost=10.99",
    "/v1/movie_3/percentage_of_films_in_category?category_name=Documentary",
    "/v1/movie_3/count_films_by_language_and_rating?language_name=English&rating=NC-17",
    "/v1/movie_3/count_actors_by_first_name?first_name=Dan",
    "/v1/movie_3/most_common_customer_first_name?limit=1",
    "/v1/movie_3/film_ratings_by_special_features?special_features=%25Behind%20the%20Scenes%25",
    "/v1/movie_3/customer_with_most_rentals?limit=1",
    "/v1/movie_3/film_titles_by_description?description=%25Lacklusture%25",
    "/v1/movie_3/count_rentals_by_customer?first_name=FRANCIS&last_name=SIKES",
    "/v1/movie_3/top_manager_by_film_count?limit=1",
    "/v1/movie_3/customer_addresses_by_active_status?active=0",
    "/v1/movie_3/top_customer_by_payment_amount?limit=1",
    "/v1/movie_3/top_films_by_rental_count?limit=5",
    "/v1/movie_3/store_addresses",
    "/v1/movie_3/top_city_by_customer_count?limit=1",
    "/v1/movie_3/top_actor_by_film_count?limit=1",
    "/v1/movie_3/actor_details_by_id?actor_id=5",
    "/v1/movie_3/count_films_in_category?category_id=11",
    "/v1/movie_3/top_actor_by_rental_rate?limit=1",
    "/v1/movie_3/film_descriptions_by_actor?first_name=JOHNNY&last_name=DAVIS",
    "/v1/movie_3/customers_by_payment_amount?amount=10",
    "/v1/movie_3/customer_address_by_name_and_postal_code?first_name=SUSAN&postal_code=77948",
    "/v1/movie_3/count_customers_in_city?city=Abu%20Dhabi",
    "/v1/movie_3/customer_names_by_address?address=692%20Joliet%20Street",
    "/v1/movie_3/film_titles_by_category_and_length?category_name=action&min_length=120",
    "/v1/movie_3/top_film_title_by_category?category_name=Horror",
    "/v1/movie_3/film_descriptions_by_category?category_name=Travel",
    "/v1/movie_3/total_payment_by_district?district=Nagasaki",
    "/v1/movie_3/payment_percentage_by_customer?first_name=MARGARET&last_name=MOORE",
    "/v1/movie_3/film_percentage_by_category_and_length?category_name=Horror&min_length=120",
    "/v1/movie_3/film_count_by_release_year?release_year=2006",
    "/v1/movie_3/film_titles_by_id_range?min_id=1&max_id=10",
    "/v1/movie_3/max_rental_duration_films",
    "/v1/movie_3/max_rental_rate_films",
    "/v1/movie_3/film_language_by_title?title=CHILL%20LUCK",
    "/v1/movie_3/distinct_last_update_by_language_and_year?language=English&release_year=2006",
    "/v1/movie_3/count_films_by_language_and_special_features?language=Italian&special_features=deleted%20scenes",
    "/v1/movie_3/count_films_by_category_and_rating?category=animation&rating=NC-17",
    "/v1/movie_3/film_ids_by_category?category=comedy",
    "/v1/movie_3/longest_film_title_by_category?category=documentary",
    "/v1/movie_3/film_category_by_title?title=BLADE%20POLISH",
    "/v1/movie_3/rental_ids_by_customer_name?first_name=MARY&last_name=SMITH",
    "/v1/movie_3/customer_names_by_staff_id?staff_id=1",
    "/v1/movie_3/customer_emails_by_staff_id?staff_id=2",
    "/v1/movie_3/actor_ids_by_film_title?title=BOUND%20CHEAPER",
    "/v1/movie_3/inventory_ids_by_customer_name?first_name=KAREN&last_name=JACKSON",
    "/v1/movie_3/total_rental_rate_by_category?category_name=Animation",
    "/v1/movie_3/average_rental_rate_by_category?category_name=Sci-Fi",
    "/v1/movie_3/percentage_films_category_language?category_name=Horror&language_name=English",
    "/v1/movie_3/count_films_rating_rental_duration?rating=NC-17&rental_duration=4",
    "/v1/movie_3/film_titles_replacement_cost_rating_length?replacement_cost=29.99&rating=R&length=71",
    "/v1/movie_3/customer_emails_rental_date_range?start_date=2005-5-25%2007:37:47&end_date=2005-5-26%2010:06:49&active=1",
    "/v1/movie_3/total_payment_customer?first_name=SARAH&last_name=LEWIS",
    "/v1/movie_3/count_customers_payment_date_range?start_date=2005-05-30%2003:43:54&end_date=2005-07-31%2010:08:29",
    "/v1/movie_3/actors_of_film?film_title=ALABAMA%20DEVIL",
    "/v1/movie_3/film_titles_of_actor?first_name=SANDRA&last_name=KILMER",
    "/v1/movie_3/film_details_rental_duration_rate?product_value=30",
    "/v1/movie_3/most_recent_rented_film_by_customer?first_name=DOROTHY&last_name=TAYLOR",
    "/v1/movie_3/film_count_by_category?category_name=Action",
    "/v1/movie_3/store_details_and_rental_rate_by_film_title?film_title=WYOMING%20STORM",
    "/v1/movie_3/rental_duration_by_customer_and_film?first_name=AUSTIN&film_title=DESTINY%20SATURDAY",
    "/v1/movie_3/film_count_by_actor?first_name=NICK&last_name=STALLONE",
    "/v1/movie_3/shortest_film_with_highest_rental_duration_rate",
    "/v1/movie_3/total_payment_by_customer_and_month?first_name=STEPHANIE&last_name=MITCHELL&month_year=2005-06",
    "/v1/movie_3/average_replacement_cost_by_rental_rate?rental_rate=4.99",
    "/v1/movie_3/average_rental_rate_by_rating?rating=PG-13",
    "/v1/movie_3/rental_duration_by_title?title=DIRTY%20ACE",
    "/v1/movie_3/count_distinct_categories",
    "/v1/movie_3/rental_count_and_percentage_july?first_name=Maria&last_name=Miller&year=2005",
    "/v1/movie_3/film_titles_by_rental_duration_and_customer_count?rental_duration=5&customer_count=10",
    "/v1/movie_3/film_titles_by_actor_name_and_rating?first_name=KARL&last_name=BERRY&rating=PG",
    "/v1/movie_3/most_rented_film_title",
    "/v1/movie_3/film_ids_by_actor_name?first_name=LUCILLE&last_name=TRACY",
    "/v1/movie_3/actor_last_names_by_film_rating?rating=NC-17",
    "/v1/movie_3/average_rental_rate_by_actor_name?first_name=LUCILLE&last_name=TRACY",
    "/v1/movie_3/film_count_by_length_range?min_length=100&max_length=110",
    "/v1/movie_3/active_customer_count_by_first_name?first_name=Nina&active=1",
    "/v1/movie_3/film_count_by_store_and_rating?store_id=2&rating=R",
    "/v1/movie_3/store_ids_by_actor_and_film_length?max_length=100&first_name=Reese&last_name=West",
    "/v1/movie_3/highest_rental_rate_film_by_actor?first_name=Nick&last_name=Wahlberg",
    "/v1/movie_3/store_ids_by_film_title?title=Amadeus%20Holy",
    "/v1/movie_3/film_count_by_rental_rate_and_actor?rental_rate=2.99&first_name=Nina&last_name=Soto",
    "/v1/movie_3/film_count_diff_by_actor?film_id_1=1&film_id_2=2&first_name=Reese&last_name=West",
    "/v1/movie_3/postal_code_by_address?address=692%20Joliet%20Street",
    "/v1/movie_3/active_customer_count_by_city?active=1&city=Arlington",
    "/v1/movie_3/customer_names_by_country?country=Italy",
    "/v1/movie_3/customer_country?first_name=MARY&last_name=SMITH",
    "/v1/movie_3/highest_payment_amount?first_name=MARY&last_name=SMITH",
    "/v1/movie_3/payment_count_by_customer?first_name=MARY&last_name=SMITH",
    "/v1/movie_3/total_payment_amount_by_customer?first_name=MARY&last_name=SMITH",
    "/v1/movie_3/payment_count_by_customer_year_month?first_name=MARY&last_name=SMITH&year=2005&month=06",
    "/v1/movie_3/customer_with_highest_payment",
    "/v1/movie_3/total_payment_by_country?country=Italy",
    "/v1/movie_3/payment_count_by_customer_name_and_amount?first_name=MARY&last_name=SMITH&min_amount=4.99",
    "/v1/movie_3/average_payment_by_country?country=Italy"
]
