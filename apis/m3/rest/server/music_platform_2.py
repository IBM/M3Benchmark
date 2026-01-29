from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/music_platform_2/music_platform_2.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of podcasts in the most popular category
@app.get("/v1/music_platform_2/count_podcasts_most_popular_category", operation_id="get_count_podcasts_most_popular_category", summary="Retrieves the total number of podcasts in the most popular category. The popularity of a category is determined by the count of podcasts it contains. The operation does not require any input parameters and returns a single integer value.")
async def get_count_podcasts_most_popular_category():
    cursor.execute("SELECT COUNT(podcast_id) FROM categories WHERE category = ( SELECT category FROM categories GROUP BY category ORDER BY COUNT(podcast_id) DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of podcasts with a minimum number of categories
@app.get("/v1/music_platform_2/count_podcasts_min_categories", operation_id="get_count_podcasts_min_categories", summary="Retrieves the total count of podcasts that are associated with at least a specified minimum number of categories. The input parameter determines the minimum category count required for a podcast to be included in the result.")
async def get_count_podcasts_min_categories(min_categories: int = Query(..., description="Minimum number of categories a podcast must have")):
    cursor.execute("SELECT COUNT(T1.podcast_id) FROM ( SELECT podcast_id FROM categories GROUP BY podcast_id HAVING COUNT(category) >= ? ) AS T1", (min_categories,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the itunes_id and itunes_url of a podcast by title
@app.get("/v1/music_platform_2/podcast_details_by_title", operation_id="get_podcast_details_by_title", summary="Retrieves the iTunes ID and URL of a podcast based on its title. The provided title is used to search for a matching podcast in the database.")
async def get_podcast_details_by_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT itunes_id, itunes_url FROM podcasts WHERE title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the itunes_url of podcasts with titles containing a specific keyword
@app.get("/v1/music_platform_2/podcast_itunes_url_by_title_keyword", operation_id="get_podcast_itunes_url_by_title_keyword", summary="Retrieve the iTunes URLs of podcasts whose titles include a specified keyword. The search is case-insensitive and returns unique URLs.")
async def get_podcast_itunes_url_by_title_keyword(keyword: str = Query(..., description="Keyword to search in podcast titles")):
    cursor.execute("SELECT itunes_url FROM podcasts WHERE title LIKE ? GROUP BY itunes_url", ('%' + keyword + '%',))
    result = cursor.fetchall()
    if not result:
        return {"itunes_urls": []}
    return {"itunes_urls": result}

# Endpoint to get the categories of a podcast by its title
@app.get("/v1/music_platform_2/podcast_categories_by_title", operation_id="get_podcast_categories_by_title", summary="Retrieves the categories associated with a specific podcast, identified by its title. The operation returns a list of categories that the podcast belongs to, providing insights into the podcast's genre or theme.")
async def get_podcast_categories_by_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT T1.category FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": result}

# Endpoint to get the titles and itunes_urls of podcasts in a specific category
@app.get("/v1/music_platform_2/podcasts_by_category", operation_id="get_podcasts_by_category", summary="Retrieve the titles and iTunes URLs of podcasts that belong to a specified category. The category is used to filter the podcasts and return only those that match the provided category.")
async def get_podcasts_by_category(category: str = Query(..., description="Category of the podcasts")):
    cursor.execute("SELECT T2.title, T2.itunes_url FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"podcasts": []}
    return {"podcasts": result}

# Endpoint to get the count of podcasts with specific criteria
@app.get("/v1/music_platform_2/count_podcasts_by_criteria", operation_id="get_count_podcasts_by_criteria", summary="Retrieve the number of podcasts that match a specified keyword in their title, belong to a certain category, and have a particular rating. This operation allows you to filter podcasts based on these criteria and provides a count of the matching results.")
async def get_count_podcasts_by_criteria(keyword: str = Query(..., description="Keyword to search in podcast titles"), category: str = Query(..., description="Category of the podcasts"), rating: int = Query(..., description="Rating of the podcasts")):
    cursor.execute("SELECT COUNT(T3.podcast_id) FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id INNER JOIN reviews AS T3 ON T3.podcast_id = T2.podcast_id WHERE T2.title LIKE ? AND T1.category = ? AND T3.rating = ?", ('%' + keyword + '%', category, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the author_ids of reviews for a specific podcast title, created in a specific year, and with a rating below a certain value
@app.get("/v1/music_platform_2/review_authors_by_criteria", operation_id="get_review_authors_by_criteria", summary="Retrieve the unique identifiers of authors who have reviewed a specific podcast, published in a given year, and rated below a certain value. The response includes author IDs of those who have left reviews meeting the specified criteria.")
async def get_review_authors_by_criteria(title: str = Query(..., description="Title of the podcast"), year: str = Query(..., description="Year the review was created in 'YYYY-%' format"), max_rating: int = Query(..., description="Maximum rating of the reviews")):
    cursor.execute("SELECT T2.author_id FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ? AND T2.created_at LIKE ? AND T2.rating < ?", (title, year + '%', max_rating))
    result = cursor.fetchall()
    if not result:
        return {"author_ids": []}
    return {"author_ids": result}

# Endpoint to get the titles and categories of podcasts with an average rating above a certain value
@app.get("/v1/music_platform_2/podcasts_above_avg_rating", operation_id="get_podcasts_above_avg_rating", summary="Retrieves the titles and categories of podcasts that have an average rating above a specified minimum value. This operation filters podcasts based on their average review rating and returns the relevant details from the categories and podcasts tables.")
async def get_podcasts_above_avg_rating(min_avg_rating: float = Query(..., description="Minimum average rating of the podcasts")):
    cursor.execute("SELECT T2.title, T1.category FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id INNER JOIN reviews AS T3 ON T3.podcast_id = T2.podcast_id GROUP BY T3.podcast_id HAVING AVG(T3.rating) > ?", (min_avg_rating,))
    result = cursor.fetchall()
    if not result:
        return {"podcasts": []}
    return {"podcasts": result}

# Endpoint to get the distinct titles of podcasts with a specific rating and category
@app.get("/v1/music_platform_2/distinct_podcast_titles_by_rating_category", operation_id="get_distinct_podcast_titles_by_rating_category", summary="Retrieve a unique list of podcast titles that match a specified rating and category. This operation filters podcasts based on the provided rating and category, ensuring that only distinct titles are returned.")
async def get_distinct_podcast_titles_by_rating_category(rating: int = Query(..., description="Rating of the podcasts"), category: str = Query(..., description="Category of the podcasts")):
    cursor.execute("SELECT DISTINCT T2.title FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id INNER JOIN reviews AS T3 ON T3.podcast_id = T2.podcast_id WHERE T3.rating = ? AND T1.category = ?", (rating, category))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get distinct podcast titles, review content, and ratings based on review title
@app.get("/v1/music_platform_2/distinct_podcast_reviews_by_title", operation_id="get_distinct_podcast_reviews_by_title", summary="Retrieve unique podcast titles along with their corresponding review content and ratings, filtered by a specific review title.")
async def get_distinct_podcast_reviews_by_title(review_title: str = Query(..., description="Title of the review")):
    cursor.execute("SELECT DISTINCT T1.title, T2.content, T2.rating FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.title = ?", (review_title,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get review details based on podcast title
@app.get("/v1/music_platform_2/review_details_by_podcast_title", operation_id="get_review_details_by_podcast_title", summary="Retrieves unique review details, including the author's ID, rating, and creation date, for a specific podcast. The podcast is identified by its title.")
async def get_review_details_by_podcast_title(podcast_title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT T2.author_id, T2.rating, T2.created_at FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ? GROUP BY T2.author_id, T2.rating, T2.created_at", (podcast_title,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get the latest review details
@app.get("/v1/music_platform_2/latest_review_details", operation_id="get_latest_review_details", summary="Retrieves the most recent review data for a podcast, including the podcast's unique identifier, the review's creation date, title, and rating. The review data is obtained by joining the podcasts and reviews tables on the podcast ID and sorting the results by the review's creation date in descending order. Only the latest review is returned.")
async def get_latest_review_details():
    cursor.execute("SELECT T1.podcast_id, T2.created_at, T2.title, T2.rating FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id ORDER BY T2.created_at DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"review": []}
    return {"review": result}

# Endpoint to get review details based on author ID
@app.get("/v1/music_platform_2/review_details_by_author_id", operation_id="get_review_details_by_author_id", summary="Retrieves detailed review information, including title, rating, and content, for a specific author. The author is identified by the provided author_id.")
async def get_review_details_by_author_id(author_id: str = Query(..., description="Author ID")):
    cursor.execute("SELECT T2.title, T2.rating, T2.content FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.author_id = ?", (author_id,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get distinct podcast titles and review details based on rating
@app.get("/v1/music_platform_2/distinct_podcast_reviews_by_rating", operation_id="get_distinct_podcast_reviews_by_rating", summary="Retrieve unique podcast titles along with their respective review titles and content, filtered by a specified review rating.")
async def get_distinct_podcast_reviews_by_rating(rating: int = Query(..., description="Rating of the review")):
    cursor.execute("SELECT DISTINCT T1.title, T2.title, T2.content FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get distinct podcast titles and ratings based on creation date
@app.get("/v1/music_platform_2/distinct_podcast_ratings_by_date", operation_id="get_distinct_podcast_ratings_by_date", summary="Retrieves unique podcast titles and their corresponding ratings based on a specified creation date. The creation date should be provided in the 'YYYY-MM-%' format.")
async def get_distinct_podcast_ratings_by_date(created_at: str = Query(..., description="Creation date in 'YYYY-MM-%' format")):
    cursor.execute("SELECT DISTINCT T1.title, T2.rating FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.created_at LIKE ?", (created_at,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get the average rating of the podcast with the most reviews
@app.get("/v1/music_platform_2/average_rating_most_reviews", operation_id="get_average_rating_most_reviews", summary="Retrieves the average rating of the podcast that has received the highest number of reviews. This operation calculates the average rating for each podcast and then identifies the podcast with the most reviews, returning its average rating.")
async def get_average_rating_most_reviews():
    cursor.execute("SELECT AVG(T2.rating) FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id GROUP BY T1.podcast_id ORDER BY COUNT(T2.content) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get podcast details based on review title
@app.get("/v1/music_platform_2/podcast_details_by_review_title", operation_id="get_podcast_details_by_review_title", summary="Retrieves the podcast details, including the podcast ID and iTunes URL, for the podcast associated with a review that has a specific title. The review title is used to identify the podcast.")
async def get_podcast_details_by_review_title(review_title: str = Query(..., description="Title of the review")):
    cursor.execute("SELECT podcast_id, itunes_url FROM podcasts WHERE podcast_id = ( SELECT podcast_id FROM reviews WHERE title = ? )", (review_title,))
    result = cursor.fetchone()
    if not result:
        return {"podcast": []}
    return {"podcast": result}

# Endpoint to get podcast titles based on category
@app.get("/v1/music_platform_2/podcast_titles_by_category", operation_id="get_podcast_titles_by_category", summary="Retrieves a list of podcast titles that belong to the specified category. The category is used to filter the podcasts and return only those that match the provided category.")
async def get_podcast_titles_by_category(category: str = Query(..., description="Category of the podcast")):
    cursor.execute("SELECT T2.title FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"podcasts": []}
    return {"podcasts": result}

# Endpoint to get the content of reviews for a specific podcast title
@app.get("/v1/music_platform_2/reviews_content_by_podcast_title", operation_id="get_reviews_content_by_podcast_title", summary="Retrieves the content of reviews for a podcast identified by its title. The operation filters the reviews based on the provided podcast title and returns the content of the corresponding reviews.")
async def get_reviews_content_by_podcast_title(podcast_title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT content FROM reviews WHERE podcast_id = ( SELECT podcast_id FROM podcasts WHERE title = ? )", (podcast_title,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [row[0] for row in result]}

# Endpoint to get the title and content of reviews for a specific podcast title and rating
@app.get("/v1/music_platform_2/reviews_title_content_by_podcast_title_rating", operation_id="get_reviews_title_content_by_podcast_title_rating", summary="Retrieves the title and content of reviews for a podcast, filtered by a specific podcast title and review rating. This operation allows you to focus on reviews that match a particular rating for a given podcast, providing a targeted view of user feedback.")
async def get_reviews_title_content_by_podcast_title_rating(podcast_title: str = Query(..., description="Title of the podcast"), rating: int = Query(..., description="Rating of the review")):
    cursor.execute("SELECT title, content FROM reviews WHERE podcast_id = ( SELECT podcast_id FROM podcasts WHERE title = ? ) AND rating = ?", (podcast_title, rating))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": [{"title": row[0], "content": row[1]} for row in result]}

# Endpoint to get the count of ratings for a specific podcast title and maximum rating
@app.get("/v1/music_platform_2/count_ratings_by_podcast_title_max_rating", operation_id="get_count_ratings_by_podcast_title_max_rating", summary="Retrieves the total number of reviews for a specified podcast, considering only those with a rating equal to or less than the provided maximum rating.")
async def get_count_ratings_by_podcast_title_max_rating(podcast_title: str = Query(..., description="Title of the podcast"), max_rating: int = Query(..., description="Maximum rating of the review")):
    cursor.execute("SELECT COUNT(T2.rating) FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ? AND T2.rating <= ?", (podcast_title, max_rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average rating of reviews for a specific podcast title
@app.get("/v1/music_platform_2/avg_rating_by_podcast_title", operation_id="get_avg_rating_by_podcast_title", summary="Retrieves the average rating of reviews for a specified podcast. The operation calculates the average rating from the reviews associated with the podcast, providing a summary of its overall rating.")
async def get_avg_rating_by_podcast_title(podcast_title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT AVG(T3.rating) FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id INNER JOIN reviews AS T3 ON T3.podcast_id = T2.podcast_id WHERE T2.title = ?", (podcast_title,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to compare the number of podcasts in two categories
@app.get("/v1/music_platform_2/compare_podcasts_in_categories", operation_id="compare_podcasts_in_categories", summary="This endpoint compares the number of podcasts in two specified categories. It returns the category with the higher count and the difference in the number of podcasts between the two categories. The input parameters are the two categories to be compared.")
async def compare_podcasts_in_categories(category1: str = Query(..., description="First category"), category2: str = Query(..., description="Second category")):
    cursor.execute("SELECT ( SELECT category FROM categories WHERE category = ? OR category = ? GROUP BY category ORDER BY COUNT(podcast_id) DESC LIMIT 1 ) \"has more podcasts\" , ( SELECT SUM(CASE WHEN category = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN category = ? THEN 1 ELSE 0 END) FROM categories ) \"differenct BETWEEN arts-books and arts-design\"", (category1, category2, category1, category2))
    result = cursor.fetchone()
    if not result:
        return {"comparison": []}
    return {"comparison": {"has_more_podcasts": result[0], "difference": result[1]}}

# Endpoint to get the sum of reviews added in a specific month
@app.get("/v1/music_platform_2/sum_reviews_added_by_month", operation_id="get_sum_reviews_added_by_month", summary="Retrieves the total number of reviews added during a specified month. The month should be provided in the 'YYYY-MM-%' format. This operation calculates the sum of reviews added within the given month from the runs data.")
async def get_sum_reviews_added_by_month(month: str = Query(..., description="Month in 'YYYY-MM-%' format")):
    cursor.execute("SELECT SUM(reviews_added) FROM runs WHERE run_at LIKE ?", (month,))
    result = cursor.fetchone()
    if not result:
        return {"sum_reviews_added": []}
    return {"sum_reviews_added": result[0]}

# Endpoint to get the count of reviews with a specific rating within a date range
@app.get("/v1/music_platform_2/count_reviews_by_rating_date_range", operation_id="get_count_reviews_by_rating_date_range", summary="Retrieve the total number of reviews with a specified rating, submitted within a defined date range.")
async def get_count_reviews_by_rating_date_range(rating: int = Query(..., description="Rating of the review"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DDTHH:MM:SS-HH:MM' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DDTHH:MM:SS-HH:MM' format")):
    cursor.execute("SELECT COUNT(podcast_id) FROM reviews WHERE rating = ? AND created_at BETWEEN ? AND ?", (rating, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of podcasts in a specific category
@app.get("/v1/music_platform_2/percentage_podcasts_in_category", operation_id="get_percentage_podcasts_in_category", summary="Retrieves the percentage of podcasts that belong to a specified category. The calculation is based on the total number of podcasts in the database. The category is provided as an input parameter.")
async def get_percentage_podcasts_in_category(category: str = Query(..., description="Category of the podcast")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN category = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(podcast_id) OR '%' \"percentage\" FROM categories", (category,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average rating of reviews within a date range
@app.get("/v1/music_platform_2/avg_rating_by_date_range", operation_id="get_avg_rating_by_date_range", summary="Retrieves the average rating of reviews that were created within a specified date range. The date range is defined by a start date and an end date, both provided in the 'YYYY-MM-DDTHH:MM:SS-HH:MM' format. This operation is useful for analyzing review trends over time.")
async def get_avg_rating_by_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DDTHH:MM:SS-HH:MM' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DDTHH:MM:SS-HH:MM' format")):
    cursor.execute("SELECT AVG(rating) FROM reviews WHERE created_at BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the percentage change in reviews added between two years
@app.get("/v1/music_platform_2/percentage_change_reviews_added", operation_id="get_percentage_change_reviews_added", summary="Retrieves the percentage change in the total number of reviews added between two specified years. The calculation is based on the sum of reviews added in each year, with the result expressed as a percentage of the total reviews added across all years. The input parameters are two years in 'YYYY-%' format, representing the time periods for comparison.")
async def get_percentage_change_reviews_added(year1: str = Query(..., description="First year in 'YYYY-%' format"), year2: str = Query(..., description="Second year in 'YYYY-%' format")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN run_at LIKE ? THEN reviews_added ELSE 0 END) - SUM(CASE WHEN run_at LIKE ? THEN reviews_added ELSE 0 END)) AS REAL) * 100 / SUM(reviews_added) OR '%' \"percentage\" FROM runs", (year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_change": []}
    return {"percentage_change": result[0]}

# Endpoint to get distinct ratings and categories for a specific podcast title
@app.get("/v1/music_platform_2/ratings_categories_by_podcast_title", operation_id="get_ratings_categories_by_podcast_title", summary="Retrieves unique combinations of ratings and categories associated with a specified podcast title. This operation allows you to understand the distinct ratings and categories attributed to a particular podcast, providing insights into its content and audience reception.")
async def get_ratings_categories_by_podcast_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT DISTINCT T3.rating, T1.category FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id INNER JOIN reviews AS T3 ON T3.podcast_id = T2.podcast_id WHERE T2.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"ratings_categories": []}
    return {"ratings_categories": [{"rating": row[0], "category": row[1]} for row in result]}

# Endpoint to get author IDs based on iTunes ID
@app.get("/v1/music_platform_2/author_ids_by_itunes_id", operation_id="get_author_ids_by_itunes_id", summary="Retrieves the unique identifiers of authors who have reviewed a podcast associated with a given iTunes ID. The operation returns a list of author IDs, each corresponding to a reviewer of the podcast identified by the provided iTunes ID.")
async def get_author_ids_by_itunes_id(itunes_id: int = Query(..., description="iTunes ID of the podcast")):
    cursor.execute("SELECT T2.author_id FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.itunes_id = ?", (itunes_id,))
    result = cursor.fetchall()
    if not result:
        return {"author_ids": []}
    return {"author_ids": [row[0] for row in result]}

# Endpoint to get distinct podcast titles based on review creation date range
@app.get("/v1/music_platform_2/podcast_titles_by_review_date_range", operation_id="get_podcast_titles_by_review_date_range", summary="Retrieve a list of unique podcast titles that have been reviewed within a specified date range. The date range is determined by the provided start and end dates, which should be formatted as 'YYYY-MM-DDTHH:MM:SS-07:00'.")
async def get_podcast_titles_by_review_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DDTHH:MM:SS-07:00' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DDTHH:MM:SS-07:00' format")):
    cursor.execute("SELECT DISTINCT T1.title FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.created_at BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct categories based on author ID
@app.get("/v1/music_platform_2/categories_by_author_id", operation_id="get_categories_by_author_id", summary="Retrieves a unique set of categories associated with a given author, based on their podcast reviews.")
async def get_categories_by_author_id(author_id: str = Query(..., description="Author ID")):
    cursor.execute("SELECT DISTINCT T1.category FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.author_id = ?", (author_id,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get podcast slugs and iTunes URLs based on review content
@app.get("/v1/music_platform_2/podcast_info_by_review_content", operation_id="get_podcast_info_by_review_content", summary="Retrieves the slug and iTunes URL of podcasts that have a review containing the specified content. The content parameter is used to filter the reviews and identify the relevant podcasts.")
async def get_podcast_info_by_review_content(content: str = Query(..., description="Content of the review")):
    cursor.execute("SELECT slug, itunes_url FROM podcasts WHERE podcast_id IN ( SELECT podcast_id FROM reviews WHERE content = ? )", (content,))
    result = cursor.fetchall()
    if not result:
        return {"podcast_info": []}
    return {"podcast_info": [{"slug": row[0], "itunes_url": row[1]} for row in result]}

# Endpoint to get review creation dates based on podcast title
@app.get("/v1/music_platform_2/review_dates_by_podcast_title", operation_id="get_review_dates_by_podcast_title", summary="Retrieves the creation dates of all reviews associated with a specified podcast. The podcast is identified by its title.")
async def get_review_dates_by_podcast_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT created_at FROM reviews WHERE podcast_id = ( SELECT podcast_id FROM podcasts WHERE title = ? )", (title,))
    result = cursor.fetchall()
    if not result:
        return {"review_dates": []}
    return {"review_dates": [row[0] for row in result]}

# Endpoint to get the count of distinct categories based on review creation date range
@app.get("/v1/music_platform_2/count_distinct_categories_by_review_date_range", operation_id="get_count_distinct_categories_by_review_date_range", summary="Retrieves the total count of unique categories that have been reviewed within a specified date range. The date range is defined by the provided start and end dates, which should be in the format 'YYYY-MM-DDTHH:MM:SS-07:00'.")
async def get_count_distinct_categories_by_review_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DDTHH:MM:SS-07:00' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DDTHH:MM:SS-07:00' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.category) FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.created_at BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average rating for a specific category
@app.get("/v1/music_platform_2/average_rating_by_category", operation_id="get_average_rating_by_category", summary="Retrieves the average rating of podcasts in a specified category. The category parameter is used to filter the podcasts and calculate the average rating from the associated reviews.")
async def get_average_rating_by_category(category: str = Query(..., description="Category of the podcast")):
    cursor.execute("SELECT AVG(T2.rating) FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get distinct titles of reviews in a specific category and year
@app.get("/v1/music_platform_2/distinct_titles_by_category_and_year", operation_id="get_distinct_titles_by_category_and_year", summary="Retrieves a unique set of review titles for a specified category and year. The category and year parameters are used to filter the results, providing a targeted list of distinct review titles.")
async def get_distinct_titles_by_category_and_year(category: str = Query(..., description="Category of the podcasts"), year: str = Query(..., description="Year in 'YYYY-%' format")):
    cursor.execute("SELECT DISTINCT T2.title FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.category = ? AND T2.created_at LIKE ?", (category, year))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct titles of reviews in a specific category with a rating greater than a specified value
@app.get("/v1/music_platform_2/distinct_titles_by_category_and_rating", operation_id="get_distinct_titles_by_category_and_rating", summary="Retrieves unique titles of reviews for podcasts in a specified category, with a rating higher than a given threshold. This operation allows you to filter and obtain distinct review titles based on the category and minimum rating criteria.")
async def get_distinct_titles_by_category_and_rating(category: str = Query(..., description="Category of the podcasts"), min_rating: int = Query(..., description="Minimum rating")):
    cursor.execute("SELECT DISTINCT T2.title FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.category = ? AND T2.rating > ?", (category, min_rating))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct titles of reviews based on multiple criteria
@app.get("/v1/music_platform_2/distinct_titles_by_multiple_criteria", operation_id="get_distinct_titles_by_multiple_criteria", summary="Get distinct titles of reviews based on multiple criteria")
async def get_distinct_titles_by_multiple_criteria(year1: str = Query(..., description="Year in 'YYYY-%' format for the first criteria"), category1: str = Query(..., description="Category for the first criteria"), content1: str = Query(..., description="Content keyword for the first criteria"), year2: str = Query(..., description="Year in 'YYYY-%' format for the second criteria"), category2: str = Query(..., description="Category for the second criteria"), content2: str = Query(..., description="Content keyword for the second criteria")):
    cursor.execute("SELECT DISTINCT T2.title FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE (T2.created_at LIKE ? AND T1.category = ? AND T2.content LIKE ?) OR (T2.created_at LIKE ? AND T1.category = ? AND T2.content LIKE ?)", (year1, category1, content1, year2, category2, content2))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the category and iTunes URL of a podcast by title
@app.get("/v1/music_platform_2/category_and_itunes_url_by_title", operation_id="get_category_and_itunes_url_by_title", summary="Retrieves the category and iTunes URL associated with a specific podcast, based on the provided podcast title.")
async def get_category_and_itunes_url_by_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT T1.category, T2.itunes_url FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"category_and_itunes_url": []}
    return {"category_and_itunes_url": [{"category": row[0], "itunes_url": row[1]} for row in result]}

# Endpoint to get the category with the least number of podcasts
@app.get("/v1/music_platform_2/least_popular_category", operation_id="get_least_popular_category", summary="Retrieves the category with the fewest associated podcasts. This operation identifies the category with the least number of podcasts by grouping categories and ordering them based on the count of podcasts. The category with the lowest count is returned.")
async def get_least_popular_category():
    cursor.execute("SELECT category FROM categories GROUP BY category ORDER BY COUNT(podcast_id) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category": []}
    return {"category": result[0]}

# Endpoint to get the review with the longest content
@app.get("/v1/music_platform_2/longest_review", operation_id="get_longest_review", summary="Retrieves the title of the review with the longest content from the reviews database. The operation ranks reviews by the length of their content and returns the title of the review with the highest rank.")
async def get_longest_review():
    cursor.execute("SELECT title FROM reviews ORDER BY LENGTH(content) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the podcast title based on a review title
@app.get("/v1/music_platform_2/podcast_title_by_review_title", operation_id="get_podcast_title_by_review_title", summary="Retrieves the title of a podcast that corresponds to a given review title. The operation searches for the review title in the reviews table and returns the title of the associated podcast from the podcasts table.")
async def get_podcast_title_by_review_title(review_title: str = Query(..., description="Title of the review")):
    cursor.execute("SELECT title FROM podcasts WHERE podcast_id = ( SELECT podcast_id FROM reviews WHERE title = ? )", (review_title,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the longest podcast title in a specific category
@app.get("/v1/music_platform_2/longest_podcast_title_by_category", operation_id="get_longest_podcast_title_by_category", summary="Retrieves the longest title of a podcast belonging to a specified category. The operation filters podcasts by the provided category and returns the one with the longest title.")
async def get_longest_podcast_title_by_category(category: str = Query(..., description="Category of the podcasts")):
    cursor.execute("SELECT T2.title FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.category = ? ORDER BY LENGTH(T2.title) DESC LIMIT 1", (category,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get categories of podcasts with titles containing a specific substring
@app.get("/v1/music_platform_2/categories_by_podcast_title_substring", operation_id="get_categories_by_podcast_title_substring", summary="Retrieves the categories of podcasts whose titles contain a specified substring. The operation filters podcasts based on the provided substring and returns the corresponding categories.")
async def get_categories_by_podcast_title_substring(title_substring: str = Query(..., description="Substring to search in podcast titles")):
    cursor.execute("SELECT category FROM categories WHERE podcast_id IN ( SELECT podcast_id FROM podcasts WHERE title LIKE ? )", ('%' + title_substring + '%',))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get categories of podcasts with a specific title
@app.get("/v1/music_platform_2/categories_by_podcast_title", operation_id="get_categories_by_podcast_title", summary="Retrieves the categories associated with podcasts that have a specified title. The operation filters podcasts by title and returns the corresponding categories.")
async def get_categories_by_podcast_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT category FROM categories WHERE podcast_id IN ( SELECT podcast_id FROM podcasts WHERE title = ? )", (title,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get categories of podcasts with a specific rating
@app.get("/v1/music_platform_2/categories_by_rating", operation_id="get_categories_by_rating", summary="Retrieves the categories of podcasts that have been rated with the specified value. The rating parameter is used to filter the categories based on the user-provided rating.")
async def get_categories_by_rating(rating: int = Query(..., description="Rating of the podcast")):
    cursor.execute("SELECT T1.category FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get podcast titles with a specific review title
@app.get("/v1/music_platform_2/podcast_titles_by_review_title", operation_id="get_podcast_titles_by_review_title", summary="Retrieves the titles of podcasts that have been reviewed with a specific title. The operation filters podcasts based on the provided review title and returns their corresponding titles.")
async def get_podcast_titles_by_review_title(review_title: str = Query(..., description="Title of the review")):
    cursor.execute("SELECT T1.title FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.title = ?", (review_title,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get podcast titles and categories with reviews containing a specific substring
@app.get("/v1/music_platform_2/podcast_titles_categories_by_review_content", operation_id="get_podcast_titles_categories_by_review_content", summary="Retrieves podcast titles and their respective categories where the review content contains a specified substring. The substring is provided as an input parameter.")
async def get_podcast_titles_categories_by_review_content(content_substring: str = Query(..., description="Substring to search in review content")):
    cursor.execute("SELECT T2.title, T1.category FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id INNER JOIN reviews AS T3 ON T3.podcast_id = T2.podcast_id WHERE T3.content LIKE ?", ('%' + content_substring + '%',))
    result = cursor.fetchall()
    if not result:
        return {"podcasts": []}
    return {"podcasts": [{"title": row[0], "category": row[1]} for row in result]}

# Endpoint to get the most reviewed category
@app.get("/v1/music_platform_2/most_reviewed_category", operation_id="get_most_reviewed_category", summary="Retrieves the category with the highest number of reviews. This operation identifies the category that has been reviewed the most by users, providing insights into the most popular category in the music platform. The category is determined by counting the number of reviews associated with each podcast within a category and then selecting the category with the highest count.")
async def get_most_reviewed_category():
    cursor.execute("SELECT T1.category FROM categories AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id GROUP BY T1.category ORDER BY COUNT(T2.podcast_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category": []}
    return {"category": result[0]}

# Endpoint to get iTunes URLs of podcasts in a specific category
@app.get("/v1/music_platform_2/itunes_urls_by_category", operation_id="get_itunes_urls_by_category", summary="Retrieve the iTunes URLs of podcasts that belong to a specified category. The category parameter is used to filter the podcasts and return only those that match the provided category.")
async def get_itunes_urls_by_category(category: str = Query(..., description="Category of the podcast")):
    cursor.execute("SELECT itunes_url FROM podcasts WHERE podcast_id IN ( SELECT podcast_id FROM categories WHERE category = ? )", (category,))
    result = cursor.fetchall()
    if not result:
        return {"itunes_urls": []}
    return {"itunes_urls": [row[0] for row in result]}

# Endpoint to get the earliest review content for a specific podcast title
@app.get("/v1/music_platform_2/earliest_review_by_podcast_title", operation_id="get_earliest_review_by_podcast_title", summary="Retrieves the earliest review content for a specified podcast title. The operation identifies the podcast by its title and fetches the corresponding review with the earliest creation date.")
async def get_earliest_review_by_podcast_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT T2.content FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ? ORDER BY T2.created_at ASC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"content": []}
    return {"content": result[0]}

# Endpoint to get the count of reviews for a specific podcast title
@app.get("/v1/music_platform_2/review_count_by_podcast_title", operation_id="get_review_count_by_podcast_title", summary="Retrieves the total number of reviews for a specified podcast. The podcast is identified by its title.")
async def get_review_count_by_podcast_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT COUNT(T2.podcast_id) FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average rating for a specific podcast title
@app.get("/v1/music_platform_2/average_rating_by_podcast_title", operation_id="get_average_rating_by_podcast_title", summary="Retrieves the average rating of a podcast, based on its title. This operation calculates the average rating from all reviews associated with the specified podcast title.")
async def get_average_rating_by_podcast_title(title: str = Query(..., description="Title of the podcast")):
    cursor.execute("SELECT AVG(T2.rating) FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"average_rating": []}
    return {"average_rating": result[0]}

# Endpoint to get the content of a review based on title and creation date
@app.get("/v1/music_platform_2/review_content_by_title_and_date", operation_id="get_review_content", summary="Retrieves the content of a specific review based on its title and creation date. The review's title and exact creation date are required as input parameters to accurately locate the desired review.")
async def get_review_content(title: str = Query(..., description="Title of the review"), created_at: str = Query(..., description="Creation date of the review in 'YYYY-MM-DDTHH:MM:SS-TZ' format")):
    cursor.execute("SELECT content FROM reviews WHERE title = ? AND created_at = ?", (title, created_at))
    result = cursor.fetchone()
    if not result:
        return {"content": []}
    return {"content": result[0]}

# Endpoint to get distinct podcast titles based on category
@app.get("/v1/music_platform_2/distinct_podcast_titles_by_category", operation_id="get_distinct_podcast_titles_by_category", summary="Retrieves a list of unique podcast titles that belong to the specified category. This operation allows you to explore the diverse range of podcasts available within a particular category, providing a comprehensive overview of the podcast titles without duplication.")
async def get_distinct_podcast_titles_by_category(category: str = Query(..., description="Category of the podcast")):
    cursor.execute("SELECT DISTINCT T2.title FROM categories AS T1 INNER JOIN podcasts AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of reviews created in a specific year for a specific podcast title
@app.get("/v1/music_platform_2/review_count_by_year_and_title", operation_id="get_review_count_by_year_and_title", summary="Retrieves the total number of reviews for a specified podcast title in a given year. The operation filters reviews based on the provided podcast title and the year they were created, then returns the count of matching reviews.")
async def get_review_count_by_year_and_title(title: str = Query(..., description="Title of the podcast"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.created_at) FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ? AND T2.created_at LIKE ?", (title, year + '-%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get podcast titles reviewed by a specific author
@app.get("/v1/music_platform_2/podcast_titles_by_author", operation_id="get_podcast_titles_by_author", summary="Retrieves the titles of podcasts that a specific author has reviewed. The author is identified by their unique ID.")
async def get_podcast_titles_by_author(author_id: str = Query(..., description="Author ID")):
    cursor.execute("SELECT T2.title FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T2.author_id = ?", (author_id,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of reviews with a specific rating for a specific podcast title
@app.get("/v1/music_platform_2/review_count_by_rating_and_title", operation_id="get_review_count_by_rating_and_title", summary="Retrieves the total number of reviews with a specified rating for a given podcast title. This operation allows you to analyze the popularity and reception of a podcast based on the review count and rating.")
async def get_review_count_by_rating_and_title(title: str = Query(..., description="Title of the podcast"), rating: int = Query(..., description="Rating of the review")):
    cursor.execute("SELECT COUNT(T2.rating) FROM podcasts AS T1 INNER JOIN reviews AS T2 ON T2.podcast_id = T1.podcast_id WHERE T1.title = ? AND T2.rating = ?", (title, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

api_calls = [
    "/v1/music_platform_2/count_podcasts_most_popular_category",
    "/v1/music_platform_2/count_podcasts_min_categories?min_categories=4",
    "/v1/music_platform_2/podcast_details_by_title?title=Brown%20Suga%20Diaries",
    "/v1/music_platform_2/podcast_itunes_url_by_title_keyword?keyword=Dream",
    "/v1/music_platform_2/podcast_categories_by_title?title=I%20Heart%20My%20Life%20Show",
    "/v1/music_platform_2/podcasts_by_category?category=society-culture",
    "/v1/music_platform_2/count_podcasts_by_criteria?keyword=spoilers&category=arts&rating=5",
    "/v1/music_platform_2/review_authors_by_criteria?title=Pop%20Rocket&year=2016&max_rating=5",
    "/v1/music_platform_2/podcasts_above_avg_rating?min_avg_rating=3",
    "/v1/music_platform_2/distinct_podcast_titles_by_rating_category?rating=5&category=fiction",
    "/v1/music_platform_2/distinct_podcast_reviews_by_title?review_title=Love%20it!",
    "/v1/music_platform_2/review_details_by_podcast_title?podcast_title=In%20The%20Thick",
    "/v1/music_platform_2/latest_review_details",
    "/v1/music_platform_2/review_details_by_author_id?author_id=76A4C24B6038145",
    "/v1/music_platform_2/distinct_podcast_reviews_by_rating?rating=1",
    "/v1/music_platform_2/distinct_podcast_ratings_by_date?created_at=2019-05-%",
    "/v1/music_platform_2/average_rating_most_reviews",
    "/v1/music_platform_2/podcast_details_by_review_title?review_title=Long%20time%20listener%2C%20calling%20it%20quits",
    "/v1/music_platform_2/podcast_titles_by_category?category=true-crime",
    "/v1/music_platform_2/reviews_content_by_podcast_title?podcast_title=StormCast%3A%20The%20Official%20Warhammer%20Age%20of%20Sigmar%20Podcast",
    "/v1/music_platform_2/reviews_title_content_by_podcast_title_rating?podcast_title=More%20Stupider%3A%20A%2090-Day%20Fiance%20Podcast&rating=1",
    "/v1/music_platform_2/count_ratings_by_podcast_title_max_rating?podcast_title=LifeAfter%2FThe%20Message&max_rating=3",
    "/v1/music_platform_2/avg_rating_by_podcast_title?podcast_title=More%20Stupider%3A%20A%2090-Day%20Fiance%20Podcast",
    "/v1/music_platform_2/compare_podcasts_in_categories?category1=arts-books&category2=arts-design",
    "/v1/music_platform_2/sum_reviews_added_by_month?month=2022-06-%",
    "/v1/music_platform_2/count_reviews_by_rating_date_range?rating=3&start_date=2015-01-01T00%3A00%3A00-07%3A00&end_date=2015-03-31T23%3A59%3A59-07%3A00",
    "/v1/music_platform_2/percentage_podcasts_in_category?category=fiction-science-fiction",
    "/v1/music_platform_2/avg_rating_by_date_range?start_date=2019-01-01T00%3A00%3A00-07%3A00&end_date=2019-12-31T23%3A59%3A59-07%3A00",
    "/v1/music_platform_2/percentage_change_reviews_added?year1=2022-%&year2=2021-%",
    "/v1/music_platform_2/ratings_categories_by_podcast_title?title=Sitcomadon",
    "/v1/music_platform_2/author_ids_by_itunes_id?itunes_id=1516665400",
    "/v1/music_platform_2/podcast_titles_by_review_date_range?start_date=2018-08-22T11:53:16-07:00&end_date=2018-11-20T11:14:20-07:00",
    "/v1/music_platform_2/categories_by_author_id?author_id=EFB34EAC8E9397C",
    "/v1/music_platform_2/podcast_info_by_review_content?content=Can't%20stop%20listening",
    "/v1/music_platform_2/review_dates_by_podcast_title?title=Don't%20Lie%20To%20Your%20Life%20Coach",
    "/v1/music_platform_2/count_distinct_categories_by_review_date_range?start_date=2016-07-01T00:00:00-07:00&end_date=2016-12-31T23:59:59-07:00",
    "/v1/music_platform_2/average_rating_by_category?category=true-crime",
    "/v1/music_platform_2/distinct_titles_by_category_and_year?category=arts&year=2018-%",
    "/v1/music_platform_2/distinct_titles_by_category_and_rating?category=music&min_rating=3",
    "/v1/music_platform_2/distinct_titles_by_multiple_criteria?year1=2018-%&category1=arts&content1=%25love%25&year2=2019-%&category2=arts&content2=%25love%25",
    "/v1/music_platform_2/category_and_itunes_url_by_title?title=Scaling%20Global",
    "/v1/music_platform_2/least_popular_category",
    "/v1/music_platform_2/longest_review",
    "/v1/music_platform_2/podcast_title_by_review_title?review_title=Hosts%20bring%20the%20show%20down",
    "/v1/music_platform_2/longest_podcast_title_by_category?category=music",
    "/v1/music_platform_2/categories_by_podcast_title_substring?title_substring=jessica",
    "/v1/music_platform_2/categories_by_podcast_title?title=Moist%20Boys",
    "/v1/music_platform_2/categories_by_rating?rating=2",
    "/v1/music_platform_2/podcast_titles_by_review_title?review_title=Inspired%20%26%20On%20Fire!",
    "/v1/music_platform_2/podcast_titles_categories_by_review_content?content_substring=Absolutely%20fantastic",
    "/v1/music_platform_2/most_reviewed_category",
    "/v1/music_platform_2/itunes_urls_by_category?category=fiction-science-fiction",
    "/v1/music_platform_2/earliest_review_by_podcast_title?title=Stuff%20You%20Should%20Know",
    "/v1/music_platform_2/review_count_by_podcast_title?title=Planet%20Money",
    "/v1/music_platform_2/average_rating_by_podcast_title?title=Crime%20Junkie",
    "/v1/music_platform_2/review_content_by_title_and_date?title=really%20interesting!&created_at=2018-04-24T12:05:16-07:00",
    "/v1/music_platform_2/distinct_podcast_titles_by_category?category=arts-performing-arts",
    "/v1/music_platform_2/review_count_by_year_and_title?title=Please%20Excuse%20My%20Dead%20Aunt%20Sally&year=2019",
    "/v1/music_platform_2/podcast_titles_by_author?author_id=F7E5A318989779D",
    "/v1/music_platform_2/review_count_by_rating_and_title?title=Please%20Excuse%20My%20Dead%20Aunt%20Sally&rating=5"
]
