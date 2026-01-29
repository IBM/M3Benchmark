from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/books/books.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of books based on publisher ID and minimum number of pages
@app.get("/v1/books/count_books_by_publisher_and_pages", operation_id="get_count_books_by_publisher_and_pages", summary="Retrieves the total count of books from a specific publisher that have more than a specified number of pages. The operation requires the publisher's ID and the minimum number of pages as input parameters.")
async def get_count_books_by_publisher_and_pages(publisher_id: int = Query(..., description="ID of the publisher"), min_pages: int = Query(..., description="Minimum number of pages")):
    cursor.execute("SELECT COUNT(*) FROM book WHERE publisher_id = ? AND num_pages > ?", (publisher_id, min_pages))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the publication date of the book with the most pages
@app.get("/v1/books/publication_date_most_pages", operation_id="get_publication_date_most_pages", summary="Retrieves the publication date of the book with the most pages. This operation identifies the book with the highest page count and returns its publication date.")
async def get_publication_date_most_pages():
    cursor.execute("SELECT publication_date FROM book ORDER BY num_pages DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"publication_date": []}
    return {"publication_date": result[0]}

# Endpoint to get the publisher name of a book by its title
@app.get("/v1/books/publisher_name_by_title", operation_id="get_publisher_name_by_title", summary="Retrieves the name of the publisher associated with a specific book title. The operation uses the provided book title to search for a match in the book database and returns the corresponding publisher name.")
async def get_publisher_name_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of books by publisher name
@app.get("/v1/books/count_books_by_publisher_name", operation_id="get_count_books_by_publisher_name", summary="Retrieves the total number of books published by a specific publisher. The operation requires the publisher's name as input and returns the corresponding count of books.")
async def get_count_books_by_publisher_name(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the publisher with the most books
@app.get("/v1/books/publisher_with_most_books", operation_id="get_publisher_with_most_books", summary="Retrieves the name of the publisher who has published the most books. This operation identifies the publisher with the highest number of books in the database by counting the number of books associated with each publisher and then selecting the publisher with the highest count.")
async def get_publisher_with_most_books():
    cursor.execute("SELECT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id GROUP BY T2.publisher_name ORDER BY COUNT(T1.book_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the earliest published book by a specific publisher
@app.get("/v1/books/earliest_book_by_publisher", operation_id="get_earliest_book_by_publisher", summary="Retrieves the title of the earliest published book from the specified publisher. The operation filters books by publisher name and sorts them by publication date in ascending order to identify the earliest book.")
async def get_earliest_book_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ? ORDER BY T1.publication_date ASC LIMIT 1", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of books by publisher name and minimum number of pages
@app.get("/v1/books/count_books_by_publisher_and_min_pages", operation_id="get_count_books_by_publisher_and_min_pages", summary="Retrieves the total number of books from a specified publisher that have more than a given minimum number of pages. This operation requires the publisher's name and the minimum number of pages as input parameters.")
async def get_count_books_by_publisher_and_min_pages(publisher_name: str = Query(..., description="Name of the publisher"), min_pages: int = Query(..., description="Minimum number of pages")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ? AND T1.num_pages > ?", (publisher_name, min_pages))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the publisher of the book with the most pages
@app.get("/v1/books/publisher_of_book_with_most_pages", operation_id="get_publisher_of_book_with_most_pages", summary="Retrieves the name of the publisher associated with the book that has the most number of pages. This operation identifies the book with the highest page count and returns the name of its publisher.")
async def get_publisher_of_book_with_most_pages():
    cursor.execute("SELECT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id ORDER BY T1.num_pages DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of books by language name
@app.get("/v1/books/count_books_by_language", operation_id="get_count_books_by_language", summary="Retrieves the total number of books written in a specified language. The operation filters the books based on the provided language name and returns the count.")
async def get_count_books_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id WHERE T2.language_name = ?", (language_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the titles of books by language name
@app.get("/v1/books/titles_by_language", operation_id="get_titles_by_language", summary="Retrieves the titles of books written in a specified language. The language is identified by its name, which is provided as an input parameter.")
async def get_titles_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id WHERE T2.language_name = ?", (language_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the minimum price of a book by title
@app.get("/v1/books/min_price_by_title", operation_id="get_min_price_by_title", summary="Retrieves the lowest price at which a book has been sold, based on the provided book title. This operation considers all sales records associated with the book and returns the minimum price found.")
async def get_min_price_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT MIN(T2.price) FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"min_price": []}
    return {"min_price": result[0]}

# Endpoint to get book titles ordered by a specific customer
@app.get("/v1/books/titles_by_customer", operation_id="get_titles_by_customer", summary="Retrieves the titles of books ordered by a customer specified by their first and last names. The operation returns a list of book titles associated with the customer's orders.")
async def get_titles_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T4.first_name = ? AND T4.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of books ordered by a specific customer with more than a specified number of pages
@app.get("/v1/books/count_by_customer_and_pages", operation_id="get_count_by_customer_and_pages", summary="Retrieve the count of books, ordered by a specific customer, that have more than a specified number of pages. The operation requires the customer's first and last name, as well as the minimum number of pages for the books to be considered.")
async def get_count_by_customer_and_pages(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), num_pages: int = Query(..., description="Minimum number of pages")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T4.first_name = ? AND T4.last_name = ? AND T1.num_pages > ?", (first_name, last_name, num_pages))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total price of orders by a specific customer
@app.get("/v1/books/total_price_by_customer", operation_id="get_total_price_by_customer", summary="Retrieves the total price of all orders placed by a customer identified by their first and last names. The operation calculates the sum of the prices of all order lines associated with the customer's orders.")
async def get_total_price_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT SUM(T1.price) FROM order_line AS T1 INNER JOIN cust_order AS T2 ON T2.order_id = T1.order_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get the average price of orders by a specific customer
@app.get("/v1/books/average_price_by_customer", operation_id="get_average_price_by_customer", summary="Retrieves the average price of all orders placed by a specific customer, identified by their first and last names. This operation calculates the total sum of order prices and divides it by the total number of orders for the given customer.")
async def get_average_price_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT SUM(T1.price) / COUNT(*) FROM order_line AS T1 INNER JOIN cust_order AS T2 ON T2.order_id = T1.order_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the percentage of orders above a certain price by a specific customer
@app.get("/v1/books/percentage_above_price_by_customer", operation_id="get_percentage_above_price_by_customer", summary="Retrieves the percentage of orders placed by a specific customer that exceed a given price threshold. This operation calculates the proportion of orders above the price threshold by querying the order_line, cust_order, and customer tables. The customer is identified by their first and last names, and the price threshold is provided as an input parameter.")
async def get_percentage_above_price_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), price_threshold: float = Query(..., description="Price threshold")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.price > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM order_line AS T1 INNER JOIN cust_order AS T2 ON T2.order_id = T1.order_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T3.first_name = ? AND T3.last_name = ?", (price_threshold, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the city of a specific address
@app.get("/v1/books/city_by_address_id", operation_id="get_city_by_address_id", summary="Retrieves the city associated with a specific address, identified by its unique address ID.")
async def get_city_by_address_id(address_id: int = Query(..., description="ID of the address")):
    cursor.execute("SELECT city FROM address WHERE address_id = ?", (address_id,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of orders by a specific customer
@app.get("/v1/books/order_count_by_customer", operation_id="get_order_count_by_customer", summary="Retrieves the total number of orders placed by a customer identified by their first and last names. The endpoint uses the provided first and last names to filter the customer records and calculate the order count.")
async def get_order_count_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(*) FROM customer AS T1 INNER JOIN cust_order AS T2 ON T1.customer_id = T2.customer_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get the earliest published book in a specific language
@app.get("/v1/books/earliest_book_by_language", operation_id="get_earliest_book_by_language", summary="Retrieves the title of the earliest published book in a specified language. The operation filters books by language and sorts them by publication date in ascending order, returning the title of the first book.")
async def get_earliest_book_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id WHERE T2.language_name = ? ORDER BY T1.publication_date ASC LIMIT 1", (language_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the publisher with the most books
@app.get("/v1/books/top_publisher", operation_id="get_top_publisher", summary="Retrieves the name of the publisher with the highest number of books in the catalog. The operation ranks publishers by the count of their associated books and returns the top-ranked publisher.")
async def get_top_publisher():
    cursor.execute("SELECT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id GROUP BY T2.publisher_name ORDER BY COUNT(T2.publisher_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of books from a specific publisher
@app.get("/v1/books/count_books_by_publisher", operation_id="get_count_books_by_publisher", summary="Retrieves the total number of books published by a specific publisher. The operation requires the publisher's name as input and returns the corresponding count of books.")
async def get_count_books_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(T1.book_id) FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ?", (publisher_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the language of a specific book
@app.get("/v1/books/get_language_by_book_id", operation_id="get_language_by_book_id", summary="Retrieves the language of a book identified by its unique ID. The operation fetches the language name from the book_language table, which is associated with the book's language ID in the book table.")
async def get_language_by_book_id(book_id: int = Query(..., description="ID of the book")):
    cursor.execute("SELECT T2.language_name FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id WHERE T1.book_id = ?", (book_id,))
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get the most frequent customer by order count
@app.get("/v1/books/most_frequent_customer", operation_id="get_most_frequent_customer", summary="Retrieves the customer who has placed the most orders. The response includes the first and last name of the customer with the highest order count.")
async def get_most_frequent_customer():
    cursor.execute("SELECT T1.first_name, T1.last_name FROM customer AS T1 INNER JOIN cust_order AS T2 ON T1.customer_id = T2.customer_id GROUP BY T1.first_name, T1.last_name ORDER BY COUNT(*) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the most ordered book title
@app.get("/v1/books/most_ordered_book", operation_id="get_most_ordered_book", summary="Retrieves the title of the book that has been ordered the most. This operation returns the most popular book based on order history, providing a snapshot of the most frequently purchased book.")
async def get_most_ordered_book():
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id GROUP BY T1.title ORDER BY COUNT(T1.title) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of books by a specific author
@app.get("/v1/books/count_books_by_author", operation_id="get_count_books_by_author", summary="Retrieves the total number of books written by a specific author. The author's name is used to filter the results.")
async def get_count_books_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT COUNT(T1.title) FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ?", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of orders for a specific book title
@app.get("/v1/books/count_orders_by_title", operation_id="get_count_orders_by_title", summary="Retrieves the total number of orders for a specific book title. The operation requires the title of the book as an input parameter to accurately calculate the order count.")
async def get_count_orders_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country of a customer by first name, last name, and address status
@app.get("/v1/books/get_country_by_customer_name_status", operation_id="get_country_by_customer_name_status", summary="Retrieves the country associated with a customer, based on the provided first name, last name, and address status. This operation returns the name of the country where the customer resides, as determined by the specified address status.")
async def get_country_by_customer_name_status(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), status_id: int = Query(..., description="Status ID of the address")):
    cursor.execute("SELECT T4.country_name FROM customer AS T1 INNER JOIN customer_address AS T2 ON T1.customer_id = T2.customer_id INNER JOIN address AS T3 ON T3.address_id = T2.address_id INNER JOIN country AS T4 ON T4.country_id = T3.country_id WHERE T1.first_name = ? AND T1.last_name = ? AND T2.status_id = ?", (first_name, last_name, status_id))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of addresses in a specific country
@app.get("/v1/books/count_addresses_by_country", operation_id="get_count_addresses_by_country", summary="Retrieves the total number of addresses located in a specified country. The operation requires the name of the country as an input parameter.")
async def get_count_addresses_by_country(country_name: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT COUNT(*) FROM country AS T1 INNER JOIN address AS T2 ON T1.country_id = T2.country_id WHERE T1.country_name = ?", (country_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country name by city
@app.get("/v1/books/get_country_by_city", operation_id="get_country_by_city", summary="Retrieves the name of the country associated with the specified city. The operation uses the provided city name to search for a matching entry in the address table, then retrieves the corresponding country name from the country table.")
async def get_country_by_city(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.country_name FROM country AS T1 INNER JOIN address AS T2 ON T1.country_id = T2.country_id WHERE T2.city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the percentage of orders using a specific shipping method on a given date
@app.get("/v1/books/percentage_orders_by_shipping_method", operation_id="get_percentage_orders_by_shipping_method", summary="Retrieves the percentage of orders that used a specified shipping method on a given date. The calculation is based on the total number of orders placed on that date. The shipping method is identified by its name, and the date is provided in the 'YYYY-MM-DD%' format.")
async def get_percentage_orders_by_shipping_method(method_name: str = Query(..., description="Name of the shipping method"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.method_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM shipping_method AS T1 INNER JOIN cust_order AS T2 ON T1.method_id = T2.shipping_method_id WHERE T2.order_date LIKE ?", (method_name, order_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average number of pages of books by a specific author
@app.get("/v1/books/average_pages_by_author", operation_id="get_average_pages_by_author", summary="Retrieves the average number of pages for books written by a specific author. The author's name is used to filter the results.")
async def get_average_pages_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT AVG(T1.num_pages) FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ?", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_pages": []}
    return {"average_pages": result[0]}

# Endpoint to get the cheapest shipping method
@app.get("/v1/books/cheapest_shipping_method", operation_id="get_cheapest_shipping_method", summary="Retrieves the name of the most cost-effective shipping method available. The method is determined by sorting all available shipping methods in ascending order based on their respective costs and selecting the first one.")
async def get_cheapest_shipping_method():
    cursor.execute("SELECT method_name FROM shipping_method ORDER BY cost ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"method_name": []}
    return {"method_name": result[0]}

# Endpoint to get the first book published in a specific year
@app.get("/v1/books/first_book_by_year", operation_id="get_first_book_by_year", summary="Retrieves the title of the first book published in a given year. The year is specified in 'YYYY' format. The book is determined based on the earliest publication date within the specified year.")
async def get_first_book_by_year(year: str = Query(..., description="Year of publication in 'YYYY' format")):
    cursor.execute("SELECT title FROM book WHERE STRFTIME('%Y', publication_date) = ? ORDER BY publication_date LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get customer details by email
@app.get("/v1/books/customer_details_by_email", operation_id="get_customer_details_by_email", summary="Retrieves the first and last name of a customer based on the provided email address.")
async def get_customer_details_by_email(email: str = Query(..., description="Email of the customer")):
    cursor.execute("SELECT first_name, last_name FROM customer WHERE email = ?", (email,))
    result = cursor.fetchone()
    if not result:
        return {"customer_details": []}
    return {"customer_details": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of orders to a specific country in a specific year
@app.get("/v1/books/order_count_by_country_year", operation_id="get_order_count_by_country_year", summary="Retrieves the total number of orders placed to a specific country in a given year. The operation requires the country name and the year (in 'YYYY' format) as input parameters. The result is a single value representing the total count of orders.")
async def get_order_count_by_country_year(country_name: str = Query(..., description="Name of the country"), year: str = Query(..., description="Year of the order in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM country AS T1 INNER JOIN address AS T2 ON T1.country_id = T2.country_id INNER JOIN cust_order AS T3 ON T3.dest_address_id = T2.address_id WHERE T1.country_name = ? AND STRFTIME('%Y', T3.order_date) = ?", (country_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of orders by a specific customer using a specific shipping method
@app.get("/v1/books/order_count_by_customer_shipping_method", operation_id="get_order_count_by_customer_shipping_method", summary="Retrieves the total number of orders placed by a specific customer using a particular shipping method. The operation requires the customer's first and last names, as well as the name of the shipping method used.")
async def get_order_count_by_customer_shipping_method(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), method_name: str = Query(..., description="Name of the shipping method")):
    cursor.execute("SELECT COUNT(*) FROM customer AS T1 INNER JOIN cust_order AS T2 ON T1.customer_id = T2.customer_id INNER JOIN shipping_method AS T3 ON T3.method_id = T2.shipping_method_id WHERE T1.first_name = ? AND T1.last_name = ? AND T3.method_name = ?", (first_name, last_name, method_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of orders with a specific status by a specific customer
@app.get("/v1/books/order_count_by_status_customer", operation_id="get_order_count_by_status_customer", summary="Retrieves the total number of orders with a specified status for a particular customer. The operation requires the status value of the order, as well as the first and last names of the customer to accurately filter the results.")
async def get_order_count_by_status_customer(status_value: str = Query(..., description="Status value of the order"), first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(*) FROM order_status AS T1 INNER JOIN order_history AS T2 ON T1.status_id = T2.status_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T1.status_value = ? AND T4.first_name = ? AND T4.last_name = ?", (status_value, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most used shipping method
@app.get("/v1/books/most_used_shipping_method", operation_id="get_most_used_shipping_method", summary="Retrieves the most frequently used shipping method based on the order history. The method is determined by aggregating and counting the shipping method IDs associated with each order, then selecting the method with the highest count.")
async def get_most_used_shipping_method():
    cursor.execute("SELECT T2.method_name FROM cust_order AS T1 INNER JOIN shipping_method AS T2 ON T1.shipping_method_id = T2.method_id GROUP BY T2.method_name ORDER BY COUNT(T2.method_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"method_name": []}
    return {"method_name": result[0]}

# Endpoint to get the count of orders with a specific status in a specific year
@app.get("/v1/books/order_count_by_status_year", operation_id="get_order_count_by_status_year", summary="Retrieves the total number of orders that have a specified status in a given year. The status value and year are provided as input parameters to filter the results.")
async def get_order_count_by_status_year(status_value: str = Query(..., description="Status value of the order"), year: str = Query(..., description="Year of the status in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM order_status AS T1 INNER JOIN order_history AS T2 ON T1.status_id = T2.status_id WHERE T1.status_value = ? AND STRFTIME('%Y', T2.status_date) = ?", (status_value, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first book published by a specific author
@app.get("/v1/books/first_book_by_author", operation_id="get_first_book_by_author", summary="Retrieves the title of the earliest published book by the specified author. The author's name is required as an input parameter.")
async def get_first_book_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ? ORDER BY T1.publication_date ASC LIMIT 1", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the publisher name of the earliest book by a specific author
@app.get("/v1/books/publisher_of_earliest_book_by_author", operation_id="get_publisher_of_earliest_book_by_author", summary="Retrieves the name of the publisher that published the earliest book written by the specified author. The author's name is provided as an input parameter.")
async def get_publisher_of_earliest_book_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T4.publisher_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN publisher AS T4 ON T4.publisher_id = T1.publisher_id WHERE T3.author_name = ? ORDER BY T1.publication_date ASC LIMIT 1", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the titles of books by a specific author
@app.get("/v1/books/titles_by_author", operation_id="get_titles_by_author", summary="Retrieve the titles of books written by a specific author. The operation filters books based on the provided author's name and returns a list of corresponding titles.")
async def get_titles_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ?", (author_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of books by a specific author and publisher
@app.get("/v1/books/count_books_by_author_and_publisher", operation_id="get_count_books_by_author_and_publisher", summary="Retrieves the total number of books written by a specific author and published by a certain publisher. The operation requires the author's name and the publisher's name as input parameters.")
async def get_count_books_by_author_and_publisher(author_name: str = Query(..., description="Name of the author"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN publisher AS T4 ON T4.publisher_id = T1.publisher_id WHERE T3.author_name = ? AND T4.publisher_name = ?", (author_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total shipping cost for a specific customer in a specific year
@app.get("/v1/books/total_shipping_cost_by_customer_year", operation_id="get_total_shipping_cost_by_customer_year", summary="Retrieve the total shipping cost incurred by a specific customer in a given year. This operation calculates the sum of shipping costs for all orders placed by the customer in the specified year. The customer is identified by their first and last names, and the year is provided in the 'YYYY' format.")
async def get_total_shipping_cost_by_customer_year(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(T3.cost) FROM customer AS T1 INNER JOIN cust_order AS T2 ON T1.customer_id = T2.customer_id INNER JOIN shipping_method AS T3 ON T3.method_id = T2.shipping_method_id WHERE T1.first_name = ? AND T1.last_name = ? AND STRFTIME('%Y', T2.order_date) = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"total_cost": []}
    return {"total_cost": result[0]}

# Endpoint to get the publisher name by publisher ID
@app.get("/v1/books/publisher_name_by_id", operation_id="get_publisher_name_by_id", summary="Retrieves the name of the publisher associated with the provided publisher ID. This operation allows you to look up the publisher's name using a unique identifier, enabling efficient data retrieval and validation.")
async def get_publisher_name_by_id(publisher_id: int = Query(..., description="ID of the publisher")):
    cursor.execute("SELECT publisher_name FROM publisher WHERE publisher_id = ?", (publisher_id,))
    result = cursor.fetchone()
    if not result:
        return {"publisher_name": []}
    return {"publisher_name": result[0]}

# Endpoint to get the count of books by a specific author with fewer than a specified number of pages
@app.get("/v1/books/count_books_by_author_and_page_limit", operation_id="get_count_books_by_author_and_page_limit", summary="Retrieves the total count of books written by a specific author that have fewer than a specified number of pages. The operation filters books based on the provided author's name and the maximum number of pages, then returns the count of matching books.")
async def get_count_books_by_author_and_page_limit(author_name: str = Query(..., description="Name of the author"), max_pages: int = Query(..., description="Maximum number of pages")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ? AND T1.num_pages < ?", (author_name, max_pages))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the author and publisher names of books published on a specific date
@app.get("/v1/books/author_publisher_by_publication_date", operation_id="get_author_publisher_by_publication_date", summary="Retrieves the names of authors and publishers associated with books published on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_author_publisher_by_publication_date(publication_date: str = Query(..., description="Publication date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T3.author_name, T4.publisher_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN publisher AS T4 ON T4.publisher_id = T1.publisher_id WHERE T1.publication_date = ?", (publication_date,))
    result = cursor.fetchall()
    if not result:
        return {"author_publisher": []}
    return {"author_publisher": [{"author_name": row[0], "publisher_name": row[1]} for row in result]}

# Endpoint to get the language name of a book by ISBN-13
@app.get("/v1/books/language_by_isbn13", operation_id="get_language_by_isbn13", summary="Retrieves the language name of a book using its unique ISBN-13 identifier. The operation returns the language name associated with the provided ISBN-13, enabling users to identify the language of the book.")
async def get_language_by_isbn13(isbn13: str = Query(..., description="ISBN-13 of the book")):
    cursor.execute("SELECT T2.language_name FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id WHERE T1.isbn13 = ?", (isbn13,))
    result = cursor.fetchone()
    if not result:
        return {"language_name": []}
    return {"language_name": result[0]}

# Endpoint to get the title of the most expensive book in an order
@app.get("/v1/books/most_expensive_book_title", operation_id="get_most_expensive_book_title", summary="Retrieves the title of the book with the highest price from the order line. The operation returns the title of the most expensive book based on the order line data.")
async def get_most_expensive_book_title():
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id ORDER BY T2.price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get ISBN13 of books based on language name
@app.get("/v1/books/isbn13_by_language", operation_id="get_isbn13_by_language", summary="Retrieves the ISBN13 of books written in a specified language. The operation filters books by their language name and returns the corresponding ISBN13 values.")
async def get_isbn13_by_language(language_name: str = Query(..., description="Language name of the book")):
    cursor.execute("SELECT T1.isbn13 FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id WHERE T2.language_name = ?", (language_name,))
    result = cursor.fetchall()
    if not result:
        return {"isbn13": []}
    return {"isbn13": [row[0] for row in result]}

# Endpoint to get the count of books from a publisher with a price less than a specified value
@app.get("/v1/books/count_books_by_publisher_price", operation_id="get_count_books_by_publisher_price", summary="Retrieve the number of books from a specific publisher that are priced below a certain threshold. This operation requires the publisher's name and the maximum price value as input parameters.")
async def get_count_books_by_publisher_price(publisher_name: str = Query(..., description="Name of the publisher"), max_price: float = Query(..., description="Maximum price of the book")):
    cursor.execute("SELECT COUNT(*) FROM publisher AS T1 INNER JOIN book AS T2 ON T1.publisher_id = T2.publisher_id INNER JOIN order_line AS T3 ON T3.book_id = T2.book_id WHERE T1.publisher_name = ? AND T3.price < ?", (publisher_name, max_price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the author of the book with the highest number of pages
@app.get("/v1/books/author_of_longest_book", operation_id="get_author_of_longest_book", summary="Retrieves the name of the author who wrote the book with the most pages. This operation involves querying the book, book_author, and author tables to find the author with the highest page count.")
async def get_author_of_longest_book():
    cursor.execute("SELECT T3.author_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id ORDER BY T1.num_pages DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"author_name": []}
    return {"author_name": result[0]}

# Endpoint to get customer emails who ordered a specific book title
@app.get("/v1/books/customer_emails_by_book_title", operation_id="get_customer_emails_by_book_title", summary="Retrieves the email addresses of customers who have ordered a book with the specified title. The operation filters the customer database based on the provided book title and returns the corresponding email addresses.")
async def get_customer_emails_by_book_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T4.email FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get author names for books published by a specific publisher
@app.get("/v1/books/authors_by_publisher", operation_id="get_authors_by_publisher", summary="Retrieves the names of authors who have published books with a specific publisher. The operation filters authors based on the provided publisher name.")
async def get_authors_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T3.author_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN publisher AS T4 ON T4.publisher_id = T1.publisher_id WHERE T4.publisher_name = ?", (publisher_name,))
    result = cursor.fetchall()
    if not result:
        return {"author_names": []}
    return {"author_names": [row[0] for row in result]}

# Endpoint to get the percentage of books published in a specific year by a specific author
@app.get("/v1/books/percentage_books_by_year_author", operation_id="get_percentage_books_by_year_author", summary="Retrieves the percentage of books authored by a specific writer and published in a given year. The calculation is based on the total number of books written by the author across all years.")
async def get_percentage_books_by_year_author(year: str = Query(..., description="Year of publication in 'YYYY' format"), author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN STRFTIME('%Y', T1.publication_date) = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ?", (year, author_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get book titles and publisher names for books published in a specific year with pages above a certain threshold
@app.get("/v1/books/titles_publishers_by_year_pages", operation_id="get_titles_publishers_by_year_pages", summary="Retrieve the titles and corresponding publisher names of books published in a specified year, with page counts exceeding the average by at least 70%. The year of publication is provided as input.")
async def get_titles_publishers_by_year_pages(year: str = Query(..., description="Year of publication in 'YYYY' format")):
    cursor.execute("SELECT T1.title, T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE STRFTIME('%Y', T1.publication_date) = ? AND T1.num_pages * 100 > ( SELECT AVG(num_pages) FROM book ) * 70", (year,))
    result = cursor.fetchall()
    if not result:
        return {"books": []}
    return {"books": [{"title": row[0], "publisher_name": row[1]} for row in result]}

# Endpoint to get customer emails based on first and last name
@app.get("/v1/books/customer_emails_by_name", operation_id="get_customer_emails_by_name", summary="Retrieves the email addresses of customers who match the provided first and last names. The operation uses the specified names to filter the customer database and return the corresponding email addresses.")
async def get_customer_emails_by_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT email FROM customer WHERE first_name = ? AND last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get street names based on city
@app.get("/v1/books/street_names_by_city", operation_id="get_street_names_by_city", summary="Retrieves a list of street names within the specified city. The operation filters the address records based on the provided city name and returns the corresponding street names.")
async def get_street_names_by_city(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT street_name FROM address WHERE city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": [row[0] for row in result]}

# Endpoint to get book titles by author name and publication year
@app.get("/v1/books/titles_by_author_and_year", operation_id="get_titles_by_author_and_year", summary="Retrieves the titles of books written by a specific author and published in a given year. The author's name and the four-digit publication year are required as input parameters.")
async def get_titles_by_author_and_year(author_name: str = Query(..., description="Name of the author"), publication_year: str = Query(..., description="Publication year in 'YYYY' format")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ? AND STRFTIME('%Y', T1.publication_date) = ?", (author_name, publication_year))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of books by author name
@app.get("/v1/books/count_by_author", operation_id="get_count_by_author", summary="Retrieves the total number of books written by a specific author. The author is identified by their name, which is provided as an input parameter.")
async def get_count_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ?", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get author names and book titles by minimum number of pages
@app.get("/v1/books/author_titles_by_min_pages", operation_id="get_author_titles_by_min_pages", summary="Retrieves a list of author names and their respective book titles where the number of pages in the book exceeds the specified minimum. The minimum number of pages is provided as an input parameter.")
async def get_author_titles_by_min_pages(min_pages: int = Query(..., description="Minimum number of pages")):
    cursor.execute("SELECT T3.author_name, T1.title FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T1.num_pages > ?", (min_pages,))
    result = cursor.fetchall()
    if not result:
        return {"author_titles": []}
    return {"author_titles": [{"author_name": row[0], "title": row[1]} for row in result]}

# Endpoint to get author names by book title
@app.get("/v1/books/authors_by_title", operation_id="get_authors_by_title", summary="Retrieves the names of authors associated with a specific book title. The operation searches for the book by its title and returns the corresponding author names.")
async def get_authors_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T3.author_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": [row[0] for row in result]}

# Endpoint to get publisher names by author name
@app.get("/v1/books/publishers_by_author", operation_id="get_publishers_by_author", summary="Retrieves the names of publishers associated with a specific author. The author's name is used to search for corresponding books, and subsequently, the publishers of those books are identified.")
async def get_publishers_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T4.publisher_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN publisher AS T4 ON T4.publisher_id = T1.publisher_id WHERE T3.author_name = ?", (author_name,))
    result = cursor.fetchall()
    if not result:
        return {"publishers": []}
    return {"publishers": [row[0] for row in result]}

# Endpoint to get the count of books by language name
@app.get("/v1/books/count_by_language", operation_id="get_count_by_language", summary="Retrieves the total number of books written in a specified language. The language is identified by its name, which is provided as an input parameter.")
async def get_count_by_language(language_name: str = Query(..., description="Name of the language")):
    cursor.execute("SELECT COUNT(T2.book_id) FROM book_language AS T1 INNER JOIN book AS T2 ON T1.language_id = T2.language_id WHERE T1.language_name = ?", (language_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total price of a book by title
@app.get("/v1/books/total_price_by_title", operation_id="get_total_price_by_title", summary="Retrieves the total price of a book by its title. This operation calculates the sum of all prices for a specific book title from the order line data, providing a comprehensive view of the book's total sales value.")
async def get_total_price_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT SUM(T1.price) FROM order_line AS T1 INNER JOIN book AS T2 ON T1.book_id = T2.book_id WHERE T2.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get the count of orders by customer name and order year
@app.get("/v1/books/count_orders_by_customer_and_year", operation_id="get_count_orders_by_customer_and_year", summary="Retrieves the total number of orders placed by a specific customer in a given year. The operation requires the customer's first and last name, as well as the year of the orders to be counted. The year should be provided in the 'YYYY' format.")
async def get_count_orders_by_customer_and_year(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), order_year: str = Query(..., description="Order year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM cust_order AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id WHERE T2.first_name = ? AND T2.last_name = ? AND STRFTIME('%Y', T1.order_date) = ?", (first_name, last_name, order_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names who ordered a specific book title
@app.get("/v1/books/customer_names_by_book_title", operation_id="get_customer_names_by_book_title", summary="Retrieves the first and last names of customers who have ordered a book with the specified title. The operation filters the customer records based on the provided book title.")
async def get_customer_names_by_book_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT T4.first_name, T4.last_name FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the percentage of orders using a specific shipping method in a given year
@app.get("/v1/books/percentage_orders_by_shipping_method_year", operation_id="get_percentage_orders_by_shipping_method_year", summary="Retrieves the percentage of orders that used a specified shipping method in a given year. The calculation is based on the total number of orders placed in the specified year. The shipping method is identified by its name, and the year is provided in the 'YYYY' format.")
async def get_percentage_orders_by_shipping_method_year(method_name: str = Query(..., description="Name of the shipping method"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.method_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM cust_order AS T1 INNER JOIN shipping_method AS T2 ON T1.shipping_method_id = T2.method_id WHERE STRFTIME('%Y', T1.order_date) = ?", (method_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get author names that start with a specific prefix
@app.get("/v1/books/author_names_by_prefix", operation_id="get_author_names_by_prefix", summary="Retrieves a list of author names that start with the provided prefix. The endpoint filters the author names based on the specified prefix and returns the matching results.")
async def get_author_names_by_prefix(prefix: str = Query(..., description="Prefix of the author name")):
    cursor.execute("SELECT author_name FROM author WHERE author_name LIKE ?", (prefix + '%',))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get the year with the most orders
@app.get("/v1/books/year_with_most_orders", operation_id="get_year_with_most_orders", summary="Retrieves the year in which the highest number of orders were placed. The operation calculates the total orders for each year and returns the year with the maximum count.")
async def get_year_with_most_orders():
    cursor.execute("SELECT strftime('%Y', order_date) FROM cust_order GROUP BY strftime('%Y', order_date) ORDER BY COUNT(strftime('%Y', order_date)) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the average price of all order lines
@app.get("/v1/books/average_order_line_price", operation_id="get_average_order_line_price", summary="Retrieves the average price of all order lines from the database. This operation provides a single value representing the mean price of all order lines, which can be used for various analytical purposes.")
async def get_average_order_line_price():
    cursor.execute("SELECT AVG(price) FROM order_line")
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get book titles published in a specific year
@app.get("/v1/books/book_titles_by_publication_year", operation_id="get_book_titles_by_publication_year", summary="Retrieves a list of book titles that were published in a specific year. The year should be provided in 'YYYY' format.")
async def get_book_titles_by_publication_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT title FROM book WHERE STRFTIME('%Y', publication_date) = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the most common email domain among customers
@app.get("/v1/books/most_common_email_domain", operation_id="get_most_common_email_domain", summary="Retrieves the most frequently occurring email domain among customers. This operation identifies the email domain that appears most frequently in the customer database and returns it. The result provides insights into the most common email domain used by customers.")
async def get_most_common_email_domain():
    cursor.execute("SELECT SUBSTR(email, INSTR(email, '@') + 1, LENGTH(email) - INSTR(email, '@')) AS ym FROM customer GROUP BY SUBSTR(email, INSTR(email, '@') + 1, LENGTH(email) - INSTR(email, '@')) ORDER BY COUNT(*) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"domain": []}
    return {"domain": result[0]}

# Endpoint to get the count of publishers with names containing a specific substring
@app.get("/v1/books/publisher_count_by_name_substring", operation_id="get_publisher_count_by_name_substring", summary="Retrieves the total number of publishers whose names contain a specified substring. The substring is used to filter the publisher names and count the number of matches.")
async def get_publisher_count_by_name_substring(substring: str = Query(..., description="Substring to search in publisher names")):
    cursor.execute("SELECT COUNT(*) FROM publisher WHERE publisher_name LIKE ?", ('%' + substring + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the least common language of books
@app.get("/v1/books/least_common_book_language", operation_id="get_least_common_book_language", summary="Retrieves the least frequently used language among all books. This operation identifies the language with the fewest associated books by joining the book and book_language tables, grouping by language name, and ordering by the count of language names in ascending order. The language name of the least common language is then returned.")
async def get_least_common_book_language():
    cursor.execute("SELECT T2.language_name FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id GROUP BY T2.language_name ORDER BY COUNT(T2.language_name) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"language": []}
    return {"language": result[0]}

# Endpoint to get order dates for a specific customer
@app.get("/v1/books/order_dates_by_customer", operation_id="get_order_dates_by_customer", summary="Retrieves the order dates for a specific customer identified by their first and last name. This operation returns a list of dates when the customer placed orders for books. The input parameters are used to filter the results by the customer's first and last name.")
async def get_order_dates_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T3.order_date FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T4.first_name = ? AND T4.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"order_dates": []}
    return {"order_dates": [row[0] for row in result]}

# Endpoint to get the most prolific author
@app.get("/v1/books/most_prolific_author", operation_id="get_most_prolific_author", summary="Retrieves the name of the author who has written the most books. This operation identifies the author with the highest number of published books by joining the author and book_author tables and grouping by author name. The result is ordered in descending order based on the count of author IDs, with the top result being returned.")
async def get_most_prolific_author():
    cursor.execute("SELECT T1.author_name FROM author AS T1 INNER JOIN book_author AS T2 ON T1.author_id = T2.author_id GROUP BY T1.author_name ORDER BY COUNT(T2.author_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"author_name": []}
    return {"author_name": result[0]}

# Endpoint to get book titles from a specific publisher
@app.get("/v1/books/book_titles_by_publisher", operation_id="get_book_titles_by_publisher", summary="Retrieves a list of book titles published by a specific publisher. The operation filters books based on the provided publisher name and returns their corresponding titles.")
async def get_book_titles_by_publisher(publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ?", (publisher_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the least used shipping method
@app.get("/v1/books/least_used_shipping_method", operation_id="get_least_used_shipping_method", summary="Get the least used shipping method")
async def get_least_used_shipping_method():
    cursor.execute("SELECT T2.method_name FROM cust_order AS T1 INNER JOIN shipping_method AS T2 ON T1.shipping_method_id = T2.method_id GROUP BY T2.method_name ORDER BY COUNT(T2.method_id) ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"method_name": []}
    return {"method_name": result[0]}

# Endpoint to get the count of addresses with a specific status
@app.get("/v1/books/count_addresses_by_status", operation_id="get_count_addresses_by_status", summary="Retrieves the total number of customer addresses that have a specified status. The status is provided as an input parameter.")
async def get_count_addresses_by_status(address_status: str = Query(..., description="Status of the address")):
    cursor.execute("SELECT COUNT(*) FROM customer_address AS T1 INNER JOIN address_status AS T2 ON T1.status_id = T2.status_id WHERE T2.address_status = ?", (address_status,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most ordered book title
@app.get("/v1/books/most_ordered_book_title", operation_id="get_most_ordered_book_title", summary="Retrieves the title of the book that has been ordered the most. This operation returns the title of the book with the highest order count, based on the order_line and book tables. The result is determined by grouping orders by book title and counting the number of occurrences for each title, then selecting the title with the highest count.")
async def get_most_ordered_book_title():
    cursor.execute("SELECT T2.title FROM order_line AS T1 INNER JOIN book AS T2 ON T1.book_id = T2.book_id GROUP BY T2.title ORDER BY COUNT(T1.book_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the most common destination address
@app.get("/v1/books/most_common_destination_address", operation_id="get_most_common_destination_address", summary="Retrieves the most frequently used destination address from the order database. The response includes the street name and city of the most common destination address.")
async def get_most_common_destination_address():
    cursor.execute("SELECT T2.street_name, T2.city FROM cust_order AS T1 INNER JOIN address AS T2 ON T1.dest_address_id = T2.address_id GROUP BY T2.street_number, T2.street_name, T2.city ORDER BY COUNT(T1.dest_address_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"street_name": [], "city": []}
    return {"street_name": result[0], "city": result[1]}

# Endpoint to get the difference in days between order date and status date for a specific order
@app.get("/v1/books/order_status_date_difference", operation_id="get_order_status_date_difference", summary="Get the difference in days between order date and status date for a specific order")
async def get_order_status_date_difference(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT strftime('%J', T2.status_date) - strftime('%J', T1.order_date) FROM cust_order AS T1 INNER JOIN order_history AS T2 ON T1.order_id = T2.order_id WHERE T1.order_id = ?", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the most frequent customer names based on the number of addresses
@app.get("/v1/books/customers/most_frequent_customer_names", operation_id="get_most_frequent_customer_names", summary="Retrieve a list of the most frequent customer names, ranked by the number of associated addresses. The response is limited to the specified number of results.")
async def get_most_frequent_customer_names(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.first_name, T1.last_name FROM customer AS T1 INNER JOIN customer_address AS T2 ON T1.customer_id = T2.customer_id GROUP BY T1.first_name, T1.last_name ORDER BY COUNT(T2.customer_id) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": result}

# Endpoint to get the percentage of orders shipped by a specific method
@app.get("/v1/books/orders/shipping_method_percentage", operation_id="get_shipping_method_percentage", summary="Retrieves the percentage of orders shipped using a specified shipping method. This operation calculates the proportion of orders shipped via the given method by comparing the count of orders shipped using that method to the total number of orders.")
async def get_shipping_method_percentage(method_name: str = Query(..., description="Shipping method name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.method_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM cust_order AS T1 INNER JOIN order_line AS T2 ON T1.order_id = T2.order_id INNER JOIN shipping_method AS T3 ON T3.method_id = T1.shipping_method_id", (method_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get authors of books with fewer pages than the average
@app.get("/v1/books/authors_below_avg_pages", operation_id="get_authors_below_avg_pages", summary="Retrieves the names of authors who have written books with a page count less than the average page count of all books in the database. This operation does not require any input parameters and returns a list of author names.")
async def get_authors_below_avg_pages():
    cursor.execute("SELECT T3.author_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T1.num_pages < ( SELECT AVG(num_pages) FROM book )")
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": result}

# Endpoint to get the minimum price of order lines excluding a specific price
@app.get("/v1/books/order_lines/min_price_excluding", operation_id="get_min_price_excluding", summary="Retrieves the lowest price among all order lines, excluding a specified price. This operation is useful for identifying the second lowest price or determining the minimum price when a certain price is not applicable.")
async def get_min_price_excluding(excluded_price: float = Query(..., description="Price to be excluded")):
    cursor.execute("SELECT MIN(price) FROM order_line WHERE price <> ?", (excluded_price,))
    result = cursor.fetchone()
    if not result:
        return {"min_price": []}
    return {"min_price": result[0]}

# Endpoint to get the count of addresses in a specific city
@app.get("/v1/books/addresses/count_by_city", operation_id="get_count_by_city", summary="Retrieves the total number of addresses located in a specified city.")
async def get_count_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(address_id) FROM address WHERE city = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of authors with names starting with a specific prefix
@app.get("/v1/books/authors/count_by_name_prefix", operation_id="get_count_by_name_prefix", summary="Retrieves the total number of authors whose names begin with a specified prefix. This operation is useful for determining the quantity of authors that meet a certain naming criterion.")
async def get_count_by_name_prefix(name_prefix: str = Query(..., description="Prefix of the author name")):
    cursor.execute("SELECT COUNT(*) FROM author WHERE author_name LIKE ?", (name_prefix + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers with emails ending with a specific suffix
@app.get("/v1/books/customers/count_by_email_suffix", operation_id="get_count_by_email_suffix", summary="Retrieves the total number of customers whose email addresses end with a specified suffix. This operation is useful for identifying the prevalence of certain email domains among customers.")
async def get_count_by_email_suffix(email_suffix: str = Query(..., description="Suffix of the email address")):
    cursor.execute("SELECT COUNT(*) FROM customer WHERE email LIKE ?", ('%' + email_suffix,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct cities in a specific country
@app.get("/v1/books/countries/distinct_cities", operation_id="get_distinct_cities", summary="Retrieves a list of unique cities within a specified country. The operation filters the address data based on the provided country name and returns only the distinct city names.")
async def get_distinct_cities(country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT DISTINCT T2.city FROM country AS T1 INNER JOIN address AS T2 ON T1.country_id = T2.country_id WHERE T1.country_name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": result}

# Endpoint to get the count of orders for a specific customer with a specific shipping method in a specific year
@app.get("/v1/books/customers/order_count_by_customer_shipping_year", operation_id="get_order_count_by_customer_shipping_year", summary="Retrieves the total number of orders placed by a specific customer using a particular shipping method in a given year. The customer is identified by their first and last names, and the year is specified in the 'YYYY' format. The shipping method is determined by its name.")
async def get_order_count_by_customer_shipping_year(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), order_year: str = Query(..., description="Year of the order in 'YYYY' format"), method_name: str = Query(..., description="Shipping method name")):
    cursor.execute("SELECT COUNT(*) FROM customer AS T1 INNER JOIN cust_order AS T2 ON T1.customer_id = T2.customer_id INNER JOIN shipping_method AS T3 ON T3.method_id = T2.shipping_method_id WHERE T1.first_name = ? AND T1.last_name = ? AND STRFTIME('%Y', T2.order_date) = ? AND T3.method_name = ?", (first_name, last_name, order_year, method_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum price of a book with a specific title
@app.get("/v1/books/max_price_by_title", operation_id="get_max_price_by_title", summary="Retrieves the highest price at which a book with the specified title has been sold. The title of the book is required as an input parameter.")
async def get_max_price_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT MAX(T2.price) FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"max_price": []}
    return {"max_price": result[0]}

# Endpoint to get the count of books grouped by publication date and ordered by publication date in ascending order
@app.get("/v1/books/count_books_by_publication_date", operation_id="get_count_books_by_publication_date", summary="Retrieves the total number of books sold, grouped by their respective publication dates and arranged in ascending order. The result provides a comprehensive overview of book sales distribution across different publication dates.")
async def get_count_books_by_publication_date():
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id GROUP BY T1.publication_date ORDER BY T1.publication_date ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get book titles based on language and publisher
@app.get("/v1/books/titles_by_language_and_publisher", operation_id="get_titles_by_language_and_publisher", summary="Retrieves a list of unique book titles published by a specific publisher and written in a certain language. The operation filters books based on the provided language and publisher names.")
async def get_titles_by_language_and_publisher(language_name: str = Query(..., description="Language name"), publisher_name: str = Query(..., description="Publisher name")):
    cursor.execute("SELECT T2.title FROM book_language AS T1 INNER JOIN book AS T2 ON T2.language_id = T1.language_id INNER JOIN publisher AS T3 ON T3.publisher_id = T2.publisher_id WHERE T1.language_name = ? AND T3.publisher_name = ? GROUP BY T2.title", (language_name, publisher_name))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of publication dates for books by a specific author
@app.get("/v1/books/count_publication_dates_by_author", operation_id="get_count_publication_dates_by_author", summary="Retrieve the number of distinct publication dates for books written by a specific author. The author is identified by their name. The count is ordered by the publication dates in ascending order and limited to the earliest date.")
async def get_count_publication_dates_by_author(author_name: str = Query(..., description="Author name")):
    cursor.execute("SELECT COUNT(T1.publication_date) FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN order_line AS T4 ON T4.book_id = T1.book_id WHERE T3.author_name = ? ORDER BY T1.publication_date ASC LIMIT 1", (author_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the language names of the first two books ordered by publication date
@app.get("/v1/books/language_names_by_publication_date", operation_id="get_language_names_by_publication_date", summary="Retrieves the names of the languages used in the first two books, sorted by their publication date in ascending order.")
async def get_language_names_by_publication_date():
    cursor.execute("SELECT T2.language_name FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id ORDER BY T1.publication_date ASC LIMIT 2")
    result = cursor.fetchall()
    if not result:
        return {"language_names": []}
    return {"language_names": [row[0] for row in result]}

# Endpoint to get distinct publisher names for a specific book title
@app.get("/v1/books/distinct_publishers_by_title", operation_id="get_distinct_publishers_by_title", summary="Retrieves a list of unique publisher names associated with a specific book title. The operation filters the book records based on the provided title and returns the distinct publisher names linked to those books.")
async def get_distinct_publishers_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT DISTINCT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T1.title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"publishers": []}
    return {"publishers": [row[0] for row in result]}

# Endpoint to get the count of books by a specific publisher and author
@app.get("/v1/books/count_books_by_publisher_and_author", operation_id="get_count_books_by_publisher_and_author", summary="Retrieves the total number of books published by a specific publisher and authored by a specific author. The operation requires the publisher's name and the author's name as input parameters.")
async def get_count_books_by_publisher_and_author(publisher_name: str = Query(..., description="Publisher name"), author_name: str = Query(..., description="Author name")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id INNER JOIN book_author AS T3 ON T3.book_id = T1.book_id INNER JOIN author AS T4 ON T4.author_id = T3.author_id WHERE T2.publisher_name = ? AND T4.author_name = ?", (publisher_name, author_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get publisher names with a minimum number of books
@app.get("/v1/books/publishers_with_min_books", operation_id="get_publishers_with_min_books", summary="Retrieves the names of publishers who have published at least a specified minimum number of books. The minimum number of books is provided as an input parameter.")
async def get_publishers_with_min_books(min_books: int = Query(..., description="Minimum number of books")):
    cursor.execute("SELECT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id GROUP BY T2.publisher_name HAVING COUNT(T2.publisher_name) >= ?", (min_books,))
    result = cursor.fetchall()
    if not result:
        return {"publishers": []}
    return {"publishers": [row[0] for row in result]}

# Endpoint to get all street numbers from the address table
@app.get("/v1/books/addresses/street_numbers", operation_id="get_street_numbers", summary="Retrieves a comprehensive list of all street numbers from the address records. This operation does not require any input parameters and returns a collection of unique street numbers.")
async def get_street_numbers():
    cursor.execute("SELECT street_number FROM address")
    result = cursor.fetchall()
    if not result:
        return {"street_numbers": []}
    return {"street_numbers": [row[0] for row in result]}

# Endpoint to get address details based on city
@app.get("/v1/books/address_details_by_city", operation_id="get_address_details", summary="Retrieves address details, including street number, street name, and country ID, for a specific city. The city name is provided as an input parameter.")
async def get_address_details(city: str = Query(..., description="City name")):
    cursor.execute("SELECT street_number, street_name, city, country_id FROM address WHERE city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get ISBN13 of books within a specific page range
@app.get("/v1/books/isbn13_by_page_range", operation_id="get_isbn13_by_page_range", summary="Retrieves the ISBN13 of books that have a number of pages within the specified range. The range is defined by a minimum and maximum number of pages, which are provided as input parameters. This operation returns a list of ISBN13s that meet the specified criteria.")
async def get_isbn13_by_page_range(max_pages: int = Query(..., description="Maximum number of pages"), min_pages: int = Query(..., description="Minimum number of pages")):
    cursor.execute("SELECT isbn13 FROM book WHERE num_pages < ? AND num_pages > ?", (max_pages, min_pages))
    result = cursor.fetchall()
    if not result:
        return {"isbn13_list": []}
    return {"isbn13_list": result}

# Endpoint to get book titles ordered by publication date
@app.get("/v1/books/book_titles_by_publication_date", operation_id="get_book_titles", summary="Retrieve a specified number of book titles, sorted by their publication date in ascending order.")
async def get_book_titles(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT title FROM book ORDER BY publication_date ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"book_titles": []}
    return {"book_titles": result}

# Endpoint to get the count of orders based on order date pattern
@app.get("/v1/books/order_count_by_date_pattern", operation_id="get_order_count", summary="Retrieves the total number of orders placed in a specific month, as determined by the provided date pattern in 'YYYY-MM%' format.")
async def get_order_count(order_date_pattern: str = Query(..., description="Order date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT COUNT(*) FROM cust_order WHERE order_date LIKE ?", (order_date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names based on last name pattern
@app.get("/v1/books/customer_names_by_last_name", operation_id="get_customer_names", summary="Retrieves the first and last names of customers whose last names match the provided pattern. The pattern should be provided in the format 'K%' to match last names starting with 'K'.")
async def get_customer_names(last_name_pattern: str = Query(..., description="Last name pattern in 'K%' format")):
    cursor.execute("SELECT first_name, last_name FROM customer WHERE last_name LIKE ?", (last_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": result}

# Endpoint to get cities based on country name
@app.get("/v1/books/cities_by_country_name", operation_id="get_cities_by_country", summary="Retrieves a list of cities located within a specified country. The operation filters the cities based on the provided country name.")
async def get_cities_by_country(country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT T1.city FROM address AS T1 INNER JOIN country AS T2 ON T2.country_id = T1.country_id WHERE T2.country_name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": result}

# Endpoint to get distinct street names based on address status
@app.get("/v1/books/street_names_by_address_status", operation_id="get_street_names", summary="Retrieves a unique list of street names associated with a given address status. This operation filters addresses based on their status and returns the distinct street names linked to them.")
async def get_street_names(address_status: str = Query(..., description="Address status")):
    cursor.execute("SELECT DISTINCT T1.street_name FROM address AS T1 INNER JOIN customer_address AS T2 ON T1.address_id = T2.address_id INNER JOIN address_status AS T3 ON T3.status_id = T2.status_id WHERE T3.address_status = ?", (address_status,))
    result = cursor.fetchall()
    if not result:
        return {"street_names": []}
    return {"street_names": result}

# Endpoint to get customer names based on city
@app.get("/v1/books/customer_names_by_city", operation_id="get_customer_names_by_city", summary="Retrieves the first and last names of customers who reside in the specified city. The city is provided as an input parameter.")
async def get_customer_names_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T3.first_name, T3.last_name FROM address AS T1 INNER JOIN customer_address AS T2 ON T1.address_id = T2.address_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T1.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": result}

# Endpoint to get customer emails based on shipping method
@app.get("/v1/books/customer_emails_by_shipping_method", operation_id="get_customer_emails", summary="Retrieves the email addresses of customers who have used a specific shipping method. The shipping method is identified by its name, which is provided as an input parameter.")
async def get_customer_emails(method_name: str = Query(..., description="Shipping method name")):
    cursor.execute("SELECT T1.email FROM customer AS T1 INNER JOIN cust_order AS T2 ON T1.customer_id = T2.customer_id INNER JOIN shipping_method AS T3 ON T3.method_id = T2.shipping_method_id WHERE T3.method_name = ?", (method_name,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": result}

# Endpoint to get order dates based on street number
@app.get("/v1/books/order_dates_by_street_number", operation_id="get_order_dates", summary="Retrieves the dates of orders associated with a given street number. This operation fetches the order dates from the database by matching the provided street number with the corresponding address in the address table. The result is a list of order dates for orders delivered to the specified street number.")
async def get_order_dates(street_number: int = Query(..., description="Street number")):
    cursor.execute("SELECT T1.order_date FROM cust_order AS T1 INNER JOIN address AS T2 ON T1.dest_address_id = T2.address_id WHERE T2.street_number = ?", (street_number,))
    result = cursor.fetchall()
    if not result:
        return {"order_dates": []}
    return {"order_dates": result}

# Endpoint to get order IDs based on order status
@app.get("/v1/books/order_ids_by_status", operation_id="get_order_ids_by_status", summary="Retrieves a list of order IDs that match the specified order status value. The operation filters order history based on the provided status value and returns the corresponding order IDs.")
async def get_order_ids_by_status(status_value: str = Query(..., description="Status value of the order")):
    cursor.execute("SELECT T2.order_id FROM order_status AS T1 INNER JOIN order_history AS T2 ON T1.status_id = T2.status_id WHERE T1.status_value = ?", (status_value,))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get distinct order status values based on order date
@app.get("/v1/books/distinct_status_values_by_order_date", operation_id="get_distinct_status_values_by_order_date", summary="Retrieves the unique order status values associated with orders placed on a specific date. The operation filters order statuses based on the provided order date, which should be in the 'YYYY-MM-DD%' format.")
async def get_distinct_status_values_by_order_date(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT DISTINCT T1.status_value FROM order_status AS T1 INNER JOIN order_history AS T2 ON T1.status_id = T2.status_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id WHERE T3.order_date LIKE ?", (order_date,))
    result = cursor.fetchall()
    if not result:
        return {"status_values": []}
    return {"status_values": [row[0] for row in result]}

# Endpoint to get order dates based on the price of the order line
@app.get("/v1/books/order_dates_by_price", operation_id="get_order_dates_by_price", summary="Retrieves the order dates for orders containing an order line with the specified price. The operation filters the order lines based on the provided price and returns the corresponding order dates.")
async def get_order_dates_by_price(price: float = Query(..., description="Price of the order line")):
    cursor.execute("SELECT T1.order_date FROM cust_order AS T1 INNER JOIN order_line AS T2 ON T1.order_id = T2.order_id WHERE T2.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"order_dates": []}
    return {"order_dates": [row[0] for row in result]}

# Endpoint to get the percentage of orders shipped internationally based on customer's first name
@app.get("/v1/books/percentage_international_shipping_by_first_name", operation_id="get_percentage_international_shipping_by_first_name", summary="Retrieves the percentage of orders shipped using a specified method for customers with a given first name. This operation calculates the proportion of international shipments by comparing the count of orders shipped using the specified method to the total number of orders for customers with the provided first name.")
async def get_percentage_international_shipping_by_first_name(method_name: str = Query(..., description="Shipping method name"), first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.method_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM customer AS T1 INNER JOIN cust_order AS T2 ON T1.customer_id = T2.customer_id INNER JOIN shipping_method AS T3 ON T3.method_id = T2.shipping_method_id WHERE T1.first_name = ?", (method_name, first_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the address of a specific customer
@app.get("/v1/books/customer_address", operation_id="get_customer_address", summary="Retrieves the street number, street name, and city of a customer's address using their first and last name. The operation joins multiple tables to accurately locate the customer and their corresponding address details.")
async def get_customer_address(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T3.street_number, T3.street_name, T3.city FROM customer AS T1 INNER JOIN customer_address AS T2 ON T1.customer_id = T2.customer_id INNER JOIN address AS T3 ON T3.address_id = T2.address_id INNER JOIN country AS T4 ON T4.country_id = T3.country_id WHERE T1.first_name = ? AND T1.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [{"street_number": row[0], "street_name": row[1], "city": row[2]} for row in result]}

# Endpoint to get the publisher of the earliest published book
@app.get("/v1/books/earliest_publisher", operation_id="get_earliest_publisher", summary="Retrieves the name of the publisher associated with the book that has the earliest publication date. This operation returns the publisher's name as a single result.")
async def get_earliest_publisher():
    cursor.execute("SELECT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id ORDER BY T1.publication_date ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"publisher": []}
    return {"publisher": result[0]}

# Endpoint to get the difference in cost between Priority and Express shipping methods
@app.get("/v1/books/shipping_cost_difference", operation_id="get_shipping_cost_difference", summary="Retrieves the total cost difference between the Priority and Express shipping methods. This operation calculates the sum of costs for each shipping method and returns the difference, providing a comparison of their overall expenses.")
async def get_shipping_cost_difference():
    cursor.execute("SELECT SUM(CASE WHEN method_name = 'Priority' THEN cost ELSE 0 END) - SUM(CASE WHEN method_name = 'Express' THEN cost ELSE 0 END) FROM shipping_method")
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of books published in a specific year
@app.get("/v1/books/count_books_by_year", operation_id="get_count_books_by_year", summary="Retrieves the total number of books published in a given year. The year must be provided in the 'YYYY' format.")
async def get_count_books_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM book WHERE STRFTIME('%Y', publication_date) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ISBN13 of a book by its title
@app.get("/v1/books/isbn13_by_title", operation_id="get_isbn13_by_title", summary="Retrieves the ISBN13 of a book by providing its title. The operation searches for a book with the given title and returns its corresponding ISBN13.")
async def get_isbn13_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT isbn13 FROM book WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"isbn13": []}
    return {"isbn13": result[0]}

# Endpoint to get the percentage of orders with a specific status in a given year
@app.get("/v1/books/order_status_percentage", operation_id="get_order_status_percentage", summary="Retrieves the percentage of orders with a specific status that occurred in a given year. The status value and year are provided as input parameters to filter the results. The calculation is based on the total count of orders with the specified status in the given year, divided by the total count of all orders in that year.")
async def get_order_status_percentage(status_value: str = Query(..., description="Status value of the order"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.status_value = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM order_status AS T1 INNER JOIN order_history AS T2 ON T1.status_id = T2.status_id WHERE STRFTIME('%Y', T2.status_date) = ?", (status_value, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of addresses with a specific status
@app.get("/v1/books/address_status_percentage", operation_id="get_address_status_percentage", summary="Retrieves the percentage of customer addresses that have a specific status. The status is provided as an input parameter, and the calculation is based on the total number of customer addresses in the database.")
async def get_address_status_percentage(address_status: str = Query(..., description="Status of the address")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.address_status = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM customer_address AS T1 INNER JOIN address_status AS T2 ON T2.status_id = T1.status_id", (address_status,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the number of pages of a book by its title
@app.get("/v1/books/num_pages_by_title", operation_id="get_num_pages_by_title", summary="Retrieves the total number of pages for a book, based on the provided book title.")
async def get_num_pages_by_title(title: str = Query(..., description="Title of the book")):
    cursor.execute("SELECT num_pages FROM book WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"num_pages": []}
    return {"num_pages": result[0]}

# Endpoint to get the titles of books by an author's name and publication date
@app.get("/v1/books/titles_by_author_name_and_publication_date", operation_id="get_titles_by_author_name_and_publication_date", summary="Retrieve the titles of books written by a specific author and published on a given date. The operation requires the author's name and the publication date in 'YYYY-MM-DD' format as input parameters. The result is a list of book titles that match the provided criteria.")
async def get_titles_by_author_name_and_publication_date(author_name: str = Query(..., description="Name of the author"), publication_date: str = Query(..., description="Publication date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id WHERE T3.author_name = ? AND T1.publication_date = ?", (author_name, publication_date))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct customer names by address details
@app.get("/v1/books/customer_names_by_address", operation_id="get_customer_names_by_address", summary="Retrieve a list of unique customer names associated with a specific address, identified by street number, street name, and city.")
async def get_customer_names_by_address(street_number: int = Query(..., description="Street number"), street_name: str = Query(..., description="Street name"), city: str = Query(..., description="City")):
    cursor.execute("SELECT DISTINCT T1.first_name, T1.last_name FROM customer AS T1 INNER JOIN customer_address AS T2 ON T1.customer_id = T2.customer_id INNER JOIN address AS T3 ON T3.address_id = T2.address_id WHERE T3.street_number = ? AND T3.street_name = ? AND T3.city = ?", (street_number, street_name, city))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the country name based on address details
@app.get("/v1/books/country_name_by_address", operation_id="get_country_name_by_address", summary="Retrieves the country name associated with a specific address, identified by its street number, street name, and city.")
async def get_country_name_by_address(street_number: int = Query(..., description="Street number"), street_name: str = Query(..., description="Street name"), city: str = Query(..., description="City")):
    cursor.execute("SELECT T2.country_name FROM address AS T1 INNER JOIN country AS T2 ON T2.country_id = T1.country_id WHERE T1.street_number = ? AND T1.street_name = ? AND T1.city = ?", (street_number, street_name, city))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get address details based on country name
@app.get("/v1/books/address_details_by_country", operation_id="get_address_details_by_country", summary="Retrieves the first 10 address details (street number, street name, and city) associated with the specified country. The country is identified by its name.")
async def get_address_details_by_country(country_name: str = Query(..., description="Country name")):
    cursor.execute("SELECT T1.street_number, T1.street_name, T1.city FROM address AS T1 INNER JOIN country AS T2 ON T2.country_id = T1.country_id WHERE T2.country_name = ? LIMIT 10", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"address_details": []}
    return {"address_details": result}

# Endpoint to get shipping method based on customer details and order date
@app.get("/v1/books/shipping_method_by_customer_order", operation_id="get_shipping_method_by_customer_order", summary="Retrieves the shipping method associated with a specific customer order. The customer is identified by their first and last name, and the order is specified by its date. The shipping method details are returned as a result.")
async def get_shipping_method_by_customer_order(first_name: str = Query(..., description="Customer first name"), last_name: str = Query(..., description="Customer last name"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT T3.method_name FROM cust_order AS T1 INNER JOIN customer AS T2 ON T1.customer_id = T2.customer_id INNER JOIN shipping_method AS T3 ON T3.method_id = T1.shipping_method_id WHERE T2.first_name = ? AND T2.last_name = ? AND T1.order_date = ?", (first_name, last_name, order_date))
    result = cursor.fetchone()
    if not result:
        return {"method_name": []}
    return {"method_name": result[0]}

# Endpoint to get language name based on book title
@app.get("/v1/books/language_name_by_book_title", operation_id="get_language_name_by_book_title", summary="Retrieves the language name associated with a specific book title. The operation searches for the book using the provided title and returns the corresponding language name.")
async def get_language_name_by_book_title(title: str = Query(..., description="Book title")):
    cursor.execute("SELECT T2.language_name FROM book AS T1 INNER JOIN book_language AS T2 ON T1.language_id = T2.language_id WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"language_name": []}
    return {"language_name": result[0]}

# Endpoint to get the percentage of books by a specific author published by a specific publisher
@app.get("/v1/books/percentage_books_by_author_publisher", operation_id="get_percentage_books_by_author_publisher", summary="Retrieves the percentage of books written by a specific author and published by a specific publisher. The calculation is based on the total number of books published by the given publisher.")
async def get_percentage_books_by_author_publisher(author_name: str = Query(..., description="Author name"), publisher_name: str = Query(..., description="Publisher name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.author_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM author AS T1 INNER JOIN book_author AS T2 ON T2.author_id = T1.author_id INNER JOIN book AS T3 ON T3.book_id = T2.book_id INNER JOIN publisher AS T4 ON T4.publisher_id = T3.publisher_id WHERE T4.publisher_name = ?", (author_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the book with the highest number of pages
@app.get("/v1/books/book_with_most_pages", operation_id="get_book_with_most_pages", summary="Retrieves the title of the book with the highest number of pages from the available records.")
async def get_book_with_most_pages():
    cursor.execute("SELECT title FROM book ORDER BY num_pages DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"book_title": []}
    return {"book_title": result[0]}

# Endpoint to get the proportion of books in a specific language by publisher
@app.get("/v1/books/language_proportion_by_publisher", operation_id="get_language_proportion_by_publisher", summary="Retrieves the proportion of books in a specified language published by a given publisher. This operation calculates the ratio of books in the requested language to the total number of books published by the specified publisher.")
async def get_language_proportion_by_publisher(language_name: str = Query(..., description="Name of the language"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.language_name = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(*) FROM book_language AS T1 INNER JOIN book AS T2 ON T1.language_id = T2.language_id INNER JOIN publisher AS T3 ON T3.publisher_id = T2.publisher_id WHERE T3.publisher_name = ?", (language_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the book with the most pages from a specific publisher within a date range
@app.get("/v1/books/most_pages_by_publisher_date_range", operation_id="get_most_pages_by_publisher_date_range", summary="Retrieve the title of the book with the highest page count from a specified publisher, within a defined publication year range.")
async def get_most_pages_by_publisher_date_range(publisher_name: str = Query(..., description="Name of the publisher"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ? AND STRFTIME('%Y', T1.publication_date) BETWEEN ? AND ? ORDER BY T1.num_pages DESC LIMIT 1", (publisher_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the price of a book by title and publication year
@app.get("/v1/books/price_by_title_year", operation_id="get_price_by_title_year", summary="Get the price of a book by title and publication year")
async def get_price_by_title_year(title: str = Query(..., description="Title of the book"), publication_year: str = Query(..., description="Publication year in 'YYYY' format")):
    cursor.execute("SELECT T2.price FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T1.title = ? AND STRFTIME('%Y', T1.publication_date) = ?", (title, publication_year))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get customer address by first name
@app.get("/v1/books/customer_address_by_first_name", operation_id="get_customer_address_by_first_name", summary="Retrieves the street number, street name, and city of the address associated with a customer, based on the provided first name. The customer's first name is used to search for a match in the customer table, and the corresponding address details are returned from the address table.")
async def get_customer_address_by_first_name(first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT T3.street_number, T3.street_name, T3.city FROM customer AS T1 INNER JOIN customer_address AS T2 ON T1.customer_id = T2.customer_id INNER JOIN address AS T3 ON T3.address_id = T2.address_id INNER JOIN address_status AS T4 ON T4.status_id = T2.status_id WHERE T1.first_name = ?", (first_name,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [{"street_number": row[0], "street_name": row[1], "city": row[2]} for row in result]}

# Endpoint to get the count of orders by customer name
@app.get("/v1/books/order_count_by_customer_name", operation_id="get_order_count_by_customer_name", summary="Retrieves the total number of orders placed by a customer, identified by their first and last names. The operation calculates the count by joining the order_line, cust_order, and customer tables based on the provided customer name.")
async def get_order_count_by_customer_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(*) FROM order_line AS T1 INNER JOIN cust_order AS T2 ON T2.order_id = T1.order_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get country name by customer email
@app.get("/v1/books/country_by_customer_email", operation_id="get_country_by_customer_email", summary="Retrieves the country name associated with the provided customer email. This operation fetches the country name from the database by joining the customer, customer_address, address, and country tables using the customer's email as the search criterion.")
async def get_country_by_customer_email(email: str = Query(..., description="Email of the customer")):
    cursor.execute("SELECT T4.country_name FROM customer AS T1 INNER JOIN customer_address AS T2 ON T1.customer_id = T2.customer_id INNER JOIN address AS T3 ON T3.address_id = T2.address_id INNER JOIN country AS T4 ON T4.country_id = T3.country_id WHERE T1.email = ?", (email,))
    result = cursor.fetchone()
    if not result:
        return {"country_name": []}
    return {"country_name": result[0]}

# Endpoint to get the customer with the lowest order price
@app.get("/v1/books/customer_with_lowest_order_price", operation_id="get_customer_with_lowest_order_price", summary="Retrieves the first name and last name of the customer who placed the order with the lowest price. This operation joins the order_line, cust_order, and customer tables to find the customer with the lowest order price.")
async def get_customer_with_lowest_order_price():
    cursor.execute("SELECT T3.first_name, T3.last_name FROM order_line AS T1 INNER JOIN cust_order AS T2 ON T2.order_id = T1.order_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id ORDER BY T1.price ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get ISBN13 of books ordered by a specific customer email
@app.get("/v1/books/isbn13_by_customer_email", operation_id="get_isbn13_by_customer_email", summary="Retrieves the ISBN13 of books that a customer with a specific email has ordered. The operation uses the provided email to search for corresponding customer records and then identifies the books associated with those orders.")
async def get_isbn13_by_customer_email(email: str = Query(..., description="Email of the customer")):
    cursor.execute("SELECT T1.isbn13 FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T4.email = ?", (email,))
    result = cursor.fetchall()
    if not result:
        return {"isbn13": []}
    return {"isbn13": [row[0] for row in result]}

# Endpoint to get distinct author names for books with a price greater than a specified value
@app.get("/v1/books/authors_by_price", operation_id="get_authors_by_price", summary="Retrieve a list of unique author names who have written books priced higher than the specified minimum price.")
async def get_authors_by_price(min_price: float = Query(..., description="Minimum price of the book")):
    cursor.execute("SELECT DISTINCT T3.author_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN order_line AS T4 ON T4.book_id = T1.book_id WHERE T4.price > ?", (min_price,))
    result = cursor.fetchall()
    if not result:
        return {"authors": []}
    return {"authors": [row[0] for row in result]}

# Endpoint to get the publisher name for a specific ISBN13
@app.get("/v1/books/publisher_by_isbn13", operation_id="get_publisher_by_isbn13", summary="Retrieves the name of the publisher associated with a specific book, identified by its unique ISBN13. This operation returns the publisher's name, providing valuable information about the book's origin and publication details.")
async def get_publisher_by_isbn13(isbn13: str = Query(..., description="ISBN13 of the book")):
    cursor.execute("SELECT T2.publisher_name FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T1.isbn13 = ?", (isbn13,))
    result = cursor.fetchone()
    if not result:
        return {"publisher": []}
    return {"publisher": result[0]}

# Endpoint to get the count of books from a specific publisher, published in a specific year, and within a specific page range
@app.get("/v1/books/count_books_by_publisher_year_pages", operation_id="get_count_books_by_publisher_year_pages", summary="Retrieves the total number of books from a specified publisher, published in a given year, and within a defined page range. The response includes the count of books that meet the provided criteria.")
async def get_count_books_by_publisher_year_pages(publisher_name: str = Query(..., description="Name of the publisher"), year: str = Query(..., description="Year of publication in 'YYYY' format"), min_pages: int = Query(..., description="Minimum number of pages"), max_pages: int = Query(..., description="Maximum number of pages")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ? AND STRFTIME('%Y', T1.publication_date) = ? AND T1.num_pages BETWEEN ? AND ?", (publisher_name, year, min_pages, max_pages))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the price of a book by ISBN13
@app.get("/v1/books/price_by_isbn13", operation_id="get_price_by_isbn13", summary="Retrieves the current price of a book identified by its unique ISBN13. The price is determined based on the most recent order data.")
async def get_price_by_isbn13(isbn13: str = Query(..., description="ISBN13 of the book")):
    cursor.execute("SELECT T2.price FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T1.isbn13 = ?", (isbn13,))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get the number of pages of a book by order ID
@app.get("/v1/books/num_pages_by_order_id", operation_id="get_num_pages_by_order_id", summary="Retrieves the total number of pages for a book associated with a specific order. The operation requires the order ID as input to identify the book and calculate the total number of pages.")
async def get_num_pages_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.num_pages FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T2.order_id = ?", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"num_pages": []}
    return {"num_pages": result[0]}

# Endpoint to get the count of books published by a specific publisher in a specific year
@app.get("/v1/books/count_books_publisher_year", operation_id="get_count_books_publisher_year", summary="Retrieves the total number of books published by a specific publisher during a particular year. The operation requires the publisher's name and the year of publication as input parameters.")
async def get_count_books_publisher_year(publisher_name: str = Query(..., description="Name of the publisher"), publication_year: str = Query(..., description="Publication year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM book AS T1 INNER JOIN publisher AS T2 ON T1.publisher_id = T2.publisher_id WHERE T2.publisher_name = ? AND STRFTIME('%Y', T1.publication_date) = ?", (publisher_name, publication_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ISBN13 and price of a book based on book ID
@app.get("/v1/books/isbn13_price_by_book_id", operation_id="get_isbn13_price_by_book_id", summary="Retrieves the ISBN13 and price of a specific book, identified by its unique book_id. This operation fetches the requested data from the book and order_line tables, ensuring that the book_id matches in both tables.")
async def get_isbn13_price_by_book_id(book_id: int = Query(..., description="ID of the book")):
    cursor.execute("SELECT T1.isbn13, T2.price FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T2.book_id = ?", (book_id,))
    result = cursor.fetchall()
    if not result:
        return {"isbn13_price": []}
    return {"isbn13_price": result}

# Endpoint to get the titles of books based on order ID
@app.get("/v1/books/titles_by_order_id", operation_id="get_titles_by_order_id", summary="Retrieves the titles of books associated with a specific order. The operation uses the provided order ID to identify the relevant order and returns the corresponding book titles.")
async def get_titles_by_order_id(order_id: int = Query(..., description="ID of the order")):
    cursor.execute("SELECT T1.title FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T2.order_id = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get distinct customer emails based on price range
@app.get("/v1/books/customer_emails_by_price_range", operation_id="get_customer_emails_by_price_range", summary="Retrieves unique customer emails for orders within a specified price range. The operation filters orders based on the provided minimum and maximum price limits, ensuring that only relevant customer emails are returned.")
async def get_customer_emails_by_price_range(min_price: float = Query(..., description="Minimum price"), max_price: float = Query(..., description="Maximum price")):
    cursor.execute("SELECT DISTINCT T3.email FROM order_line AS T1 INNER JOIN cust_order AS T2 ON T2.order_id = T1.order_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T1.price BETWEEN ? AND ?", (min_price, max_price))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": result}

# Endpoint to get the ISBN13 of books based on price
@app.get("/v1/books/isbn13_by_price", operation_id="get_isbn13_by_price", summary="Retrieve the ISBN13 of books that match the specified price. This operation returns the ISBN13 of books whose order line price matches the provided price parameter. The price parameter is used to filter the results.")
async def get_isbn13_by_price(price: float = Query(..., description="Price of the book")):
    cursor.execute("SELECT T1.isbn13 FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T2.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"isbn13": []}
    return {"isbn13": result}

# Endpoint to get the publisher names of books by a specific author
@app.get("/v1/books/publisher_names_by_author", operation_id="get_publisher_names_by_author", summary="Retrieve a list of unique publisher names associated with books written by a specific author. The author's name is provided as an input parameter.")
async def get_publisher_names_by_author(author_name: str = Query(..., description="Name of the author")):
    cursor.execute("SELECT T4.publisher_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id INNER JOIN publisher AS T4 ON T4.publisher_id = T1.publisher_id WHERE T3.author_name = ? GROUP BY T4.publisher_name", (author_name,))
    result = cursor.fetchall()
    if not result:
        return {"publisher_names": []}
    return {"publisher_names": result}

# Endpoint to get the total number of pages of books ordered by a specific customer
@app.get("/v1/books/total_pages_by_customer", operation_id="get_total_pages_by_customer", summary="Retrieves the cumulative number of pages from all books ordered by a customer identified by their first and last names. This operation considers the customer's entire order history to calculate the total.")
async def get_total_pages_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT SUM(T1.num_pages) FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id INNER JOIN cust_order AS T3 ON T3.order_id = T2.order_id INNER JOIN customer AS T4 ON T4.customer_id = T3.customer_id WHERE T4.first_name = ? AND T4.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"total_pages": []}
    return {"total_pages": result[0]}

# Endpoint to get the author name of the most recently published book
@app.get("/v1/books/author_most_recent_book", operation_id="get_author_most_recent_book", summary="Retrieves the name of the author who has most recently published a book. The operation considers all books and their respective authors, sorts them by publication date in descending order, and returns the author of the most recent book.")
async def get_author_most_recent_book():
    cursor.execute("SELECT T3.author_name FROM book AS T1 INNER JOIN book_author AS T2 ON T1.book_id = T2.book_id INNER JOIN author AS T3 ON T3.author_id = T2.author_id ORDER BY T1.publication_date DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"author_name": []}
    return {"author_name": result[0]}

# Endpoint to get the percentage of books in a specific language published by a specific publisher
@app.get("/v1/books/percentage_books_language_publisher", operation_id="get_percentage_books_language_publisher", summary="Retrieves the percentage of books in a given language that were published by a specific publisher. This operation calculates the proportion of books in the specified language published by the given publisher, providing a quantitative measure of the publisher's output in that language.")
async def get_percentage_books_language_publisher(language_name: str = Query(..., description="Name of the language"), publisher_name: str = Query(..., description="Name of the publisher")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.language_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM book_language AS T1 INNER JOIN book AS T2 ON T1.language_id = T2.language_id INNER JOIN publisher AS T3 ON T3.publisher_id = T2.publisher_id WHERE T3.publisher_name = ?", (language_name, publisher_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in the count of books with pages less than a given number and more than a given number, filtered by order line price
@app.get("/v1/books/difference_in_book_counts_by_pages_and_price", operation_id="get_difference_in_book_counts", summary="Retrieves the difference in the number of books with a page count below a specified threshold and above another specified threshold, where the order line price is less than a given value.")
async def get_difference_in_book_counts(num_pages_less: int = Query(..., description="Number of pages less than this value"), num_pages_more: int = Query(..., description="Number of pages more than this value"), price: float = Query(..., description="Price of the order line")):
    cursor.execute("SELECT SUM(CASE WHEN T1.num_pages < ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.num_pages > ? THEN 1 ELSE 0 END) AS dif FROM book AS T1 INNER JOIN order_line AS T2 ON T1.book_id = T2.book_id WHERE T2.price < ?", (num_pages_less, num_pages_more, price))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct language names and book titles where the order line price is less than a percentage of the average price
@app.get("/v1/books/distinct_language_names_and_titles_by_price_percentage", operation_id="get_distinct_language_names_and_titles", summary="Retrieves unique combinations of language names and book titles for books priced below a specified percentage of the average order line price. The percentage is provided as an input parameter.")
async def get_distinct_language_names_and_titles(price_percentage: int = Query(..., description="Percentage of the average price")):
    cursor.execute("SELECT DISTINCT T3.language_name, T2.title FROM order_line AS T1 INNER JOIN book AS T2 ON T1.book_id = T2.book_id INNER JOIN book_language AS T3 ON T3.language_id = T2.language_id WHERE T1.price * 100 < ( SELECT AVG(price) FROM order_line ) * ?", (price_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"languages_and_titles": []}
    return {"languages_and_titles": result}

api_calls = [
    "/v1/books/count_books_by_publisher_and_pages?publisher_id=1929&min_pages=500",
    "/v1/books/publication_date_most_pages",
    "/v1/books/publisher_name_by_title?title=The%20Illuminati",
    "/v1/books/count_books_by_publisher_name?publisher_name=Thomas%20Nelson",
    "/v1/books/publisher_with_most_books",
    "/v1/books/earliest_book_by_publisher?publisher_name=Thomas%20Nelson",
    "/v1/books/count_books_by_publisher_and_min_pages?publisher_name=Thomas%20Nelson&min_pages=300",
    "/v1/books/publisher_of_book_with_most_pages",
    "/v1/books/count_books_by_language?language_name=English",
    "/v1/books/titles_by_language?language_name=British%20English",
    "/v1/books/min_price_by_title?title=The%20Little%20House",
    "/v1/books/titles_by_customer?first_name=Lucas&last_name=Wyldbore",
    "/v1/books/count_by_customer_and_pages?first_name=Lucas&last_name=Wyldbore&num_pages=300",
    "/v1/books/total_price_by_customer?first_name=Lucas&last_name=Wyldbore",
    "/v1/books/average_price_by_customer?first_name=Lucas&last_name=Wyldbore",
    "/v1/books/percentage_above_price_by_customer?first_name=Lucas&last_name=Wyldbore&price_threshold=13",
    "/v1/books/city_by_address_id?address_id=547",
    "/v1/books/order_count_by_customer?first_name=Cordy&last_name=Dumbarton",
    "/v1/books/earliest_book_by_language?language_name=Japanese",
    "/v1/books/top_publisher",
    "/v1/books/count_books_by_publisher?publisher_name=Kensington",
    "/v1/books/get_language_by_book_id?book_id=1405",
    "/v1/books/most_frequent_customer",
    "/v1/books/most_ordered_book",
    "/v1/books/count_books_by_author?author_name=David%20Foster%20Wallace",
    "/v1/books/count_orders_by_title?title=O%20Xar\u00e1",
    "/v1/books/get_country_by_customer_name_status?first_name=Malina&last_name=Johnson&status_id=2",
    "/v1/books/count_addresses_by_country?country_name=Ukraine",
    "/v1/books/get_country_by_city?city=\u017dirovnica",
    "/v1/books/percentage_orders_by_shipping_method?method_name=International&order_date=2022-11-10%",
    "/v1/books/average_pages_by_author?author_name=David%20Coward",
    "/v1/books/cheapest_shipping_method",
    "/v1/books/first_book_by_year?year=1900",
    "/v1/books/customer_details_by_email?email=aalleburtonkc@yellowbook.com",
    "/v1/books/order_count_by_country_year?country_name=Iran&year=2022",
    "/v1/books/order_count_by_customer_shipping_method?first_name=Daisey&last_name=Lamball&method_name=International",
    "/v1/books/order_count_by_status_customer?status_value=Returned&first_name=Antonia&last_name=Poltun",
    "/v1/books/most_used_shipping_method",
    "/v1/books/order_count_by_status_year?status_value=Delivered&year=2021",
    "/v1/books/first_book_by_author?author_name=J.K.%20Rowling",
    "/v1/books/publisher_of_earliest_book_by_author?author_name=Agatha%20Christie",
    "/v1/books/titles_by_author?author_name=Danielle%20Steel",
    "/v1/books/count_books_by_author_and_publisher?author_name=William%20Shakespeare&publisher_name=Penguin%20Classics",
    "/v1/books/total_shipping_cost_by_customer_year?first_name=Page&last_name=Holsey&year=2022",
    "/v1/books/publisher_name_by_id?publisher_id=22",
    "/v1/books/count_books_by_author_and_page_limit?author_name=Al%20Gore&max_pages=400",
    "/v1/books/author_publisher_by_publication_date?publication_date=1997-07-10",
    "/v1/books/language_by_isbn13?isbn13=23755004321",
    "/v1/books/most_expensive_book_title",
    "/v1/books/isbn13_by_language?language_name=Spanish",
    "/v1/books/count_books_by_publisher_price?publisher_name=Berkley&max_price=1",
    "/v1/books/author_of_longest_book",
    "/v1/books/customer_emails_by_book_title?title=Switch%20on%20the%20Night",
    "/v1/books/authors_by_publisher?publisher_name=Abrams",
    "/v1/books/percentage_books_by_year_author?year=1992&author_name=Abraham%20Lincoln",
    "/v1/books/titles_publishers_by_year_pages?year=2004",
    "/v1/books/customer_emails_by_name?first_name=Moss&last_name=Zarb",
    "/v1/books/street_names_by_city?city=Dallas",
    "/v1/books/titles_by_author_and_year?author_name=Orson%20Scott%20Card&publication_year=2001",
    "/v1/books/count_by_author?author_name=Orson%20Scott%20Card",
    "/v1/books/author_titles_by_min_pages?min_pages=3000",
    "/v1/books/authors_by_title?title=The%20Prophet",
    "/v1/books/publishers_by_author?author_name=Barry%20Eisler",
    "/v1/books/count_by_language?language_name=Japanese",
    "/v1/books/total_price_by_title?title=The%20Prophet",
    "/v1/books/count_orders_by_customer_and_year?first_name=Daisey&last_name=Lamball&order_year=2021",
    "/v1/books/customer_names_by_book_title?title=Fantasmas",
    "/v1/books/percentage_orders_by_shipping_method_year?method_name=International&year=2020",
    "/v1/books/author_names_by_prefix?prefix=George",
    "/v1/books/year_with_most_orders",
    "/v1/books/average_order_line_price",
    "/v1/books/book_titles_by_publication_year?year=1995",
    "/v1/books/most_common_email_domain",
    "/v1/books/publisher_count_by_name_substring?substring=book",
    "/v1/books/least_common_book_language",
    "/v1/books/order_dates_by_customer?first_name=Adrian&last_name=Kunzelmann",
    "/v1/books/most_prolific_author",
    "/v1/books/book_titles_by_publisher?publisher_name=Harper%20Collins",
    "/v1/books/least_used_shipping_method",
    "/v1/books/count_addresses_by_status?address_status=Inactive",
    "/v1/books/most_ordered_book_title",
    "/v1/books/most_common_destination_address",
    "/v1/books/order_status_date_difference?order_id=2398",
    "/v1/books/customers/most_frequent_customer_names?limit=1",
    "/v1/books/orders/shipping_method_percentage?method_name=International",
    "/v1/books/authors_below_avg_pages",
    "/v1/books/order_lines/min_price_excluding?excluded_price=0",
    "/v1/books/addresses/count_by_city?city=Villeneuve-la-Garenne",
    "/v1/books/authors/count_by_name_prefix?name_prefix=Adam",
    "/v1/books/customers/count_by_email_suffix?email_suffix=@yahoo.com",
    "/v1/books/countries/distinct_cities?country_name=United%20States%20of%20America",
    "/v1/books/customers/order_count_by_customer_shipping_year?first_name=Marcelia&last_name=Goering&order_year=2021&method_name=Priority",
    "/v1/books/max_price_by_title?title=Bite%20Me%20If%20You%20Can%20(Argeneau%20%236)",
    "/v1/books/count_books_by_publication_date",
    "/v1/books/titles_by_language_and_publisher?language_name=Spanish&publisher_name=Alfaguara",
    "/v1/books/count_publication_dates_by_author?author_name=Stephen%20King",
    "/v1/books/language_names_by_publication_date",
    "/v1/books/distinct_publishers_by_title?title=The%20Secret%20Garden",
    "/v1/books/count_books_by_publisher_and_author?publisher_name=Scholastic&author_name=J.K.%20Rowling",
    "/v1/books/publishers_with_min_books?min_books=30",
    "/v1/books/addresses/street_numbers",
    "/v1/books/address_details_by_city?city=Lazaro%20Cardenas",
    "/v1/books/isbn13_by_page_range?max_pages=140&min_pages=135",
    "/v1/books/book_titles_by_publication_date?limit=6",
    "/v1/books/order_count_by_date_pattern?order_date_pattern=2020-12%25",
    "/v1/books/customer_names_by_last_name?last_name_pattern=K%25",
    "/v1/books/cities_by_country_name?country_name=Costa%20Rica",
    "/v1/books/street_names_by_address_status?address_status=Inactive",
    "/v1/books/customer_names_by_city?city=Baiyin",
    "/v1/books/customer_emails_by_shipping_method?method_name=Priority",
    "/v1/books/order_dates_by_street_number?street_number=460",
    "/v1/books/order_ids_by_status?status_value=Cancelled",
    "/v1/books/distinct_status_values_by_order_date?order_date=2022-04-10%",
    "/v1/books/order_dates_by_price?price=16.54",
    "/v1/books/percentage_international_shipping_by_first_name?method_name=International&first_name=Kaleena",
    "/v1/books/customer_address?first_name=Ursola&last_name=Purdy",
    "/v1/books/earliest_publisher",
    "/v1/books/shipping_cost_difference",
    "/v1/books/count_books_by_year?year=2017",
    "/v1/books/isbn13_by_title?title=The%20Mystery%20in%20the%20Rocky%20Mountains",
    "/v1/books/order_status_percentage?status_value=Returned&year=2022",
    "/v1/books/address_status_percentage?address_status=Inactive",
    "/v1/books/num_pages_by_title?title=Seaward",
    "/v1/books/titles_by_author_name_and_publication_date?author_name=Hirohiko%20Araki&publication_date=2006-06-06",
    "/v1/books/customer_names_by_address?street_number=55&street_name=Dorton%20Pass&city=Huangqiao",
    "/v1/books/country_name_by_address?street_number=9&street_name=Green%20Ridge%20Point&city=Arendal",
    "/v1/books/address_details_by_country?country_name=Poland",
    "/v1/books/shipping_method_by_customer_order?first_name=Nicolette&last_name=Sadler&order_date=2020-06-29%2019:40:07",
    "/v1/books/language_name_by_book_title?title=El%20plan%20infinito",
    "/v1/books/percentage_books_by_author_publisher?author_name=Hirohiko%20Araki&publisher_name=VIZ%20Media",
    "/v1/books/book_with_most_pages",
    "/v1/books/language_proportion_by_publisher?language_name=English&publisher_name=Carole%20Marsh%20Mysteries",
    "/v1/books/most_pages_by_publisher_date_range?publisher_name=Free%20Press&start_year=1990&end_year=2000",
    "/v1/books/price_by_title_year?title=The%20Servant%20Leader&publication_year=2003",
    "/v1/books/customer_address_by_first_name?first_name=Kandy",
    "/v1/books/order_count_by_customer_name?first_name=Kandy&last_name=Adamec",
    "/v1/books/country_by_customer_email?email=rturbitT2@geocities.jp",
    "/v1/books/customer_with_lowest_order_price",
    "/v1/books/isbn13_by_customer_email?email=fsier3e@ihg.com",
    "/v1/books/authors_by_price?min_price=19",
    "/v1/books/publisher_by_isbn13?isbn13=76092025986",
    "/v1/books/count_books_by_publisher_year_pages?publisher_name=Birlinn&year=2008&min_pages=600&max_pages=700",
    "/v1/books/price_by_isbn13?isbn13=9780763628321",
    "/v1/books/num_pages_by_order_id?order_id=1167",
    "/v1/books/count_books_publisher_year?publisher_name=Brava&publication_year=2006",
    "/v1/books/isbn13_price_by_book_id?book_id=6503",
    "/v1/books/titles_by_order_id?order_id=931",
    "/v1/books/customer_emails_by_price_range?min_price=3&max_price=5",
    "/v1/books/isbn13_by_price?price=7.5",
    "/v1/books/publisher_names_by_author?author_name=Alan%20Lee",
    "/v1/books/total_pages_by_customer?first_name=Mick&last_name=Sever",
    "/v1/books/author_most_recent_book",
    "/v1/books/percentage_books_language_publisher?language_name=English&publisher_name=Ace%20Book",
    "/v1/books/difference_in_book_counts_by_pages_and_price?num_pages_less=500&num_pages_more=500&price=1",
    "/v1/books/distinct_language_names_and_titles_by_price_percentage?price_percentage=20"
]
