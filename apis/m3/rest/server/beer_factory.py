from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/beer_factory/beer_factory.sqlite')
cursor = conn.cursor()

# Endpoint to get the brand name of the first brewed root beer
@app.get("/v1/beer_factory/first_brewed_brand", operation_id="get_first_brewed_brand", summary="Retrieves the brand name of the root beer that was first brewed. This operation identifies the earliest year of brewing from the available brands and returns the corresponding brand name.")
async def get_first_brewed_brand():
    cursor.execute("SELECT BrandName FROM rootbeerbrand WHERE FirstBrewedYear = ( SELECT MIN(FirstBrewedYear) FROM rootbeerbrand )")
    result = cursor.fetchone()
    if not result:
        return {"brand_name": []}
    return {"brand_name": result[0]}

# Endpoint to get the count of root beer brands from a specific country
@app.get("/v1/beer_factory/count_brands_by_country", operation_id="get_count_brands_by_country", summary="Retrieves the total number of distinct root beer brands originating from a specified country.")
async def get_count_brands_by_country(country: str = Query(..., description="Country of the root beer brand")):
    cursor.execute("SELECT COUNT(BrandID) FROM rootbeerbrand WHERE Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the brand names of root beer brands with a Facebook page
@app.get("/v1/beer_factory/brands_with_facebook", operation_id="get_brands_with_facebook", summary="Retrieves the names of root beer brands that have a Facebook page. This operation returns a list of brand names for root beer brands that have a presence on Facebook. The data is sourced from the rootbeerbrand table.")
async def get_brands_with_facebook():
    cursor.execute("SELECT BrandName FROM rootbeerbrand WHERE FacebookPage IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": [row[0] for row in result]}

# Endpoint to get the brand name with the highest profit margin
@app.get("/v1/beer_factory/brand_with_highest_profit_margin", operation_id="get_brand_with_highest_profit_margin", summary="Retrieves the name of the beer brand with the highest profit margin, calculated as the difference between the current retail price and the wholesale cost.")
async def get_brand_with_highest_profit_margin():
    cursor.execute("SELECT BrandName FROM rootbeerbrand ORDER BY CurrentRetailPrice - WholesaleCost LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"brand_name": []}
    return {"brand_name": result[0]}

# Endpoint to get the description of a specific root beer brand
@app.get("/v1/beer_factory/brand_description", operation_id="get_brand_description", summary="Retrieves the description of a specific root beer brand by its name. The brand name is provided as an input parameter, which is used to locate the corresponding description in the database.")
async def get_brand_description(brand_name: str = Query(..., description="Brand name of the root beer")):
    cursor.execute("SELECT Description FROM rootbeerbrand WHERE BrandName = ?", (brand_name,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the city of a specific brewery
@app.get("/v1/beer_factory/brewery_city", operation_id="get_brewery_city", summary="Retrieves the city where a specific brewery is located. The operation requires the brewery's name as input and returns the corresponding city.")
async def get_brewery_city(brewery_name: str = Query(..., description="Name of the brewery")):
    cursor.execute("SELECT City FROM rootbeerbrand WHERE BreweryName = ?", (brewery_name,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get the count of transactions for a specific customer in a given month
@app.get("/v1/beer_factory/count_transactions_by_customer_month", operation_id="get_count_transactions_by_customer_month", summary="Retrieves the total number of transactions made by a specific customer during a given month. The operation requires the customer's first and last name, as well as the month of the transactions in 'YYYY-MM' format.")
async def get_count_transactions_by_customer_month(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), transaction_month: str = Query(..., description="Transaction month in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.First = ? AND T1.Last = ? AND STRFTIME('%Y-%m', T2.TransactionDate) = ?", (first_name, last_name, transaction_month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of transactions for customers of a specific gender in a given month
@app.get("/v1/beer_factory/count_transactions_by_gender_month", operation_id="get_count_transactions_by_gender_month", summary="Retrieves the total number of transactions made by customers of a specific gender during a given month. The gender and month are provided as input parameters.")
async def get_count_transactions_by_gender_month(gender: str = Query(..., description="Gender of the customer"), transaction_month: str = Query(..., description="Transaction month in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Gender = ? AND STRFTIME('%Y-%m', T2.TransactionDate) = ?", (gender, transaction_month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of transactions for subscribed customers using a specific credit card type in a given month
@app.get("/v1/beer_factory/count_transactions_by_subscription_credit_card_month", operation_id="get_count_transactions_by_subscription_credit_card_month", summary="Retrieves the total number of transactions made by subscribed customers using a specific credit card type during a particular month. The input parameters include the subscription status to the email list, the type of credit card, and the transaction month in 'YYYY-MM' format.")
async def get_count_transactions_by_subscription_credit_card_month(subscribed_to_email_list: str = Query(..., description="Subscription status to email list"), credit_card_type: str = Query(..., description="Credit card type"), transaction_month: str = Query(..., description="Transaction month in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.SubscribedToEmaillist = ? AND T2.CreditCardType = ? AND STRFTIME('%Y-%m', T2.TransactionDate) = ?", (subscribed_to_email_list, credit_card_type, transaction_month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer with the most transactions in a given month
@app.get("/v1/beer_factory/customer_with_most_transactions_month", operation_id="get_customer_with_most_transactions_month", summary="Retrieves the customer who has made the most transactions in a specified month. The month is provided in 'YYYY-MM' format. The operation returns the first and last name of the customer with the highest number of transactions for the given month.")
async def get_customer_with_most_transactions_month(transaction_month: str = Query(..., description="Transaction month in 'YYYY-MM' format")):
    cursor.execute("SELECT T1.First, T1.Last FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE STRFTIME('%Y-%m', T2.TransactionDate) = ? GROUP BY T1.CustomerID ORDER BY COUNT(T2.CustomerID) DESC LIMIT 1", (transaction_month,))
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get distinct brand names based on customer first name, last name, and transaction date
@app.get("/v1/beer_factory/distinct_brand_names", operation_id="get_distinct_brand_names", summary="Retrieve a unique list of beer brand names associated with a specific customer, based on their first and last name and a given transaction date.")
async def get_distinct_brand_names(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), transaction_date: str = Query(..., description="Transaction date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T4.BrandName FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeer AS T3 ON T2.RootBeerID = T3.RootBeerID INNER JOIN rootbeerbrand AS T4 ON T3.BrandID = T4.BrandID WHERE T1.First = ? AND T1.Last = ? AND T2.TransactionDate = ?", (first, last, transaction_date))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": [row[0] for row in result]}

# Endpoint to get the count of customers based on first name, last name, transaction date, and container type
@app.get("/v1/beer_factory/customer_count_by_container_type", operation_id="get_customer_count_by_container_type", summary="Retrieve the total number of customers who have purchased a specific root beer container type on a given date, filtered by first and last name.")
async def get_customer_count_by_container_type(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), transaction_date: str = Query(..., description="Transaction date in 'YYYY-MM-DD' format"), container_type: str = Query(..., description="Container type")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeer AS T3 ON T2.RootBeerID = T3.RootBeerID WHERE T1.First = ? AND T1.Last = ? AND T2.TransactionDate = ? AND T3.ContainerType = ?", (first, last, transaction_date, container_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of brand IDs based on transaction date pattern and brand name
@app.get("/v1/beer_factory/brand_id_count_by_date_pattern", operation_id="get_brand_id_count_by_date_pattern", summary="Retrieves the total count of a specific brand's transactions, filtered by a given date pattern. The date pattern should be provided in 'YYYY-MM%' format. The brand is identified by its name.")
async def get_brand_id_count_by_date_pattern(date_pattern: str = Query(..., description="Transaction date pattern in 'YYYY-MM%' format"), brand_name: str = Query(..., description="Brand name")):
    cursor.execute("SELECT COUNT(T1.BrandID) FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T2.TransactionDate LIKE ? AND T3.BrandName = ?", (date_pattern, brand_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names based on brewery name
@app.get("/v1/beer_factory/customer_names_by_brewery", operation_id="get_customer_names_by_brewery", summary="Retrieves the first and last names of customers who have purchased root beers from a specific brewery. The brewery is identified by its name.")
async def get_customer_names_by_brewery(brewery_name: str = Query(..., description="Brewery name")):
    cursor.execute("SELECT T1.First, T1.Last FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeer AS T3 ON T2.RootBeerID = T3.RootBeerID INNER JOIN rootbeerbrand AS T4 ON T3.BrandID = T4.BrandID WHERE T4.BreweryName = ?", (brewery_name,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the count of brand IDs based on transaction date pattern and null Twitter
@app.get("/v1/beer_factory/brand_id_count_by_date_pattern_null_twitter", operation_id="get_brand_id_count_by_date_pattern_null_twitter", summary="Retrieves the total count of unique brand IDs that have no associated Twitter account and whose transactions occurred within a specified year and month. The date pattern should be provided in the 'YYYY-MM%' format.")
async def get_brand_id_count_by_date_pattern_null_twitter(date_pattern: str = Query(..., description="Transaction date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT COUNT(T1.BrandID) FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T2.TransactionDate LIKE ? AND T3.Twitter IS NULL", (date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct credit card numbers based on customer first name, last name, and transaction date
@app.get("/v1/beer_factory/distinct_credit_card_numbers", operation_id="get_distinct_credit_card_numbers", summary="Retrieve unique credit card numbers associated with a specific customer, identified by their first and last name, for a given transaction date.")
async def get_distinct_credit_card_numbers(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), transaction_date: str = Query(..., description="Transaction date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.CreditCardNumber FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.First = ? AND T1.Last = ? AND T2.TransactionDate = ?", (first, last, transaction_date))
    result = cursor.fetchall()
    if not result:
        return {"credit_card_numbers": []}
    return {"credit_card_numbers": [row[0] for row in result]}

# Endpoint to get the count of customers based on first name, last name, artificial sweetener, and honey
@app.get("/v1/beer_factory/customer_count_by_sweetener_honey", operation_id="get_customer_count_by_sweetener_honey", summary="Retrieves the count of customers who have purchased root beer with specific sweetener and honey attributes. The count is filtered by the customer's first and last name.")
async def get_customer_count_by_sweetener_honey(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), artificial_sweetener: str = Query(..., description="Artificial sweetener (TRUE or FALSE)"), honey: str = Query(..., description="Honey (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeer AS T3 ON T2.RootBeerID = T3.RootBeerID INNER JOIN rootbeerbrand AS T4 ON T3.BrandID = T4.BrandID WHERE T1.First = ? AND T1.Last = ? AND T4.ArtificialSweetener = ? AND T4.Honey = ?", (first, last, artificial_sweetener, honey))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get transaction dates based on gender and minimum customer count
@app.get("/v1/beer_factory/transaction_dates_by_gender_min_count", operation_id="get_transaction_dates_by_gender_min_count", summary="Retrieves the transaction dates for a specific gender with a customer count exceeding the provided minimum threshold.")
async def get_transaction_dates_by_gender_min_count(gender: str = Query(..., description="Gender of the customer"), min_count: int = Query(..., description="Minimum customer count")):
    cursor.execute("SELECT T2.TransactionDate FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Gender = ? GROUP BY T2.TransactionDate HAVING COUNT(T2.CustomerID) > ?", (gender, min_count))
    result = cursor.fetchall()
    if not result:
        return {"transaction_dates": []}
    return {"transaction_dates": [row[0] for row in result]}

# Endpoint to get the average count of brand IDs based on transaction date pattern and brand name
@app.get("/v1/beer_factory/average_brand_id_count_by_date_pattern", operation_id="get_average_brand_id_count_by_date_pattern", summary="Retrieves the average daily count of a specific brand's transactions over a given month. The calculation is based on the provided month's date pattern and brand name.")
async def get_average_brand_id_count_by_date_pattern(date_pattern: str = Query(..., description="Transaction date pattern in 'YYYY-MM%' format"), brand_name: str = Query(..., description="Brand name")):
    cursor.execute("SELECT CAST(COUNT(T1.BrandID) AS REAL) / 31 FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T2.TransactionDate LIKE ? AND T3.BrandName = ?", (date_pattern, brand_name))
    result = cursor.fetchone()
    if not result:
        return {"average_count": []}
    return {"average_count": result[0]}

# Endpoint to get the percentage of brand IDs based on brewery name and transaction date pattern
@app.get("/v1/beer_factory/percentage_brand_id_by_brewery_date_pattern", operation_id="get_percentage_brand_id_by_brewery_date_pattern", summary="Get the percentage of brand IDs based on brewery name and transaction date pattern")
async def get_percentage_brand_id_by_brewery_date_pattern(brewery_name: str = Query(..., description="Brewery name"), date_pattern: str = Query(..., description="Transaction date pattern in 'YYYY%' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T3.BreweryName = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.BrandID) FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T2.TransactionDate LIKE ?", (brewery_name, date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of customer reviews based on first and last name
@app.get("/v1/beer_factory/customer_review_count", operation_id="get_customer_review_count", summary="Retrieves the total number of reviews submitted by a customer identified by their first and last name. The count is determined by matching the provided names with the corresponding customer records and tallying the associated reviews.")
async def get_customer_review_count(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T2.CustomerID) FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.First = ? AND T1.Last = ?", (first, last))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get latitude and longitude based on location name
@app.get("/v1/beer_factory/location_coordinates", operation_id="get_location_coordinates", summary="Retrieves the geographical coordinates (latitude and longitude) for a specified location. The location is identified by its name, which is provided as an input parameter.")
async def get_location_coordinates(location_name: str = Query(..., description="Name of the location")):
    cursor.execute("SELECT T2.Latitude, T2.Longitude FROM location AS T1 INNER JOIN geolocation AS T2 ON T1.LocationID = T2.LocationID WHERE T1.LocationName = ?", (location_name,))
    result = cursor.fetchone()
    if not result:
        return {"coordinates": []}
    return {"coordinates": {"latitude": result[0], "longitude": result[1]}}

# Endpoint to get location name based on transaction ID
@app.get("/v1/beer_factory/location_name_by_transaction_id", operation_id="get_location_name_by_transaction_id", summary="Retrieves the name of the location associated with a specific transaction. The transaction is identified by its unique Transaction ID. The operation returns the location name, providing context about the transaction's origin or destination.")
async def get_location_name_by_transaction_id(transaction_id: int = Query(..., description="Transaction ID")):
    cursor.execute("SELECT T2.LocationName FROM `transaction` AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.TransactionID = ?", (transaction_id,))
    result = cursor.fetchone()
    if not result:
        return {"location_name": []}
    return {"location_name": result[0]}

# Endpoint to get city based on transaction ID
@app.get("/v1/beer_factory/city_by_transaction_id", operation_id="get_city_by_transaction_id", summary="Retrieves the city associated with a specific transaction. The operation uses the provided transaction ID to look up the corresponding city in the customers table, based on the transaction's customer ID.")
async def get_city_by_transaction_id(transaction_id: int = Query(..., description="Transaction ID")):
    cursor.execute("SELECT T1.City FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.TransactionID = ?", (transaction_id,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": result[0]}

# Endpoint to get distinct phone numbers based on credit card number
@app.get("/v1/beer_factory/distinct_phone_numbers", operation_id="get_distinct_phone_numbers", summary="Retrieves a list of unique phone numbers associated with a specific credit card number. This operation identifies distinct phone numbers linked to a given credit card, providing a concise overview of the card's usage across different customers.")
async def get_distinct_phone_numbers(credit_card_number: int = Query(..., description="Credit card number")):
    cursor.execute("SELECT DISTINCT T1.PhoneNumber FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.CreditCardNumber = ?", (credit_card_number,))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": [row[0] for row in result]}

# Endpoint to get the customer with the most reviews
@app.get("/v1/beer_factory/top_reviewer", operation_id="get_top_reviewer", summary="Retrieves the customer who has submitted the highest number of reviews. The response includes the first and last names of the top reviewer.")
async def get_top_reviewer():
    cursor.execute("SELECT T1.First, T1.Last FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID GROUP BY T1.CustomerID ORDER BY COUNT(T2.CustomerID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"top_reviewer": []}
    return {"top_reviewer": {"first": result[0], "last": result[1]}}

# Endpoint to get the first purchase date based on review text
@app.get("/v1/beer_factory/first_purchase_date_by_review", operation_id="get_first_purchase_date_by_review", summary="Retrieves the earliest purchase date of a customer who has provided a specific review text. The operation uses the review text to identify the relevant customer and returns the date of their first purchase.")
async def get_first_purchase_date_by_review(review: str = Query(..., description="Review text")):
    cursor.execute("SELECT T1.FirstPurchaseDate FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Review = ?", (review,))
    result = cursor.fetchone()
    if not result:
        return {"first_purchase_date": []}
    return {"first_purchase_date": result[0]}

# Endpoint to get the earliest transaction date based on first and last name
@app.get("/v1/beer_factory/earliest_transaction_date", operation_id="get_earliest_transaction_date", summary="Retrieves the date of the earliest transaction associated with a specific customer, identified by their first and last name.")
async def get_earliest_transaction_date(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T2.TransactionDate FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.First = ? AND T1.Last = ? ORDER BY T2.TransactionDate LIMIT 1", (first, last))
    result = cursor.fetchone()
    if not result:
        return {"earliest_transaction_date": []}
    return {"earliest_transaction_date": result[0]}

# Endpoint to get the brewery with the most 5-star ratings
@app.get("/v1/beer_factory/top_brewery_by_star_rating", operation_id="get_top_brewery_by_star_rating", summary="Retrieves the brewery with the highest count of 5-star ratings. The operation filters breweries based on a specified star rating and returns the one with the most occurrences of that rating. The input parameter determines the star rating to consider for the ranking.")
async def get_top_brewery_by_star_rating(star_rating: int = Query(..., description="Star rating (must be 5)")):
    cursor.execute("SELECT T1.BreweryName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.StarRating = ? GROUP BY T1.BrandID ORDER BY COUNT(T2.StarRating) DESC LIMIT 1", (star_rating,))
    result = cursor.fetchone()
    if not result:
        return {"top_brewery": []}
    return {"top_brewery": result[0]}

# Endpoint to get subscription status of customers who reviewed a specific brand with a specific rating on a specific date
@app.get("/v1/beer_factory/subscription_status_review", operation_id="get_subscription_status_review", summary="Retrieve the email subscription status of customers who left a review with a specified star rating for a particular root beer brand on a given date. The response will indicate whether the customer is subscribed to the email list or not.")
async def get_subscription_status_review(star_rating: int = Query(..., description="Star rating of the review"), brand_name: str = Query(..., description="Brand name of the root beer"), review_date: str = Query(..., description="Review date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CASE WHEN T1.SubscribedToEmaillist LIKE 'TRUE' THEN 'YES' ELSE 'NO' END AS result FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeerbrand AS T3 ON T2.BrandID = T3.BrandID WHERE T2.StarRating = ? AND T3.BrandName = ? AND T2.ReviewDate = ?", (star_rating, brand_name, review_date))
    result = cursor.fetchall()
    if not result:
        return {"subscription_status": []}
    return {"subscription_status": [row[0] for row in result]}

# Endpoint to get the price difference between retail and wholesale cost for a specific review
@app.get("/v1/beer_factory/price_difference_review", operation_id="get_price_difference_review", summary="Retrieves the price difference between the retail and wholesale cost for a specific review. The review text is used to identify the relevant brand and calculate the price difference.")
async def get_price_difference_review(review: str = Query(..., description="Review text")):
    cursor.execute("SELECT T1.CurrentRetailPrice - T1.WholesaleCost AS price FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.Review = ?", (review,))
    result = cursor.fetchall()
    if not result:
        return {"price_difference": []}
    return {"price_difference": [row[0] for row in result]}

# Endpoint to get the percentage of 5-star ratings for a specific brand
@app.get("/v1/beer_factory/percentage_five_star_ratings", operation_id="get_percentage_five_star_ratings", summary="Retrieves the percentage of 5-star ratings for a specific root beer brand. The operation calculates this percentage by comparing the count of 5-star ratings to the total number of ratings for the given brand.")
async def get_percentage_five_star_ratings(brand_name: str = Query(..., description="Brand name of the root beer")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.StarRating = 5 THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.StarRating) FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T1.BrandName = ?", (brand_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average number of reviews per brand in a specific state
@app.get("/v1/beer_factory/average_reviews_per_brand", operation_id="get_average_reviews_per_brand", summary="Retrieves the average number of reviews per beer brand in a specified state. This operation calculates the ratio of total reviews to the number of unique brands in the given state, providing a measure of review frequency per brand.")
async def get_average_reviews_per_brand(state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT CAST(COUNT(*) AS REAL) / COUNT(DISTINCT T1.BrandID) AS avgreview FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T1.State = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"average_reviews": []}
    return {"average_reviews": result[0]}

# Endpoint to get the count of customers based on gender and email subscription status
@app.get("/v1/beer_factory/customer_count_gender_subscription", operation_id="get_customer_count_gender_subscription", summary="Retrieves the total number of customers categorized by gender and email subscription status. The response provides a count of customers who have opted for email subscriptions and those who have not, differentiated by their gender.")
async def get_customer_count_gender_subscription(gender: str = Query(..., description="Gender of the customer"), subscribed_to_email_list: str = Query(..., description="Email subscription status")):
    cursor.execute("SELECT COUNT(CustomerID) FROM customers WHERE Gender = ? AND SubscribedToEmaillist = ?", (gender, subscribed_to_email_list))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the brand name of the most recently brewed root beer
@app.get("/v1/beer_factory/most_recent_root_beer", operation_id="get_most_recent_root_beer", summary="Retrieves the brand name of the most recently brewed root beer. The operation returns the brand name of the root beer that was brewed most recently, based on the year of its first brewing.")
async def get_most_recent_root_beer():
    cursor.execute("SELECT BrandName FROM rootbeerbrand ORDER BY FirstBrewedYear DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"brand_name": []}
    return {"brand_name": result[0]}

# Endpoint to get the first and last names of the first 10 customers based on their first purchase date
@app.get("/v1/beer_factory/first_ten_customers", operation_id="get_first_ten_customers", summary="Retrieves the first and last names of the first ten customers, sorted by their initial purchase date.")
async def get_first_ten_customers():
    cursor.execute("SELECT First, Last FROM customers ORDER BY FirstPurchaseDate LIMIT 10")
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the count of breweries in a specific country
@app.get("/v1/beer_factory/brewery_count_country", operation_id="get_brewery_count_country", summary="Retrieves the total number of breweries located in a specified country. The operation requires the country name as an input parameter to filter the brewery count accordingly.")
async def get_brewery_count_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT COUNT(BreweryName) FROM rootbeerbrand WHERE Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers with a specific first name and city
@app.get("/v1/beer_factory/customer_count_name_city", operation_id="get_customer_count_name_city", summary="Retrieves the total number of customers with a given first name and city. The response is based on the provided first name and city parameters, which are used to filter the customer records.")
async def get_customer_count_name_city(first_name: str = Query(..., description="First name of the customer"), city: str = Query(..., description="City of the customer")):
    cursor.execute("SELECT COUNT(CustomerID) FROM customers WHERE First = ? AND City = ?", (first_name, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of transactions with a specific credit card type and transaction date pattern
@app.get("/v1/beer_factory/transaction_count_credit_card_date", operation_id="get_transaction_count_credit_card_date", summary="Retrieves the total number of transactions made with a specific credit card type and matching a given transaction date pattern. The date pattern should be provided in 'YYYY%' format.")
async def get_transaction_count_credit_card_date(credit_card_type: str = Query(..., description="Credit card type"), transaction_date_pattern: str = Query(..., description="Transaction date pattern in 'YYYY%' format")):
    cursor.execute("SELECT COUNT(TransactionID) FROM `transaction` WHERE CreditCardType = ? AND TransactionDate LIKE ?", (credit_card_type, transaction_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the brand names reviewed by a specific customer with a specific star rating
@app.get("/v1/beer_factory/brand_names_by_customer_rating", operation_id="get_brand_names_by_customer_rating", summary="Retrieve the brand names of beers that a specific customer has reviewed and given a particular star rating. The operation requires the customer's first and last name, as well as the desired star rating to filter the results.")
async def get_brand_names_by_customer_rating(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), star_rating: int = Query(..., description="Star rating given by the customer")):
    cursor.execute("SELECT T3.BrandName FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeerbrand AS T3 ON T2.BrandID = T3.BrandID WHERE T1.First = ? AND T1.Last = ? AND T2.StarRating = ?", (first, last, star_rating))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": [row[0] for row in result]}

# Endpoint to get the count of root beer brands purchased at a specific location with specific attributes
@app.get("/v1/beer_factory/count_brands_by_location_attributes", operation_id="get_count_brands_by_location_attributes", summary="Retrieve the number of distinct root beer brands that meet specific criteria at a given location. The criteria include the purchase date, the presence of honey and cane sugar in the brand, and the type of container. The location is identified by its name.")
async def get_count_brands_by_location_attributes(location_name: str = Query(..., description="Name of the location"), purchase_date_pattern: str = Query(..., description="Pattern for the purchase date (e.g., '2015%')"), honey: str = Query(..., description="Whether the brand contains honey ('TRUE' or 'FALSE')"), cane_sugar: str = Query(..., description="Whether the brand contains cane sugar ('TRUE' or 'FALSE')"), container_type: str = Query(..., description="Type of container (e.g., 'Bottle')")):
    cursor.execute("SELECT COUNT(T1.BrandID) FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID INNER JOIN location AS T3 ON T1.LocationID = T3.LocationID WHERE T3.LocationName = ? AND T1.PurchaseDate LIKE ? AND T2.Honey = ? AND T2.CaneSugar = ? AND T1.ContainerType = ?", (location_name, purchase_date_pattern, honey, cane_sugar, container_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top brewery name based on purchase dates
@app.get("/v1/beer_factory/top_brewery_by_purchase_dates", operation_id="get_top_brewery_by_purchase_dates", summary="Retrieves the name of the brewery with the most purchases within a specified date range. The date range is defined by the start_date and end_date parameters, both in 'YYYY-MM-DD' format. The result is determined by counting the number of purchases for each brewery within the given date range and selecting the brewery with the highest count.")
async def get_top_brewery_by_purchase_dates(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.BreweryName FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID WHERE T1.PurchaseDate BETWEEN ? AND ? GROUP BY T2.BrandID ORDER BY COUNT(T1.BrandID) DESC LIMIT 1", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"brewery_name": []}
    return {"brewery_name": result[0]}

# Endpoint to get customer names who reviewed a specific brand with a specific star rating
@app.get("/v1/beer_factory/customer_names_by_brand_rating", operation_id="get_customer_names_by_brand_rating", summary="Retrieves the first and last names of customers who have reviewed a specific beer brand with a given star rating. The operation filters customers based on the provided brand name and star rating, and returns their names.")
async def get_customer_names_by_brand_rating(brand_name: str = Query(..., description="Name of the brand"), star_rating: int = Query(..., description="Star rating given by the customer")):
    cursor.execute("SELECT T1.First, T1.Last FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeerbrand AS T3 ON T2.BrandID = T3.BrandID WHERE T3.BrandName = ? AND T2.StarRating = ?", (brand_name, star_rating))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the count of root beer transactions for a specific customer within a date range
@app.get("/v1/beer_factory/count_transactions_by_customer_date_range", operation_id="get_count_transactions_by_customer_date_range", summary="Retrieve the total number of root beer transactions made by a specific customer within a given date range. The operation requires the customer's first and last names, as well as the start and end dates of the desired range, to accurately filter and count the transactions.")
async def get_count_transactions_by_customer_date_range(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.RootBeerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.First = ? AND T1.Last = ? AND T2.TransactionDate BETWEEN ? AND ?", (first, last, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get brand names with a specific star rating
@app.get("/v1/beer_factory/brand_names_by_rating", operation_id="get_brand_names_by_rating", summary="Retrieves the names of beer brands that have been rated with a specific star rating by customers. The star rating is provided as an input parameter.")
async def get_brand_names_by_rating(star_rating: int = Query(..., description="Star rating given by the customer")):
    cursor.execute("SELECT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.StarRating = ?", (star_rating,))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": [row[0] for row in result]}

# Endpoint to get the count of customers who purchased a specific brand
@app.get("/v1/beer_factory/count_customers_by_brand", operation_id="get_count_customers_by_brand", summary="Retrieves the number of customers who have purchased a specific brand of root beer. The operation requires the first and last name of the customer, as well as a pattern for the brand name. The result is a count of customers who match the provided criteria.")
async def get_count_customers_by_brand(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), brand_name_pattern: str = Query(..., description="Pattern for the brand name (e.g., 'Henry Weinhard%s')")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeer AS T3 ON T2.RootBeerID = T3.RootBeerID INNER JOIN rootbeerbrand AS T4 ON T3.BrandID = T4.BrandID WHERE T1.First = ? AND T1.Last = ? AND T4.BrandName LIKE ?", (first, last, brand_name_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top brewery name without social media presence
@app.get("/v1/beer_factory/top_brewery_without_social_media", operation_id="get_top_brewery_without_social_media", summary="Retrieves the name of the brewery with the highest production volume that does not have a presence on any social media platforms. The brewery is determined by analyzing the production data and checking for the absence of both a Facebook page and a Twitter account.")
async def get_top_brewery_without_social_media():
    cursor.execute("SELECT T2.BreweryName FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID WHERE T2.FacebookPage IS NULL AND T2.Twitter IS NULL GROUP BY T2.BrandID ORDER BY COUNT(T1.BrandID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"brewery_name": []}
    return {"brewery_name": result[0]}

# Endpoint to get the top location name for a specific brand
@app.get("/v1/beer_factory/top_location_by_brand", operation_id="get_top_location_by_brand", summary="Retrieves the top location for a given brand, based on the number of occurrences in the database. The operation considers only the specified locations. The brand name and location names are required as input parameters.")
async def get_top_location_by_brand(brand_name: str = Query(..., description="Name of the brand"), location_name_1: str = Query(..., description="First location name"), location_name_2: str = Query(..., description="Second location name")):
    cursor.execute("SELECT T3.LocationName FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID INNER JOIN location AS T3 ON T1.LocationID = T3.LocationID WHERE T2.BrandName = ? AND T3.LocationName IN (?, ?) GROUP BY T1.LocationID ORDER BY COUNT(T1.BrandID) DESC LIMIT 1", (brand_name, location_name_1, location_name_2))
    result = cursor.fetchone()
    if not result:
        return {"location_name": []}
    return {"location_name": result[0]}

# Endpoint to get the count of root beer brands with specific attributes
@app.get("/v1/beer_factory/count_brands_by_attributes", operation_id="get_count_brands_by_attributes", summary="Retrieve the number of root beer brands that match the specified container type, brand name, and purchase date pattern. The response provides a count of brands that meet all the given criteria.")
async def get_count_brands_by_attributes(container_type: str = Query(..., description="Type of container (e.g., 'Can')"), brand_name: str = Query(..., description="Name of the brand"), purchase_date_pattern: str = Query(..., description="Pattern for the purchase date (e.g., '2016%')")):
    cursor.execute("SELECT COUNT(T1.BrandID) FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID WHERE T1.ContainerType = ? AND T2.BrandName = ? AND T1.PurchaseDate LIKE ?", (container_type, brand_name, purchase_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get brand names with a specific star rating and minimum number of reviews
@app.get("/v1/beer_factory/brand_names_by_star_rating", operation_id="get_brand_names_by_star_rating", summary="Retrieves the names of beer brands that have a specified star rating and a minimum number of reviews. The operation filters brands based on the provided star rating and ensures that each brand has at least the specified number of reviews.")
async def get_brand_names_by_star_rating(star_rating: int = Query(..., description="Star rating of the brand"), min_reviews: int = Query(..., description="Minimum number of reviews")):
    cursor.execute("SELECT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.StarRating = ? GROUP BY T2.BrandID HAVING COUNT(T2.StarRating) >= ?", (star_rating, min_reviews))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": result}

# Endpoint to get the purchase ratio of brands within a date range and brewery name
@app.get("/v1/beer_factory/brand_purchase_ratio", operation_id="get_brand_purchase_ratio", summary="Retrieves the proportion of purchases for each brand produced by a specific brewery within a given date range. The calculation is based on the total number of purchases for each brand during the specified period, divided by the total number of purchases for all brands produced by the brewery.")
async def get_brand_purchase_ratio(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), brewery_name: str = Query(..., description="Name of the brewery")):
    cursor.execute("SELECT T1.BrandName, CAST(SUM(CASE WHEN T2.PurchaseDate >= ? AND T2.PurchaseDate <= ? THEN 1 ELSE 0 END) AS REAL) / COUNT(T2.BrandID) AS purchase FROM rootbeerbrand AS T1 INNER JOIN rootbeer AS T2 ON T1.BrandID = T2.BrandID WHERE T1.BreweryName = ? GROUP BY T2.BrandID", (start_date, end_date, brewery_name))
    result = cursor.fetchall()
    if not result:
        return {"purchase_ratio": []}
    return {"purchase_ratio": result}

# Endpoint to get the top brand and customer based on profit margin and purchase count
@app.get("/v1/beer_factory/top_brand_customer", operation_id="get_top_brand_customer", summary="Retrieves the top-performing brand and its corresponding customer based on profit margin and purchase count. The brand is determined by the highest profit margin and the highest purchase count. The customer is identified by their unique ID.")
async def get_top_brand_customer():
    cursor.execute("SELECT T3.BrandName, T2.CustomerID FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID GROUP BY T3.BrandID ORDER BY T3.CurrentRetailPrice - T3.WholesaleCost, COUNT(T1.BrandID) DESC LIMIT 1")
    result = cursor.fetchall()
    if not result:
        return {"top_brand_customer": []}
    return {"top_brand_customer": result}

# Endpoint to get customer details based on gender, city, and email subscription status
@app.get("/v1/beer_factory/customer_details", operation_id="get_customer_details", summary="Retrieves the first name, last name, and phone number of customers who match the specified gender, city, and email subscription status. This operation is useful for obtaining detailed customer information based on these criteria.")
async def get_customer_details(gender: str = Query(..., description="Gender of the customer"), city: str = Query(..., description="City of the customer"), subscribed_to_email_list: str = Query(..., description="Email subscription status of the customer")):
    cursor.execute("SELECT First, Last, PhoneNumber FROM customers WHERE Gender = ? AND City = ? AND SubscribedToEmaillist = ?", (gender, city, subscribed_to_email_list))
    result = cursor.fetchall()
    if not result:
        return {"customer_details": []}
    return {"customer_details": result}

# Endpoint to get the percentage of root beer sold in a specific container type within a year
@app.get("/v1/beer_factory/container_type_percentage", operation_id="get_container_type_percentage", summary="Retrieves the percentage of root beer sales for a specific container type within a given year. The calculation is based on the count of root beer sales for the specified container type relative to the total root beer sales in the same year.")
async def get_container_type_percentage(container_type: str = Query(..., description="Type of container"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN ContainerType = ? THEN RootBeerID ELSE NULL END) AS REAL) * 100 / COUNT(RootBeerID) FROM rootbeer WHERE PurchaseDate LIKE ?", (container_type, year + '%'))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get brand names brewed within a specific year range
@app.get("/v1/beer_factory/brands_by_year_range", operation_id="get_brands_by_year_range", summary="Retrieves a list of brand names that were first brewed within the specified year range, sorted in descending order by the year they were first brewed. The start and end years are used to filter the results.")
async def get_brands_by_year_range(start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT BrandName FROM rootbeerbrand WHERE FirstBrewedYear BETWEEN ? AND ? ORDER BY FirstBrewedYear DESC", (start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": result}

# Endpoint to get the brand with the most reviews for a specific star rating
@app.get("/v1/beer_factory/top_brand_by_star_rating", operation_id="get_top_brand_by_star_rating", summary="Retrieves the brand with the highest number of reviews for a given star rating. The star rating is specified as an input parameter.")
async def get_top_brand_by_star_rating(star_rating: int = Query(..., description="Star rating of the brand")):
    cursor.execute("SELECT BrandID FROM rootbeerreview WHERE StarRating = ? GROUP BY BrandID ORDER BY COUNT(BrandID) DESC LIMIT 1", (star_rating,))
    result = cursor.fetchone()
    if not result:
        return {"brand_id": []}
    return {"brand_id": result[0]}

# Endpoint to get the percentage of transactions made with a specific credit card type
@app.get("/v1/beer_factory/transaction_percentage_by_credit_card", operation_id="get_transaction_percentage_by_credit_card", summary="Retrieves the percentage of transactions made using a specific credit card type. The calculation is based on the count of transactions made with the specified credit card type divided by the total number of transactions.")
async def get_transaction_percentage_by_credit_card(credit_card_type: str = Query(..., description="Type of credit card")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN CreditCardType = ? THEN TransactionID ELSE NULL END) AS REAL) * 100 / COUNT(TransactionID) FROM `transaction`", (credit_card_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of brands with specific attributes
@app.get("/v1/beer_factory/brand_count_by_attributes", operation_id="get_brand_count_by_attributes", summary="Retrieves the count of beer brands that match the specified attributes, including the use of corn syrup, artificial sweetener, and availability in cans.")
async def get_brand_count_by_attributes(corn_syrup: str = Query(..., description="Corn syrup attribute"), artificial_sweetener: str = Query(..., description="Artificial sweetener attribute"), available_in_cans: str = Query(..., description="Availability in cans attribute")):
    cursor.execute("SELECT COUNT(BrandID) FROM rootbeerbrand WHERE CornSyrup = ? AND ArtificialSweetener = ? AND AvailableInCans = ?", (corn_syrup, artificial_sweetener, available_in_cans))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of transactions at a specific location
@app.get("/v1/beer_factory/transaction_percentage_by_location", operation_id="get_transaction_percentage_by_location", summary="Retrieves the percentage of total transactions that occurred at a specified location. The calculation is based on the count of transactions at the given location divided by the total number of transactions across all locations.")
async def get_transaction_percentage_by_location(location_name: str = Query(..., description="Name of the location")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.LocationName = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.TransactionID) FROM `transaction` AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID", (location_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average number of root beer purchases per day for caffeinated brands
@app.get("/v1/beer_factory/average_rootbeer_purchases_per_day", operation_id="get_average_rootbeer_purchases_per_day", summary="Retrieves the average daily purchase count for caffeinated root beer brands. The operation calculates this average by dividing the total number of purchases by the number of unique purchase dates. The caffeinated status of the root beer brand is used to filter the results.")
async def get_average_rootbeer_purchases_per_day(caffeinated: str = Query(..., description="Caffeinated status of the root beer brand (TRUE or FALSE)")):
    cursor.execute("SELECT CAST(COUNT(T2.RootBeerID) AS REAL) / COUNT(DISTINCT T2.PurchaseDate) FROM rootbeerbrand AS T1 INNER JOIN rootbeer AS T2 ON T1.BrandID = T2.BrandID INNER JOIN `transaction` AS T3 ON T2.RootBeerID = T3.RootBeerID WHERE T1.Caffeinated = ?", (caffeinated,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the brand names and container types of the most and least profitable root beers
@app.get("/v1/beer_factory/most_and_least_profitable_rootbeers", operation_id="get_most_and_least_profitable_rootbeers", summary="Retrieves the brand names and container types of the most and least profitable root beers. The profitability is determined by the difference between the current retail price and the wholesale cost. The operation returns the top and bottom entries based on this profitability metric.")
async def get_most_and_least_profitable_rootbeers():
    cursor.execute("SELECT * FROM ( SELECT T1.BrandName, T2.ContainerType FROM rootbeerbrand AS T1 INNER JOIN rootbeer AS T2 ON T1.BrandID = T2.BrandID ORDER BY T1.CurrentRetailPrice - T1.WholesaleCost DESC LIMIT 1 ) UNION ALL SELECT * FROM ( SELECT T3.BrandName, T4.ContainerType FROM rootbeerbrand AS T3 INNER JOIN rootbeer AS T4 ON T3.BrandID = T4.BrandID ORDER BY T3.CurrentRetailPrice - T3.WholesaleCost ASC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"rootbeers": []}
    return {"rootbeers": result}

# Endpoint to get the average purchase price of root beer in a specific container type and above a certain price
@app.get("/v1/beer_factory/average_purchase_price_by_container_type", operation_id="get_average_purchase_price_by_container_type", summary="Retrieves the average purchase price of root beer in a specified container type, considering only transactions with a purchase price above a given minimum threshold.")
async def get_average_purchase_price_by_container_type(container_type: str = Query(..., description="Container type of the root beer"), min_purchase_price: float = Query(..., description="Minimum purchase price")):
    cursor.execute("SELECT AVG(T2.PurchasePrice) FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T1.ContainerType = ? AND T2.PurchasePrice > ?", (container_type, min_purchase_price))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the count of root beer brands at a specific geolocation and container type
@app.get("/v1/beer_factory/count_brands_by_geolocation_and_container_type", operation_id="get_count_brands_by_geolocation_and_container_type", summary="Retrieve the total number of root beer brands available at a specific geographical location and container type. The location is determined by latitude and longitude coordinates, while the container type can be a bottle, can, or keg.")
async def get_count_brands_by_geolocation_and_container_type(latitude: float = Query(..., description="Latitude of the location"), longitude: float = Query(..., description="Longitude of the location"), container_type: str = Query(..., description="Container type of the root beer")):
    cursor.execute("SELECT COUNT(T4.BrandID) FROM `transaction` AS T1 INNER JOIN geolocation AS T2 ON T1.LocationID = T2.LocationID INNER JOIN location AS T3 ON T1.LocationID = T3.LocationID INNER JOIN rootbeer AS T4 ON T1.RootBeerID = T4.RootBeerID WHERE T2.Latitude = ? AND T2.Longitude = ? AND T4.ContainerType = ?", (latitude, longitude, container_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of customers with a star rating above a certain value who are not subscribed to the email list
@app.get("/v1/beer_factory/percentage_customers_by_star_rating_and_subscription", operation_id="get_percentage_customers_by_star_rating_and_subscription", summary="Retrieves the percentage of customers who have rated above a specified star rating and are not subscribed to the email list. The operation calculates this percentage by counting the number of customers with a star rating above the provided minimum and dividing it by the total number of customers. The input parameters include the minimum star rating and the subscription status to the email list.")
async def get_percentage_customers_by_star_rating_and_subscription(min_star_rating: int = Query(..., description="Minimum star rating"), subscribed_to_email_list: str = Query(..., description="Subscription status to the email list (TRUE or FALSE)")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.StarRating > ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.CustomerID) FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.SubscribedToEmaillist = ?", (min_star_rating, subscribed_to_email_list))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top brand name with the highest number of reviews with a specific star rating in a given year
@app.get("/v1/beer_factory/top_brand_by_star_rating_and_year", operation_id="get_top_brand_by_star_rating_and_year", summary="Retrieves the brand with the highest number of reviews that have a specific star rating in a given year. The star rating and year are provided as input parameters.")
async def get_top_brand_by_star_rating_and_year(star_rating: int = Query(..., description="Star rating of the review"), year: str = Query(..., description="Year of the review in 'YYYY' format")):
    cursor.execute("SELECT T3.BrandName FROM rootbeer AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T2.StarRating = ? AND strftime('%Y', T2.ReviewDate) = ? GROUP BY T1.BrandID ORDER BY COUNT(T2.BrandID) DESC LIMIT 1", (star_rating, year))
    result = cursor.fetchone()
    if not result:
        return {"brand_name": []}
    return {"brand_name": result[0]}

# Endpoint to get the count of customers by gender and artificial sweetener preference
@app.get("/v1/beer_factory/count_customers_by_gender_and_sweetener", operation_id="get_count_customers_by_gender_and_sweetener", summary="Retrieves the total number of customers categorized by gender and their preference for artificial sweeteners in root beer brands. The operation filters customers based on their gender and the artificial sweetener status of the root beer brands they have purchased.")
async def get_count_customers_by_gender_and_sweetener(gender: str = Query(..., description="Gender of the customer (M or F)"), artificial_sweetener: str = Query(..., description="Artificial sweetener status of the root beer brand (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeer AS T3 ON T2.RootBeerID = T3.RootBeerID INNER JOIN rootbeerbrand AS T4 ON T3.BrandID = T4.BrandID WHERE T1.Gender = ? AND T4.ArtificialSweetener = ?", (gender, artificial_sweetener))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in count of root beer brands using cane sugar and corn syrup
@app.get("/v1/beer_factory/difference_in_sweetener_count", operation_id="get_difference_in_sweetener_count", summary="Retrieves the difference in the number of root beer brands that use cane sugar and corn syrup as sweeteners. The operation filters the brands based on the provided cane sugar and corn syrup statuses and returns the count difference.")
async def get_difference_in_sweetener_count(cane_sugar: str = Query(..., description="Cane sugar status of the root beer brand (TRUE or FALSE)"), corn_syrup: str = Query(..., description="Corn syrup status of the root beer brand (TRUE or FALSE)")):
    cursor.execute("SELECT COUNT(CASE WHEN T3.CaneSugar = ? THEN T1.BrandID ELSE NULL END) - COUNT(CASE WHEN T3.CornSyrup = ? THEN T1.BrandID ELSE NULL END) AS DIFFERENCE FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID", (cane_sugar, corn_syrup))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the top brewery name with the highest number of transactions in a given year
@app.get("/v1/beer_factory/top_brewery_by_year", operation_id="get_top_brewery_by_year", summary="Retrieves the name of the brewery with the most transactions in a specified year. The year is provided in 'YYYY' format. The operation returns the brewery name that had the highest number of transactions in the given year.")
async def get_top_brewery_by_year(year: str = Query(..., description="Year of the transaction in 'YYYY' format")):
    cursor.execute("SELECT T3.BreweryName FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T2.TransactionDate LIKE ? GROUP BY T3.BrandID ORDER BY COUNT(T1.BrandID) DESC LIMIT 1", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"brewery_name": []}
    return {"brewery_name": result[0]}

# Endpoint to get the percentage of customers who prefer a specific brand in a given city, gender, and transaction date pattern
@app.get("/v1/beer_factory/customer_brand_preference_percentage", operation_id="get_customer_brand_preference_percentage", summary="Retrieves the percentage of customers in a specific city and gender who prefer a given root beer brand, based on their transaction history within a certain date pattern.")
async def get_customer_brand_preference_percentage(brand_name: str = Query(..., description="Brand name of the root beer"), city: str = Query(..., description="City of the customer"), gender: str = Query(..., description="Gender of the customer"), transaction_date_pattern: str = Query(..., description="Transaction date pattern (e.g., '2014%')")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T4.BrandName = ? THEN T1.CustomerID ELSE NULL END) AS REAL) * 100 / COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeer AS T3 ON T2.RootBeerID = T3.RootBeerID INNER JOIN rootbeerbrand AS T4 ON T3.BrandID = T4.BrandID WHERE T1.City = ? AND T1.Gender = ? AND T2.TransactionDate LIKE ?", (brand_name, city, gender, transaction_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in the average number of transactions with and without honey
@app.get("/v1/beer_factory/transaction_honey_difference", operation_id="get_transaction_honey_difference", summary="Get the difference in the average number of transactions with and without honey")
async def get_transaction_honey_difference(honey_true: str = Query(..., description="Value indicating honey is true (e.g., 'TRUE')"), honey_false: str = Query(..., description="Value indicating honey is false (e.g., 'FALSE')")):
    cursor.execute("SELECT (CAST(SUM(CASE WHEN T1.Honey = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(DISTINCT T3.TransactionDate)) - (CAST(SUM(CASE WHEN T1.Honey <> ? THEN 1 ELSE 0 END) AS REAL) / COUNT(DISTINCT T3.TransactionDate)) FROM rootbeerbrand AS T1 INNER JOIN rootbeer AS T2 ON T1.BrandID = T2.BrandID INNER JOIN `transaction` AS T3 ON T2.RootBeerID = T3.RootBeerID", (honey_true, honey_false))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct customer details based on location name and credit card type
@app.get("/v1/beer_factory/customer_details_by_location_and_credit_card", operation_id="get_customer_details_by_location_and_credit_card", summary="Retrieves unique customer details, including first name, last name, and email, based on a specific location and credit card type. This operation filters customers who have made transactions at the specified location using the provided credit card type.")
async def get_customer_details_by_location_and_credit_card(location_name: str = Query(..., description="Name of the location"), credit_card_type: str = Query(..., description="Type of credit card")):
    cursor.execute("SELECT DISTINCT T1.First, T1.Last, T1.Email FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN location AS T3 ON T2.LocationID = T3.LocationID WHERE T3.LocationName = ? AND T2.CreditCardType = ?", (location_name, credit_card_type))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get distinct brand names based on star rating and review date range
@app.get("/v1/beer_factory/brand_names_by_star_rating_and_review_date", operation_id="get_brand_names_by_star_rating_and_review_date", summary="Retrieves unique brand names of beers that have been reviewed within a specified date range and received a particular star rating. The star rating and date range are provided as input parameters.")
async def get_brand_names_by_star_rating_and_review_date(star_rating: int = Query(..., description="Star rating of the review"), start_date: str = Query(..., description="Start date of the review period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the review period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.StarRating = ? AND T2.ReviewDate BETWEEN ? AND ?", (star_rating, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": result}

# Endpoint to get distinct geolocations based on credit card type
@app.get("/v1/beer_factory/geolocations_by_credit_card_type", operation_id="get_geolocations_by_credit_card_type", summary="Retrieve unique geographical locations where transactions were made using a specific credit card type. The response includes latitude and longitude coordinates for each location.")
async def get_geolocations_by_credit_card_type(credit_card_type: str = Query(..., description="Type of credit card")):
    cursor.execute("SELECT DISTINCT T2.Latitude, T2.Longitude FROM `transaction` AS T1 INNER JOIN geolocation AS T2 ON T1.LocationID = T2.LocationID WHERE T1.CreditCardType = ?", (credit_card_type,))
    result = cursor.fetchall()
    if not result:
        return {"geolocations": []}
    return {"geolocations": result}

# Endpoint to get the count of customers based on city and credit card type
@app.get("/v1/beer_factory/customer_count_by_city_and_credit_card", operation_id="get_customer_count_by_city_and_credit_card", summary="Retrieves the total number of customers from a specific city who have used a particular credit card type for transactions. This operation provides a quantitative insight into customer distribution based on their geographical location and payment method.")
async def get_customer_count_by_city_and_credit_card(city: str = Query(..., description="City of the customer"), credit_card_type: str = Query(..., description="Type of credit card")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.City = ? AND T2.CreditCardType = ?", (city, credit_card_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct cities based on star rating and review date range
@app.get("/v1/beer_factory/cities_by_star_rating_and_review_date", operation_id="get_cities_by_star_rating_and_review_date", summary="Retrieve a list of unique cities where customers have left reviews with a specific star rating within a specified date range. The star rating and date range are provided as input parameters.")
async def get_cities_by_star_rating_and_review_date(star_rating: int = Query(..., description="Star rating of the review"), start_date: str = Query(..., description="Start date of the review period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the review period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T1.City FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.StarRating = ? AND T2.ReviewDate BETWEEN ? AND ?", (star_rating, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": result}

# Endpoint to get brand names based on customer's first and last name
@app.get("/v1/beer_factory/brand_names_by_customer_name", operation_id="get_brand_names_by_customer_name", summary="Retrieves the brand names associated with a specific customer, identified by their first and last name. This operation returns a list of brand names that the customer has reviewed, based on the provided first and last name.")
async def get_brand_names_by_customer_name(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T3.BrandName FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeerbrand AS T3 ON T2.BrandID = T3.BrandID WHERE T1.First = ? AND T1.Last = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": result}

# Endpoint to get the most reviewed brand name based on star rating
@app.get("/v1/beer_factory/most_reviewed_brand_by_star_rating", operation_id="get_most_reviewed_brand_by_star_rating", summary="Retrieves the brand name that has the highest number of reviews for a given star rating. The star rating is used to filter the reviews and determine the most reviewed brand.")
async def get_most_reviewed_brand_by_star_rating(star_rating: int = Query(..., description="Star rating of the review")):
    cursor.execute("SELECT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T2.BrandID = T1.BrandID WHERE T2.StarRating = ? GROUP BY T1.BrandName ORDER BY COUNT(T1.BrandName) DESC LIMIT 1", (star_rating,))
    result = cursor.fetchone()
    if not result:
        return {"brand_name": []}
    return {"brand_name": result[0]}

# Endpoint to get the most used credit card type for non-alcoholic root beer
@app.get("/v1/beer_factory/most_used_credit_card_type_for_non_alcoholic", operation_id="get_most_used_credit_card_type_for_non_alcoholic", summary="Retrieves the most frequently used credit card type for non-alcoholic root beer transactions. The operation filters transactions based on the alcoholic status of the root beer brand and returns the credit card type with the highest transaction count.")
async def get_most_used_credit_card_type_for_non_alcoholic(alcoholic: str = Query(..., description="Alcoholic status of the root beer (e.g., 'FALSE')")):
    cursor.execute("SELECT T2.CreditCardType FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID WHERE T3.Alcoholic = ? GROUP BY T2.CreditCardType ORDER BY COUNT(T2.CreditCardType) DESC LIMIT 1", (alcoholic,))
    result = cursor.fetchone()
    if not result:
        return {"credit_card_type": []}
    return {"credit_card_type": result[0]}

# Endpoint to get customer names who gave a specific star rating
@app.get("/v1/beer_factory/customer_names_by_star_rating", operation_id="get_customer_names_by_star_rating", summary="Retrieves the first and last names of customers who have given a specific star rating to a beer. The star rating is provided as an input parameter.")
async def get_customer_names_by_star_rating(star_rating: int = Query(..., description="Star rating given by the customer")):
    cursor.execute("SELECT T1.First, T1.Last FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.StarRating = ?", (star_rating,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the latitude of the most frequent location for a specific brand
@app.get("/v1/beer_factory/most_frequent_latitude_by_brand", operation_id="get_most_frequent_latitude_by_brand", summary="Retrieves the latitude of the most frequently occurring location for a specified root beer brand. The brand name is provided as an input parameter.")
async def get_most_frequent_latitude_by_brand(brand_name: str = Query(..., description="Brand name of the root beer")):
    cursor.execute("SELECT T3.Latitude FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID INNER JOIN geolocation AS T3 ON T1.LocationID = T3.LocationID WHERE T2.BrandName = ? GROUP BY T3.Latitude ORDER BY COUNT(T1.BrandID) DESC LIMIT 1", (brand_name,))
    result = cursor.fetchone()
    if not result:
        return {"latitude": []}
    return {"latitude": result[0]}

# Endpoint to get the most common star rating for brands with a specific corn syrup status
@app.get("/v1/beer_factory/most_common_star_rating_by_corn_syrup", operation_id="get_most_common_star_rating_by_corn_syrup", summary="Retrieves the most frequently assigned star rating for beer brands that either use or do not use corn syrup, based on the provided corn syrup status.")
async def get_most_common_star_rating_by_corn_syrup(corn_syrup: str = Query(..., description="Corn syrup status of the brand (TRUE or FALSE)")):
    cursor.execute("SELECT T2.StarRating FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T1.CornSyrup = ? GROUP BY T2.StarRating ORDER BY COUNT(T2.StarRating) DESC LIMIT 1", (corn_syrup,))
    result = cursor.fetchone()
    if not result:
        return {"star_rating": []}
    return {"star_rating": result[0]}

# Endpoint to get the latitude and longitude for a specific zip code
@app.get("/v1/beer_factory/geolocation_by_zip_code", operation_id="get_geolocation_by_zip_code", summary="Retrieves the geographical coordinates (latitude and longitude) for a specific location identified by its zip code.")
async def get_geolocation_by_zip_code(zip_code: int = Query(..., description="Zip code of the location")):
    cursor.execute("SELECT T2.Latitude, T2.Longitude FROM location AS T1 INNER JOIN geolocation AS T2 ON T1.LocationID = T2.LocationID WHERE T1.ZipCode = ?", (zip_code,))
    result = cursor.fetchall()
    if not result:
        return {"geolocation": []}
    return {"geolocation": result}

# Endpoint to get distinct brand names for a specific latitude and longitude
@app.get("/v1/beer_factory/brand_names_by_latitude_longitude", operation_id="get_brand_names_by_latitude_longitude", summary="Retrieve a unique set of beer brand names associated with a specific geographical location, identified by its latitude and longitude coordinates.")
async def get_brand_names_by_latitude_longitude(latitude: str = Query(..., description="Latitude of the location"), longitude: str = Query(..., description="Longitude of the location")):
    cursor.execute("SELECT DISTINCT T2.BrandName FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID INNER JOIN geolocation AS T3 ON T1.LocationID = T3.LocationID WHERE T3.Latitude = ? AND T3.Longitude = ?", (latitude, longitude))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": result}

# Endpoint to get the average profit margin for a specific container type
@app.get("/v1/beer_factory/average_profit_margin_by_container_type", operation_id="get_average_profit_margin_by_container_type", summary="Retrieves the average profit margin for a specific container type of root beer. The calculation is based on the difference between the current retail price and the wholesale cost of the root beer brand associated with the given container type.")
async def get_average_profit_margin_by_container_type(container_type: str = Query(..., description="Container type of the root beer")):
    cursor.execute("SELECT AVG(T2.CurrentRetailPrice - T2.WholesaleCost) FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID WHERE T1.ContainerType = ?", (container_type,))
    result = cursor.fetchone()
    if not result:
        return {"average_profit_margin": []}
    return {"average_profit_margin": result[0]}

# Endpoint to get the percentage of reviews with a specific star rating for a specific credit card type
@app.get("/v1/beer_factory/percentage_reviews_by_star_rating_credit_card", operation_id="get_percentage_reviews_by_star_rating_credit_card", summary="Retrieves the percentage of reviews with a specified star rating for a specific credit card type. This operation calculates the proportion of reviews with the given star rating, out of all reviews made by customers who used the specified credit card type for their transactions.")
async def get_percentage_reviews_by_star_rating_credit_card(star_rating: int = Query(..., description="Star rating of the review"), credit_card_type: str = Query(..., description="Credit card type used in the transaction")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.StarRating = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.CustomerID) FROM rootbeerreview AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.CreditCardType = ?", (star_rating, credit_card_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get brand IDs with star ratings above a specific value
@app.get("/v1/beer_factory/brand_ids_above_star_rating", operation_id="get_brand_ids_above_star_rating", summary="Retrieves the unique identifiers of beer brands that have received star ratings exceeding a specified minimum value. This operation allows users to filter brands based on their star ratings, providing a means to identify highly-rated beer brands.")
async def get_brand_ids_above_star_rating(min_star_rating: int = Query(..., description="Minimum star rating")):
    cursor.execute("SELECT BrandID FROM rootbeerreview WHERE StarRating > ?", (min_star_rating,))
    result = cursor.fetchall()
    if not result:
        return {"brand_ids": []}
    return {"brand_ids": result}

# Endpoint to get the count of brand IDs for a specific container type and purchase date range
@app.get("/v1/beer_factory/count_brand_ids_by_container_type_date_range", operation_id="get_count_brand_ids_by_container_type_date_range", summary="Retrieve the total number of unique brands that meet the specified container type and purchase date range criteria. The container type and date range are provided as input parameters, allowing for a targeted count of brand IDs.")
async def get_count_brand_ids_by_container_type_date_range(container_type: str = Query(..., description="Container type of the root beer"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(BrandID) FROM rootbeer WHERE ContainerType = ? AND PurchaseDate BETWEEN ? AND ?", (container_type, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names who gave a specific star rating and review
@app.get("/v1/beer_factory/customer_names_by_star_rating_review", operation_id="get_customer_names_by_star_rating_review", summary="Retrieves the first and last names of customers who have provided a specific star rating and review for a product. The operation filters customers based on the provided star rating and review, and returns their names.")
async def get_customer_names_by_star_rating_review(star_rating: int = Query(..., description="Star rating given by the customer"), review: str = Query(..., description="Review given by the customer")):
    cursor.execute("SELECT T1.First, T1.Last FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.StarRating = ? AND T2.Review = ?", (star_rating, review))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get distinct emails and phone numbers of customers based on star rating, city, and review date range
@app.get("/v1/beer_factory/customer_contact_info", operation_id="get_customer_contact_info", summary="Retrieve unique contact information (emails and phone numbers) for customers who have provided a star rating above a specified threshold, reside in a particular city, and have submitted reviews within a defined date range.")
async def get_customer_contact_info(min_star_rating: int = Query(..., description="Minimum star rating"), city: str = Query(..., description="City of the customer"), start_date: str = Query(..., description="Start date of the review period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the review period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T1.Email, T1.PhoneNumber FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.StarRating > ? AND T1.City = ? AND T2.ReviewDate BETWEEN ? AND ?", (min_star_rating, city, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"contact_info": []}
    return {"contact_info": result}

# Endpoint to get the count of customers based on star rating, city, gender, email subscription status, and review date range
@app.get("/v1/beer_factory/customer_count_by_criteria", operation_id="get_customer_count_by_criteria", summary="Retrieve the number of customers who match specific criteria, including a given star rating, city, gender, email subscription status, and have written a review within a specified date range.")
async def get_customer_count_by_criteria(star_rating: int = Query(..., description="Star rating"), city: str = Query(..., description="City of the customer"), gender: str = Query(..., description="Gender of the customer"), subscribed_to_email_list: str = Query(..., description="Email subscription status"), start_date: str = Query(..., description="Start date of the review period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the review period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.StarRating = ? AND T1.City = ? AND T1.Gender = ? AND T1.SubscribedToEmaillist = ? AND T2.ReviewDate BETWEEN ? AND ?", (star_rating, city, gender, subscribed_to_email_list, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct brewery and brand names based on purchase date and container type
@app.get("/v1/beer_factory/brewery_brand_info", operation_id="get_brewery_brand_info", summary="Retrieve unique brewery and brand names for root beers bought before a specified date and in a particular container type. This operation filters the data based on the provided purchase date and container type, ensuring that only relevant and distinct brewery and brand names are returned.")
async def get_brewery_brand_info(purchase_date: str = Query(..., description="Purchase date in 'YYYY-MM-DD' format"), container_type: str = Query(..., description="Container type")):
    cursor.execute("SELECT DISTINCT T2.BreweryName, T2.BrandName FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID WHERE T1.PurchaseDate < ? AND T1.ContainerType = ?", (purchase_date, container_type))
    result = cursor.fetchall()
    if not result:
        return {"brewery_brand_info": []}
    return {"brewery_brand_info": result}

# Endpoint to get the brand name of the oldest brewed root beer in a specific container type
@app.get("/v1/beer_factory/oldest_brewed_brand", operation_id="get_oldest_brewed_brand", summary="Retrieve the name of the root beer brand that was brewed the earliest in a specified container type. The operation filters the brands based on the provided first brewed year and container type, and returns the brand name of the oldest brewed root beer that meets the criteria.")
async def get_oldest_brewed_brand(first_brewed_year: str = Query(..., description="First brewed year in 'YYYY-MM-DD' format"), container_type: str = Query(..., description="Container type")):
    cursor.execute("SELECT T2.BrandName FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID WHERE T2.FirstBrewedYear < ? AND T1.ContainerType = ? ORDER BY T2.FirstBrewedYear LIMIT 1", (first_brewed_year, container_type))
    result = cursor.fetchone()
    if not result:
        return {"brand_name": []}
    return {"brand_name": result[0]}

# Endpoint to get the count of transactions for a specific customer based on name, credit card type, and transaction date range
@app.get("/v1/beer_factory/transaction_count_by_customer", operation_id="get_transaction_count_by_customer", summary="Retrieve the total number of transactions made by a customer, identified by their first and last name, credit card type, and a specified date range. This operation provides a comprehensive view of a customer's transaction history within the given time frame.")
async def get_transaction_count_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer"), credit_card_type: str = Query(..., description="Credit card type"), start_date: str = Query(..., description="Start date of the transaction period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the transaction period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.CustomerID) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.First = ? AND T1.Last = ? AND T2.CreditCardType = ? AND T2.TransactionDate BETWEEN ? AND ?", (first_name, last_name, credit_card_type, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average star rating for a specific brand and gender within a date range
@app.get("/v1/beer_factory/average_star_rating", operation_id="get_average_star_rating", summary="Retrieves the average star rating for a specific beer brand and customer gender within a specified date range. The operation considers reviews from customers of the given gender who have reviewed the specified brand between the provided start and end dates.")
async def get_average_star_rating(brand_id: int = Query(..., description="Brand ID"), gender: str = Query(..., description="Gender of the customer"), start_date: str = Query(..., description="Start date of the review period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the review period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT AVG(T2.StarRating) FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.BrandID = ? AND T1.Gender = ? AND T2.ReviewDate BETWEEN ? AND ?", (brand_id, gender, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_star_rating": []}
    return {"average_star_rating": result[0]}

# Endpoint to get the brand name based on customer ID, review, and star rating
@app.get("/v1/beer_factory/brand_name_by_review", operation_id="get_brand_name_by_review", summary="Retrieves the brand name of a beer based on a specific customer's review and star rating. The operation requires the customer's unique identifier, the exact review text, and the corresponding star rating to locate the brand name in the database.")
async def get_brand_name_by_review(customer_id: int = Query(..., description="Customer ID"), review: str = Query(..., description="Review text"), star_rating: int = Query(..., description="Star rating")):
    cursor.execute("SELECT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.CustomerID = ? AND T2.Review = ? AND T2.StarRating = ?", (customer_id, review, star_rating))
    result = cursor.fetchone()
    if not result:
        return {"brand_name": []}
    return {"brand_name": result[0]}

# Endpoint to get the total purchase price for transactions at a specific location, credit card type, and date range
@app.get("/v1/beer_factory/total_purchase_price", operation_id="get_total_purchase_price", summary="Retrieves the total purchase price for transactions made at a specific location, using a particular credit card type, and within a defined date range. The location is identified by its name, while the credit card type is specified by its type. The date range is determined by the start and end dates provided in 'YYYY-MM-DD' format.")
async def get_total_purchase_price(location_name: str = Query(..., description="Location name"), credit_card_type: str = Query(..., description="Credit card type"), start_date: str = Query(..., description="Start date of the transaction period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the transaction period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(T1.PurchasePrice) FROM `transaction` AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.LocationName = ? AND T1.CreditCardType = ? AND T1.TransactionDate BETWEEN ? AND ?", (location_name, credit_card_type, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"total_purchase_price": []}
    return {"total_purchase_price": result[0]}

# Endpoint to get the count of transactions at a specific location, credit card type, and date range
@app.get("/v1/beer_factory/transaction_count_by_location", operation_id="get_transaction_count_by_location", summary="Retrieve the total number of transactions that occurred at a specified location, using a particular credit card type, and within a defined date range. The response provides a comprehensive count of transactions that meet the given criteria, offering insights into transaction volume and trends.")
async def get_transaction_count_by_location(location_name: str = Query(..., description="Location name"), credit_card_type: str = Query(..., description="Credit card type"), start_date: str = Query(..., description="Start date of the transaction period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the transaction period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.TransactionID) FROM `transaction` AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.LocationName = ? AND T1.CreditCardType = ? AND T1.TransactionDate BETWEEN ? AND ?", (location_name, credit_card_type, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the review for a specific brand on a specific date
@app.get("/v1/beer_factory/review_by_brand_and_date", operation_id="get_review_by_brand_and_date", summary="Retrieves the review for a specific beer brand on a given date. The operation requires the brand name and the review date in 'YYYY-MM-DD' format as input parameters. The review data returned is associated with the provided brand and date.")
async def get_review_by_brand_and_date(brand_name: str = Query(..., description="Brand name"), review_date: str = Query(..., description="Review date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Review FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T1.BrandName = ? AND T2.ReviewDate = ?", (brand_name, review_date))
    result = cursor.fetchone()
    if not result:
        return {"review": []}
    return {"review": result[0]}

# Endpoint to get the brand name and price difference for root beer brands with a specific star rating and review date range
@app.get("/v1/beer_factory/brand_price_difference", operation_id="get_brand_price_difference", summary="Retrieves the brand name and price difference (current retail price minus wholesale cost) for root beer brands that have a specific star rating and were reviewed within a given date range.")
async def get_brand_price_difference(star_rating: int = Query(..., description="Star rating of the review"), start_date: str = Query(..., description="Start date of the review period in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the review period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.BrandName, T1.CurrentRetailPrice - T1.WholesaleCost AS result FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.StarRating = ? AND T2.ReviewDate BETWEEN ? AND ?", (star_rating, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"brands": []}
    return {"brands": result}

# Endpoint to get the first and last name of a customer and the time ago from their first purchase date to the review date
@app.get("/v1/beer_factory/customer_time_ago", operation_id="get_customer_time_ago", summary="Retrieves the full name of a customer and the time elapsed since their first purchase until a specified review date. The operation returns the first and last name of the customer, along with the number of days between the first purchase and review dates.")
async def get_customer_time_ago():
    cursor.execute("SELECT T1.First, T1.Last , strftime('%J', ReviewDate) - strftime('%J', FirstPurchaseDate) AS TIMEAGO FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": result}

# Endpoint to get distinct credit card types for customers with a specific first and last name
@app.get("/v1/beer_factory/credit_card_types", operation_id="get_credit_card_types", summary="Retrieves the unique credit card types used by a customer with a specific first and last name. This operation fetches data from the customers and transaction tables, filtering results based on the provided first and last name.")
async def get_credit_card_types(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT DISTINCT T2.CreditCardType FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.First = ? AND T1.Last = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"credit_card_types": []}
    return {"credit_card_types": result}

# Endpoint to get container type, brand name, and star rating for a specific root beer ID
@app.get("/v1/beer_factory/root_beer_details", operation_id="get_root_beer_details", summary="Retrieves the container type, brand name, and star rating associated with a specific root beer. The operation requires the root beer's unique identifier to fetch the relevant details from the database.")
async def get_root_beer_details(root_beer_id: int = Query(..., description="Root beer ID")):
    cursor.execute("SELECT T4.ContainerType, T3.BrandName, T1.StarRating FROM rootbeerreview AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID INNER JOIN rootbeer AS T4 ON T2.RootBeerID = T4.RootBeerID WHERE T2.RootBeerID = ?", (root_beer_id,))
    result = cursor.fetchall()
    if not result:
        return {"root_beer_details": []}
    return {"root_beer_details": result}

# Endpoint to get root beer IDs for customers with specific first and last names
@app.get("/v1/beer_factory/root_beer_ids_by_customer", operation_id="get_root_beer_ids_by_customer", summary="Retrieves the unique identifiers of root beers associated with customers who have the specified first and last names. This operation supports querying for two distinct customers simultaneously, allowing for comparison or aggregation of their root beer preferences.")
async def get_root_beer_ids_by_customer(first_name_1: str = Query(..., description="First name of the first customer"), last_name_1: str = Query(..., description="Last name of the first customer"), first_name_2: str = Query(..., description="First name of the second customer"), last_name_2: str = Query(..., description="Last name of the second customer")):
    cursor.execute("SELECT T2.RootBeerID FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T2.CustomerID = T1.CustomerID WHERE (T1.First = ? AND T1.Last = ?) OR (T1.First = ? AND T1.Last = ?)", (first_name_1, last_name_1, first_name_2, last_name_2))
    result = cursor.fetchall()
    if not result:
        return {"root_beer_ids": []}
    return {"root_beer_ids": result}

# Endpoint to get root beer IDs for specific brand names
@app.get("/v1/beer_factory/root_beer_ids_by_brand", operation_id="get_root_beer_ids_by_brand", summary="Retrieves the unique identifiers of root beers associated with up to five specified brand names. The operation filters root beers based on their brand names and returns their corresponding identifiers.")
async def get_root_beer_ids_by_brand(brand_name_1: str = Query(..., description="First brand name"), brand_name_2: str = Query(..., description="Second brand name"), brand_name_3: str = Query(..., description="Third brand name"), brand_name_4: str = Query(..., description="Fourth brand name"), brand_name_5: str = Query(..., description="Fifth brand name")):
    cursor.execute("SELECT T1.RootBeerID FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T2.BrandID = T1.BrandID WHERE T2.BrandName IN (?, ?, ?, ?, ?)", (brand_name_1, brand_name_2, brand_name_3, brand_name_4, brand_name_5))
    result = cursor.fetchall()
    if not result:
        return {"root_beer_ids": []}
    return {"root_beer_ids": result}

# Endpoint to get the count of container types for a specific customer and container type
@app.get("/v1/beer_factory/container_type_count", operation_id="get_container_type_count", summary="Retrieves the total count of a specific container type associated with a customer, based on the provided first and last name of the customer.")
async def get_container_type_count(container_type: str = Query(..., description="Container type"), first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T3.ContainerType) FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T2.CustomerID = T1.CustomerID INNER JOIN rootbeer AS T3 ON T3.RootBeerID = T2.RootBeerID WHERE T3.ContainerType = ? AND T1.First = ? AND T1.Last = ?", (container_type, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of root beer IDs for a specific state
@app.get("/v1/beer_factory/root_beer_count_by_state", operation_id="get_root_beer_count_by_state", summary="Retrieves the total number of unique root beer products sold in a specified state. The state is provided as an input parameter.")
async def get_root_beer_count_by_state(state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(T3.RootBeerID) FROM rootbeerbrand AS T1 INNER JOIN rootbeer AS T2 ON T1.BrandID = T2.BrandID INNER JOIN `transaction` AS T3 ON T2.RootBeerID = T3.RootBeerID WHERE T1.State = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in the count of brand IDs between two states for a specific container type
@app.get("/v1/beer_factory/brand_id_count_difference", operation_id="get_brand_id_count_difference", summary="Retrieves the difference in the number of unique brand IDs associated with a specific container type between two states. The operation compares the count of brand IDs in the first state with the count in the second state, returning the difference.")
async def get_brand_id_count_difference(state_1: str = Query(..., description="First state"), container_type: str = Query(..., description="Container type"), state_2: str = Query(..., description="Second state")):
    cursor.execute("SELECT ( SELECT COUNT(T1.BrandID) FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID WHERE T2.State = ? AND T1.ContainerType = ? ) - ( SELECT COUNT(T3.BrandID) FROM rootbeer AS T3 INNER JOIN rootbeerbrand AS T4 ON T3.BrandID = T4.BrandID WHERE T4.State = ? AND T3.ContainerType = ? ) AS DIFFERENCE", (state_1, container_type, state_2, container_type))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the percentage of transactions at one location compared to another
@app.get("/v1/beer_factory/transaction_percentage_comparison", operation_id="get_transaction_percentage_comparison", summary="Retrieve the percentage of transactions at the first specified location relative to the second specified location. This operation compares the transaction counts at two distinct locations and returns the percentage of transactions at the first location compared to the second location.")
async def get_transaction_percentage_comparison(location_name_1: str = Query(..., description="First location name"), location_name_2: str = Query(..., description="Second location name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.LocationName = ? THEN T1.TransactionID ELSE NULL END) AS REAL) * 100 / COUNT(CASE WHEN T2.LocationName = ? THEN T1.TransactionID ELSE NULL END) FROM `transaction` AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID", (location_name_1, location_name_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get top customers by total purchase price
@app.get("/v1/beer_factory/top_customers_by_purchase_price", operation_id="get_top_customers_by_purchase_price", summary="Retrieves a list of top customers ranked by their total purchase price. The list is limited to the specified number of customers. The response includes each customer's first name, last name, and credit card type.")
async def get_top_customers_by_purchase_price(limit: int = Query(..., description="Number of top customers to retrieve")):
    cursor.execute("SELECT T1.First, T1.Last, T2.CreditCardType FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID GROUP BY T1.CustomerID ORDER BY SUM(T2.PurchasePrice) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get total purchase price for root beer brands with specific characteristics
@app.get("/v1/beer_factory/total_purchase_price_by_brand", operation_id="get_total_purchase_price_by_brand", summary="Retrieve the total purchase price for root beer brands that meet the specified cane sugar and caffeine status. The response groups the results by brand name.")
async def get_total_purchase_price_by_brand(cane_sugar: str = Query(..., description="Cane sugar status (TRUE or FALSE)"), caffeinated: str = Query(..., description="Caffeinated status (TRUE or FALSE)")):
    cursor.execute("SELECT T1.BrandName, SUM(T3.PurchasePrice) FROM rootbeerbrand AS T1 INNER JOIN rootbeer AS T2 ON T1.BrandID = T2.BrandID INNER JOIN `transaction` AS T3 ON T2.RootBeerID = T3.RootBeerID WHERE T1.CaneSugar = ? AND T1.Caffeinated = ? GROUP BY T1.BrandName", (cane_sugar, caffeinated))
    result = cursor.fetchall()
    if not result:
        return {"brands": []}
    return {"brands": result}

# Endpoint to get the most popular root beer brand
@app.get("/v1/beer_factory/most_popular_root_beer_brand", operation_id="get_most_popular_root_beer_brand", summary="Retrieves the top-ranked root beer brands based on popularity, as determined by the number of root beer entries associated with each brand. The operation returns a specified number of brands, as defined by the input limit parameter.")
async def get_most_popular_root_beer_brand(limit: int = Query(..., description="Number of top brands to retrieve")):
    cursor.execute("SELECT T2.BrandName FROM rootbeer AS T1 INNER JOIN rootbeerbrand AS T2 ON T1.BrandID = T2.BrandID GROUP BY T2.BrandID ORDER BY COUNT(T1.BrandID) LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"brand": []}
    return {"brand": result[0]}

# Endpoint to get the average star rating for the most reviewed root beer brand
@app.get("/v1/beer_factory/average_star_rating_most_reviewed_brand", operation_id="get_average_star_rating_most_reviewed_brand", summary="Retrieves the average star rating for the top-reviewed root beer brands, up to the specified limit. The brands are ranked by the number of reviews they have received, with the most reviewed brand listed first. The limit parameter determines the number of top brands to include in the response.")
async def get_average_star_rating_most_reviewed_brand(limit: int = Query(..., description="Number of top brands to retrieve")):
    cursor.execute("SELECT T1.BrandID, AVG(T1.StarRating) FROM rootbeerreview AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID GROUP BY T3.BrandID ORDER BY COUNT(T1.BrandID) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"brand_id": [], "average_rating": []}
    return {"brand_id": result[0], "average_rating": result[1]}

# Endpoint to get the percentage difference in purchase price between two root beer brands
@app.get("/v1/beer_factory/purchase_price_percentage_difference", operation_id="get_purchase_price_percentage_difference", summary="Retrieve the percentage difference in purchase price between two specified root beer brands. This operation calculates the difference in total purchase price between the two brands and expresses it as a percentage of the total purchase price of the first brand.")
async def get_purchase_price_percentage_difference(brand_name_1: str = Query(..., description="First brand name"), brand_name_2: str = Query(..., description="Second brand name")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T3.BrandName = ? THEN T2.PurchasePrice ELSE 0 END) - SUM(CASE WHEN T3.BrandName = ? THEN T2.PurchasePrice ELSE 0 END)) AS REAL) * 100 / SUM(CASE WHEN T3.BrandName = ? THEN T2.PurchasePrice ELSE 0 END) FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID INNER JOIN rootbeerbrand AS T3 ON T1.BrandID = T3.BrandID", (brand_name_1, brand_name_2, brand_name_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get distinct cities in a given state
@app.get("/v1/beer_factory/distinct_cities_by_state", operation_id="get_distinct_cities_by_state", summary="Retrieves a list of unique cities within the specified state. The state is identified using its abbreviation, such as CA for California.")
async def get_distinct_cities_by_state(state: str = Query(..., description="State abbreviation (e.g., CA for California)")):
    cursor.execute("SELECT DISTINCT City FROM customers WHERE State = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the percentage of customers of a specific gender who are subscribed to the email list
@app.get("/v1/beer_factory/gender_subscription_percentage", operation_id="get_gender_subscription_percentage", summary="Retrieves the percentage of customers of a specified gender who are subscribed to the email list. The operation calculates this percentage by counting the number of customers of the given gender and dividing it by the total number of subscribed customers. The gender and subscription status are provided as input parameters.")
async def get_gender_subscription_percentage(gender: str = Query(..., description="Gender (e.g., F for female)"), subscribed_to_email_list: str = Query(..., description="Subscription status (TRUE or FALSE)")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN Gender = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(SubscribedToEmaillist) FROM customers WHERE SubscribedToEmaillist = ?", (gender, subscribed_to_email_list))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the star rating of a root beer review by a specific customer for a specific brand
@app.get("/v1/beer_factory/star_rating_by_customer_brand", operation_id="get_star_rating_by_customer_brand", summary="Retrieve the star rating of a root beer review submitted by a specific customer for a given brand. This operation requires the customer's first and last name, as well as the brand name of the root beer. The star rating returned corresponds to the review provided by the specified customer for the indicated brand.")
async def get_star_rating_by_customer_brand(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), brand_name: str = Query(..., description="Brand name of the root beer")):
    cursor.execute("SELECT T2.StarRating FROM customers AS T1 INNER JOIN rootbeerreview AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN rootbeerbrand AS T3 ON T2.BrandID = T3.BrandID WHERE T1.First = ? AND T1.Last = ? AND T3.BrandName = ?", (first, last, brand_name))
    result = cursor.fetchone()
    if not result:
        return {"star_rating": []}
    return {"star_rating": result[0]}

# Endpoint to get the brand name of root beers with a specific star rating and review
@app.get("/v1/beer_factory/brand_name_by_star_rating_review", operation_id="get_brand_name_by_star_rating_review", summary="Retrieves the brand name of root beers that have a specified star rating and review. The operation filters root beers based on the provided star rating and review text, and returns the corresponding brand names.")
async def get_brand_name_by_star_rating_review(star_rating: int = Query(..., description="Star rating of the root beer review"), review: str = Query(..., description="Review text of the root beer")):
    cursor.execute("SELECT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T2.BrandID = T1.BrandID WHERE T2.StarRating = ? AND T2.Review = ?", (star_rating, review))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": [row[0] for row in result]}

# Endpoint to get the count of transactions at a specific location with a specific credit card type
@app.get("/v1/beer_factory/transaction_count_by_location_credit_card", operation_id="get_transaction_count_by_location_credit_card", summary="Retrieves the total number of transactions conducted at a specified location using a particular credit card type. This operation provides a quantitative measure of transaction activity, aiding in the analysis of sales patterns and customer behavior.")
async def get_transaction_count_by_location_credit_card(location_name: str = Query(..., description="Name of the location"), credit_card_type: str = Query(..., description="Type of credit card used in the transaction")):
    cursor.execute("SELECT COUNT(T1.TransactionID) FROM `transaction` AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T2.LocationName = ? AND T1.CreditCardType = ?", (location_name, credit_card_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct brand names of root beers with specific ingredients, star rating, and review date pattern
@app.get("/v1/beer_factory/distinct_brand_names_by_ingredients_rating_date", operation_id="get_distinct_brand_names_by_ingredients_rating_date", summary="Retrieve a unique list of brand names for root beers that meet specific ingredient, star rating, and review date criteria. The ingredients include cane sugar and honey, which can be set to TRUE or FALSE. The star rating represents the quality of the root beer review. The review date pattern follows the 'YYYY%' format.")
async def get_distinct_brand_names_by_ingredients_rating_date(cane_sugar: str = Query(..., description="Whether the root beer contains cane sugar (TRUE or FALSE)"), honey: str = Query(..., description="Whether the root beer contains honey (TRUE or FALSE)"), star_rating: int = Query(..., description="Star rating of the root beer review"), review_date_pattern: str = Query(..., description="Pattern for the review date in 'YYYY%' format")):
    cursor.execute("SELECT DISTINCT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T1.CaneSugar = ? AND T1.Honey = ? AND T2.StarRating = ? AND T2.ReviewDate LIKE ?", (cane_sugar, honey, star_rating, review_date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": [row[0] for row in result]}

# Endpoint to get distinct geolocations of transactions by a specific customer within a specific date pattern
@app.get("/v1/beer_factory/distinct_geolocations_by_customer_date", operation_id="get_distinct_geolocations_by_customer_date", summary="Retrieve unique geographical locations where a specific customer has made transactions within a given year. The customer is identified by their first and last name, and the year is specified using a date pattern.")
async def get_distinct_geolocations_by_customer_date(first: str = Query(..., description="First name of the customer"), last: str = Query(..., description="Last name of the customer"), transaction_date_pattern: str = Query(..., description="Pattern for the transaction date in 'YYYY%' format")):
    cursor.execute("SELECT DISTINCT T1.Latitude, T1.Longitude FROM geolocation AS T1 INNER JOIN `transaction` AS T2 ON T2.LocationID = T1.LocationID INNER JOIN customers AS T3 ON T3.CustomerID = T2.CustomerID WHERE T3.First = ? AND T3.Last = ? AND T2.TransactionDate LIKE ?", (first, last, transaction_date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"geolocations": []}
    return {"geolocations": [{"latitude": row[0], "longitude": row[1]} for row in result]}

# Endpoint to get the email of a customer by transaction ID
@app.get("/v1/beer_factory/customer_email_by_transaction_id", operation_id="get_customer_email_by_transaction_id", summary="Retrieves the email address of the customer associated with the provided transaction ID. This operation uses the transaction ID to look up the corresponding customer record and returns the customer's email address.")
async def get_customer_email_by_transaction_id(transaction_id: str = Query(..., description="Transaction ID")):
    cursor.execute("SELECT T1.Email FROM customers AS T1 INNER JOIN `transaction` AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.TransactionID = ?", (transaction_id,))
    result = cursor.fetchone()
    if not result:
        return {"email": []}
    return {"email": result[0]}

# Endpoint to get the count of root beers by container type and credit card type
@app.get("/v1/beer_factory/root_beer_count_by_container_credit_card", operation_id="get_root_beer_count_by_container_credit_card", summary="Retrieves the total count of root beers categorized by their container type and the type of credit card used in the transaction. This operation provides a summary of root beer sales data, enabling analysis based on container and payment preferences.")
async def get_root_beer_count_by_container_credit_card(container_type: str = Query(..., description="Container type of the root beer"), credit_card_type: str = Query(..., description="Type of credit card used in the transaction")):
    cursor.execute("SELECT COUNT(T1.RootBeerID) FROM rootbeer AS T1 INNER JOIN `transaction` AS T2 ON T1.RootBeerID = T2.RootBeerID WHERE T1.ContainerType = ? AND T2.CreditCardType = ?", (container_type, credit_card_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the location with the highest count of a specific container type of root beer
@app.get("/v1/beer_factory/top_location_by_container_type", operation_id="get_top_location_by_container_type", summary="Retrieves the location with the highest count of a specified container type of root beer. The operation considers the provided container type and returns the name of the location with the most occurrences of that container type.")
async def get_top_location_by_container_type(container_type: str = Query(..., description="Container type of the root beer")):
    cursor.execute("SELECT T2.LocationName FROM rootbeer AS T1 INNER JOIN location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.ContainerType = ? GROUP BY T2.LocationID ORDER BY COUNT(T1.LocationID) DESC LIMIT 1", (container_type,))
    result = cursor.fetchone()
    if not result:
        return {"location_name": []}
    return {"location_name": result[0]}

# Endpoint to get distinct brand names of root beers with a specific star rating, limited by a specified number
@app.get("/v1/beer_factory/distinct_brand_names_by_star_rating_limit", operation_id="get_distinct_brand_names_by_star_rating_limit", summary="Retrieves a distinct set of brand names for root beers that have a specified star rating, with the number of results limited to a specified value. This operation is useful for obtaining a unique list of brands that meet a certain quality standard, as determined by the star rating, and for controlling the size of the result set.")
async def get_distinct_brand_names_by_star_rating_limit(star_rating: int = Query(..., description="Star rating of the root beer review"), limit: int = Query(..., description="Limit on the number of results returned")):
    cursor.execute("SELECT DISTINCT T1.BrandName FROM rootbeerbrand AS T1 INNER JOIN rootbeerreview AS T2 ON T1.BrandID = T2.BrandID WHERE T2.StarRating = ? LIMIT ?", (star_rating, limit))
    result = cursor.fetchall()
    if not result:
        return {"brand_names": []}
    return {"brand_names": [row[0] for row in result]}

api_calls = [
    "/v1/beer_factory/first_brewed_brand",
    "/v1/beer_factory/count_brands_by_country?country=United%20States",
    "/v1/beer_factory/brands_with_facebook",
    "/v1/beer_factory/brand_with_highest_profit_margin",
    "/v1/beer_factory/brand_description?brand_name=A%26W",
    "/v1/beer_factory/brewery_city?brewery_name=AJ%20Stephans%20Beverages",
    "/v1/beer_factory/count_transactions_by_customer_month?first_name=Frank-Paul&last_name=Santangelo&transaction_month=2014-07",
    "/v1/beer_factory/count_transactions_by_gender_month?gender=M&transaction_month=2014-07",
    "/v1/beer_factory/count_transactions_by_subscription_credit_card_month?subscribed_to_email_list=TRUE&credit_card_type=Visa&transaction_month=2014-07",
    "/v1/beer_factory/customer_with_most_transactions_month?transaction_month=2014-08",
    "/v1/beer_factory/distinct_brand_names?first=Frank-Paul&last=Santangelo&transaction_date=2014-07-07",
    "/v1/beer_factory/customer_count_by_container_type?first=Frank-Paul&last=Santangelo&transaction_date=2014-07-07&container_type=Can",
    "/v1/beer_factory/brand_id_count_by_date_pattern?date_pattern=2014-08%&brand_name=Bulldog",
    "/v1/beer_factory/customer_names_by_brewery?brewery_name=AJ%20Stephans%20Beverages",
    "/v1/beer_factory/brand_id_count_by_date_pattern_null_twitter?date_pattern=2014-08%",
    "/v1/beer_factory/distinct_credit_card_numbers?first=Frank-Paul&last=Santangelo&transaction_date=2014-07-07",
    "/v1/beer_factory/customer_count_by_sweetener_honey?first=Frank-Paul&last=Santangelo&artificial_sweetener=FALSE&honey=FALSE",
    "/v1/beer_factory/transaction_dates_by_gender_min_count?gender=M&min_count=3",
    "/v1/beer_factory/average_brand_id_count_by_date_pattern?date_pattern=2014-08%&brand_name=A%26W",
    "/v1/beer_factory/percentage_brand_id_by_brewery_date_pattern?brewery_name=AJ%20Stephans%20Beverages&date_pattern=2014%",
    "/v1/beer_factory/customer_review_count?first=James&last=House",
    "/v1/beer_factory/location_coordinates?location_name=Sac%20State%20American%20River%20Courtyard",
    "/v1/beer_factory/location_name_by_transaction_id?transaction_id=100885",
    "/v1/beer_factory/city_by_transaction_id?transaction_id=103545",
    "/v1/beer_factory/distinct_phone_numbers?credit_card_number=6011179359005382",
    "/v1/beer_factory/top_reviewer",
    "/v1/beer_factory/first_purchase_date_by_review?review=Tastes%20like%20Australia.",
    "/v1/beer_factory/earliest_transaction_date?first=Natalie&last=Dorris",
    "/v1/beer_factory/top_brewery_by_star_rating?star_rating=5",
    "/v1/beer_factory/subscription_status_review?star_rating=3&brand_name=Frostie&review_date=2014-04-24",
    "/v1/beer_factory/price_difference_review?review=The%20quintessential%20dessert%20root%20beer.%20No%20ice%20cream%20required.",
    "/v1/beer_factory/percentage_five_star_ratings?brand_name=River%20City",
    "/v1/beer_factory/average_reviews_per_brand?state=CA",
    "/v1/beer_factory/customer_count_gender_subscription?gender=F&subscribed_to_email_list=TRUE",
    "/v1/beer_factory/most_recent_root_beer",
    "/v1/beer_factory/first_ten_customers",
    "/v1/beer_factory/brewery_count_country?country=Australia",
    "/v1/beer_factory/customer_count_name_city?first_name=Charles&city=Sacramento",
    "/v1/beer_factory/transaction_count_credit_card_date?credit_card_type=MasterCard&transaction_date_pattern=2014%",
    "/v1/beer_factory/brand_names_by_customer_rating?first=Jayne&last=Collins&star_rating=1",
    "/v1/beer_factory/count_brands_by_location_attributes?location_name=Sac%20State%20American%20River%20Courtyard&purchase_date_pattern=2015%25&honey=TRUE&cane_sugar=FALSE&container_type=Bottle",
    "/v1/beer_factory/top_brewery_by_purchase_dates?start_date=2016-01-01&end_date=2016-12-31",
    "/v1/beer_factory/customer_names_by_brand_rating?brand_name=River%20City&star_rating=5",
    "/v1/beer_factory/count_transactions_by_customer_date_range?first=Tom&last=Hanks&start_date=2015-01-01&end_date=2016-12-31",
    "/v1/beer_factory/brand_names_by_rating?star_rating=5",
    "/v1/beer_factory/count_customers_by_brand?first=Nicholas&last=Sparks&brand_name_pattern=Henry%20Weinhard%25s",
    "/v1/beer_factory/top_brewery_without_social_media",
    "/v1/beer_factory/top_location_by_brand?brand_name=Dog%20n%20Suds&location_name_1=Sac%20State%20American%20River%20Courtyard&location_name_2=Sac%20State%20Union",
    "/v1/beer_factory/count_brands_by_attributes?container_type=Can&brand_name=A%26W&purchase_date_pattern=2016%25",
    "/v1/beer_factory/brand_names_by_star_rating?star_rating=5&min_reviews=5",
    "/v1/beer_factory/brand_purchase_ratio?start_date=2014-01-01&end_date=2016-12-31&brewery_name=Dr%20Pepper%20Snapple%20Group",
    "/v1/beer_factory/top_brand_customer",
    "/v1/beer_factory/customer_details?gender=M&city=Fair%20Oaks&subscribed_to_email_list=TRUE",
    "/v1/beer_factory/container_type_percentage?container_type=Can&year=2014",
    "/v1/beer_factory/brands_by_year_range?start_year=1996&end_year=2000",
    "/v1/beer_factory/top_brand_by_star_rating?star_rating=1",
    "/v1/beer_factory/transaction_percentage_by_credit_card?credit_card_type=Visa",
    "/v1/beer_factory/brand_count_by_attributes?corn_syrup=TRUE&artificial_sweetener=TRUE&available_in_cans=TRUE",
    "/v1/beer_factory/transaction_percentage_by_location?location_name=Sac%20State%20American%20River%20Courtyard",
    "/v1/beer_factory/average_rootbeer_purchases_per_day?caffeinated=TRUE",
    "/v1/beer_factory/most_and_least_profitable_rootbeers",
    "/v1/beer_factory/average_purchase_price_by_container_type?container_type=Bottle&min_purchase_price=2",
    "/v1/beer_factory/count_brands_by_geolocation_and_container_type?latitude=38.559615&longitude=-121.42243&container_type=Bottle",
    "/v1/beer_factory/percentage_customers_by_star_rating_and_subscription?min_star_rating=3&subscribed_to_email_list=FALSE",
    "/v1/beer_factory/top_brand_by_star_rating_and_year?star_rating=5&year=2012",
    "/v1/beer_factory/count_customers_by_gender_and_sweetener?gender=F&artificial_sweetener=TRUE",
    "/v1/beer_factory/difference_in_sweetener_count?cane_sugar=TRUE&corn_syrup=TRUE",
    "/v1/beer_factory/top_brewery_by_year?year=2015",
    "/v1/beer_factory/customer_brand_preference_percentage?brand_name=Dominion&city=Sacramento&gender=M&transaction_date_pattern=2014%",
    "/v1/beer_factory/transaction_honey_difference?honey_true=TRUE&honey_false=FALSE",
    "/v1/beer_factory/customer_details_by_location_and_credit_card?location_name=Sac%20State%20Union&credit_card_type=American%20Express",
    "/v1/beer_factory/brand_names_by_star_rating_and_review_date?star_rating=5&start_date=2014-09-01&end_date=2014-09-30",
    "/v1/beer_factory/geolocations_by_credit_card_type?credit_card_type=American%20Express",
    "/v1/beer_factory/customer_count_by_city_and_credit_card?city=Folsom&credit_card_type=Visa",
    "/v1/beer_factory/cities_by_star_rating_and_review_date?star_rating=5&start_date=2012-11-01&end_date=2012-11-30",
    "/v1/beer_factory/brand_names_by_customer_name?first_name=Peg&last_name=Winchester",
    "/v1/beer_factory/most_reviewed_brand_by_star_rating?star_rating=1",
    "/v1/beer_factory/most_used_credit_card_type_for_non_alcoholic?alcoholic=FALSE",
    "/v1/beer_factory/customer_names_by_star_rating?star_rating=5",
    "/v1/beer_factory/most_frequent_latitude_by_brand?brand_name=Thomas%20Kemper",
    "/v1/beer_factory/most_common_star_rating_by_corn_syrup?corn_syrup=TRUE",
    "/v1/beer_factory/geolocation_by_zip_code?zip_code=95819",
    "/v1/beer_factory/brand_names_by_latitude_longitude?latitude=38.566129&longitude=-121.426432",
    "/v1/beer_factory/average_profit_margin_by_container_type?container_type=Can",
    "/v1/beer_factory/percentage_reviews_by_star_rating_credit_card?star_rating=3&credit_card_type=Discover",
    "/v1/beer_factory/brand_ids_above_star_rating?min_star_rating=3",
    "/v1/beer_factory/count_brand_ids_by_container_type_date_range?container_type=Bottle&start_date=2015-04-03&end_date=2015-10-26",
    "/v1/beer_factory/customer_names_by_star_rating_review?star_rating=5&review=The%20quintessential%20dessert%20root%20beer.%20No%20ice%20cream%20required.",
    "/v1/beer_factory/customer_contact_info?min_star_rating=3&city=Sacramento&start_date=2014-01-01&end_date=2014-12-31",
    "/v1/beer_factory/customer_count_by_criteria?star_rating=4&city=Sacramento&gender=F&subscribed_to_email_list=TRUE&start_date=2013-01-03&end_date=2013-10-26",
    "/v1/beer_factory/brewery_brand_info?purchase_date=2015-06-06&container_type=Can",
    "/v1/beer_factory/oldest_brewed_brand?first_brewed_year=1930-01-01&container_type=Bottle",
    "/v1/beer_factory/transaction_count_by_customer?first_name=Anna&last_name=Himes&credit_card_type=MasterCard&start_date=2014-12-25&end_date=2016-05-20",
    "/v1/beer_factory/average_star_rating?brand_id=10018&gender=F&start_date=2013-01-25&end_date=2015-03-10",
    "/v1/beer_factory/brand_name_by_review?customer_id=331115&review=Yuk%2C%20more%20like%20licorice%20soda.&star_rating=1",
    "/v1/beer_factory/total_purchase_price?location_name=Sac%20State%20American%20River%20Courtyard&credit_card_type=Visa&start_date=2014-06-03&end_date=2015-11-27",
    "/v1/beer_factory/transaction_count_by_location?location_name=Sac%20State%20Union&credit_card_type=American%20Express&start_date=2014-01-01&end_date=2014-12-31",
    "/v1/beer_factory/review_by_brand_and_date?brand_name=Bulldog&review_date=2013-07-26",
    "/v1/beer_factory/brand_price_difference?star_rating=5&start_date=2013-01-01&end_date=2013-12-31",
    "/v1/beer_factory/customer_time_ago",
    "/v1/beer_factory/credit_card_types?first_name=Kenneth&last_name=Walton",
    "/v1/beer_factory/root_beer_details?root_beer_id=100054",
    "/v1/beer_factory/root_beer_ids_by_customer?first_name_1=Tim&last_name_1=Ocel&first_name_2=Dawn&last_name_2=Childress",
    "/v1/beer_factory/root_beer_ids_by_brand?brand_name_1=Bulldog&brand_name_2=Bundaberg&brand_name_3=Dad's&brand_name_4=Dog%20n%20Suds&brand_name_5=Virgil's",
    "/v1/beer_factory/container_type_count?container_type=Bottle&first_name=Jim&last_name=Breech",
    "/v1/beer_factory/root_beer_count_by_state?state=CA",
    "/v1/beer_factory/brand_id_count_difference?state_1=LA&container_type=Bottle&state_2=MO",
    "/v1/beer_factory/transaction_percentage_comparison?location_name_1=Sac%20State%20American%20River%20Courtyard&location_name_2=Sac%20State%20Union",
    "/v1/beer_factory/top_customers_by_purchase_price?limit=10",
    "/v1/beer_factory/total_purchase_price_by_brand?cane_sugar=FALSE&caffeinated=FALSE",
    "/v1/beer_factory/most_popular_root_beer_brand?limit=1",
    "/v1/beer_factory/average_star_rating_most_reviewed_brand?limit=1",
    "/v1/beer_factory/purchase_price_percentage_difference?brand_name_1=River%20City&brand_name_2=Frostie",
    "/v1/beer_factory/distinct_cities_by_state?state=CA",
    "/v1/beer_factory/gender_subscription_percentage?gender=F&subscribed_to_email_list=TRUE",
    "/v1/beer_factory/star_rating_by_customer_brand?first=Urijah&last=Faber&brand_name=Frostie",
    "/v1/beer_factory/brand_name_by_star_rating_review?star_rating=1&review=Too%20Spicy!",
    "/v1/beer_factory/transaction_count_by_location_credit_card?location_name=Sac%20State%20American%20River%20Courtyard&credit_card_type=MasterCard",
    "/v1/beer_factory/distinct_brand_names_by_ingredients_rating_date?cane_sugar=TRUE&honey=TRUE&star_rating=1&review_date_pattern=2012%",
    "/v1/beer_factory/distinct_geolocations_by_customer_date?first=Tommy&last=Kono&transaction_date_pattern=2014%",
    "/v1/beer_factory/customer_email_by_transaction_id?transaction_id=100016",
    "/v1/beer_factory/root_beer_count_by_container_credit_card?container_type=Bottle&credit_card_type=American%20Express",
    "/v1/beer_factory/top_location_by_container_type?container_type=Bottle",
    "/v1/beer_factory/distinct_brand_names_by_star_rating_limit?star_rating=5&limit=3"
]
