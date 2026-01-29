from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/language_corpus/language_corpus.sqlite')
cursor = conn.cursor()

# Endpoint to get the title and words of the page with the longest title
@app.get("/v1/language_corpus/longest_title_page", operation_id="get_longest_title_page", summary="Retrieves the title and word count of the page with the longest title from the corpus. This operation provides a quick overview of the page with the most extensive title, allowing users to identify the most verbose entry in the corpus.")
async def get_longest_title_page():
    cursor.execute("SELECT title, words FROM pages WHERE title = ( SELECT MAX(LENGTH(title)) FROM pages )")
    result = cursor.fetchone()
    if not result:
        return {"title": [], "words": []}
    return {"title": result[0], "words": result[1]}

# Endpoint to get titles of pages with fewer words than a specified number
@app.get("/v1/language_corpus/titles_fewer_words", operation_id="get_titles_fewer_words", summary="Retrieves the titles of pages that contain fewer words than the specified maximum limit. This operation is useful for filtering pages based on their word count and obtaining a concise list of titles that meet the criteria.")
async def get_titles_fewer_words(max_words: int = Query(..., description="Maximum number of words")):
    cursor.execute("SELECT title FROM pages WHERE words < ?", (max_words,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get pages with titles containing a specific substring
@app.get("/v1/language_corpus/pages_with_substring", operation_id="get_pages_with_substring", summary="Retrieves pages whose titles contain a specified substring. The substring can be matched anywhere within the title. This operation is useful for finding pages based on partial title matches.")
async def get_pages_with_substring(substring: str = Query(..., description="Substring to search in titles")):
    cursor.execute("SELECT page FROM pages WHERE title LIKE ? OR title LIKE ? OR title LIKE ?", (substring + '%', '%' + substring + '%', '%' + substring))
    result = cursor.fetchall()
    if not result:
        return {"pages": []}
    return {"pages": [row[0] for row in result]}

# Endpoint to get the title of a page with a specific revision number
@app.get("/v1/language_corpus/title_by_revision", operation_id="get_title_by_revision", summary="Retrieves the title of a page associated with a specific revision number. The operation returns the title of the page that corresponds to the provided revision number.")
async def get_title_by_revision(revision: int = Query(..., description="Revision number")):
    cursor.execute("SELECT title FROM pages WHERE revision = ?", (revision,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get titles of pages with revision numbers within a specified range
@app.get("/v1/language_corpus/titles_by_revision_range", operation_id="get_titles_by_revision_range", summary="Retrieves the titles of pages that have revision numbers within the specified range. The range is defined by a minimum and maximum revision number, which are provided as input parameters. This operation returns a list of titles that fall within this range.")
async def get_titles_by_revision_range(min_revision: int = Query(..., description="Minimum revision number"), max_revision: int = Query(..., description="Maximum revision number")):
    cursor.execute("SELECT title FROM pages WHERE revision BETWEEN ? AND ?", (min_revision, max_revision))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of pages with word count within a specified range
@app.get("/v1/language_corpus/count_pages_by_word_range", operation_id="get_count_pages_by_word_range", summary="Retrieves the total number of pages that have a word count within the specified range. The range is defined by the minimum and maximum number of words, which are provided as input parameters.")
async def get_count_pages_by_word_range(min_words: int = Query(..., description="Minimum number of words"), max_words: int = Query(..., description="Maximum number of words")):
    cursor.execute("SELECT COUNT(pid) FROM pages WHERE words BETWEEN ? AND ?", (min_words, max_words))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get page IDs containing a specific word
@app.get("/v1/language_corpus/page_ids_by_word", operation_id="get_page_ids_by_word", summary="Retrieves the unique identifiers of pages that contain a specified word. The search is case-insensitive and matches the exact word, not partial words or phrases. The response includes a list of page IDs that meet the search criteria.")
async def get_page_ids_by_word(word: str = Query(..., description="Word to search for")):
    cursor.execute("SELECT T2.pid FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid WHERE T1.word = ?", (word,))
    result = cursor.fetchall()
    if not result:
        return {"page_ids": []}
    return {"page_ids": [row[0] for row in result]}

# Endpoint to get the word with the maximum occurrences in pages
@app.get("/v1/language_corpus/word_max_occurrences", operation_id="get_word_max_occurrences", summary="Retrieves the word that appears most frequently across all pages in the corpus. This operation identifies the word with the highest occurrence count and returns its details.")
async def get_word_max_occurrences():
    cursor.execute("SELECT T1.word FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid WHERE T2.occurrences = ( SELECT MAX(occurrences) FROM pages_words )")
    result = cursor.fetchone()
    if not result:
        return {"word": []}
    return {"word": result[0]}

# Endpoint to get words that form a bigram with a specific word
@app.get("/v1/language_corpus/bigram_words", operation_id="get_bigram_words", summary="Retrieves all words that form a bigram with the provided word. A bigram is a pair of consecutive words in a sentence. This operation identifies the first word in a bigram pair where the second word matches the input word, and returns the identified first words.")
async def get_bigram_words(word: str = Query(..., description="Word to form a bigram with")):
    cursor.execute("SELECT T1.word FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st WHERE T2.w2nd = ( SELECT wid FROM words WHERE word = ? )", (word,))
    result = cursor.fetchall()
    if not result:
        return {"words": []}
    return {"words": [row[0] for row in result]}

# Endpoint to get titles and occurrences of a specific word in pages
@app.get("/v1/language_corpus/titles_occurrences_by_word", operation_id="get_titles_occurrences_by_word", summary="Retrieves a list of titles and the number of occurrences of a specified word within those titles. The word is provided as an input parameter.")
async def get_titles_occurrences_by_word(word: str = Query(..., description="Word to search for")):
    cursor.execute("SELECT T1.title, T2.occurrences FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid INNER JOIN words AS T3 ON T2.wid = T3.wid WHERE T3.word = ?", (word,))
    result = cursor.fetchall()
    if not result:
        return {"titles_occurrences": []}
    return {"titles_occurrences": [{"title": row[0], "occurrences": row[1]} for row in result]}

# Endpoint to get the average occurrences of a specific word in biwords
@app.get("/v1/language_corpus/average_occurrences_by_word", operation_id="get_average_occurrences_by_word", summary="Retrieves the average number of occurrences of a specific word in biwords. The input parameter is used to specify the word for which the average occurrences are calculated.")
async def get_average_occurrences_by_word(word: str = Query(..., description="The word to find the average occurrences for")):
    cursor.execute("SELECT AVG(T2.occurrences) FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st WHERE T2.w1st = ( SELECT wid FROM words WHERE word = ? )", (word,))
    result = cursor.fetchone()
    if not result:
        return {"average_occurrences": []}
    return {"average_occurrences": result[0]}

# Endpoint to get the number of pages for a specific language
@app.get("/v1/language_corpus/pages_by_language", operation_id="get_pages_by_language", summary="Retrieves the total number of pages associated with a specific language. The language is identified using a language code provided as an input parameter.")
async def get_pages_by_language(lang: str = Query(..., description="The language code to find the number of pages for")):
    cursor.execute("SELECT pages FROM langs WHERE lang = ?", (lang,))
    result = cursor.fetchone()
    if not result:
        return {"pages": []}
    return {"pages": result[0]}

# Endpoint to get the biwords with the maximum occurrences
@app.get("/v1/language_corpus/biwords_with_max_occurrences", operation_id="get_biwords_with_max_occurrences", summary="Retrieves the bigrams (two-word sequences) that appear most frequently in the corpus. The operation returns the first and second words of the most common bigrams, providing insights into the most recurring two-word combinations in the language corpus.")
async def get_biwords_with_max_occurrences():
    cursor.execute("SELECT w1st, w2nd FROM biwords WHERE occurrences = ( SELECT MAX(occurrences) FROM biwords )")
    result = cursor.fetchall()
    if not result:
        return {"biwords": []}
    return {"biwords": result}

# Endpoint to get word IDs with occurrences less than or equal to a specified number
@app.get("/v1/language_corpus/word_ids_by_occurrences", operation_id="get_word_ids_by_occurrences", summary="Retrieves word IDs from the language corpus for words that occur less than or equal to a specified number of times. The maximum number of occurrences can be set to filter the word IDs returned.")
async def get_word_ids_by_occurrences(max_occurrences: int = Query(..., description="The maximum number of occurrences to filter word IDs")):
    cursor.execute("SELECT wid FROM langs_words WHERE occurrences <= ?", (max_occurrences,))
    result = cursor.fetchall()
    if not result:
        return {"word_ids": []}
    return {"word_ids": [row[0] for row in result]}

# Endpoint to get the title of the page with the maximum number of words
@app.get("/v1/language_corpus/page_title_with_max_words", operation_id="get_page_title_with_max_words", summary="Retrieves the title of the page containing the highest number of words from the corpus.")
async def get_page_title_with_max_words():
    cursor.execute("SELECT title FROM pages WHERE words = ( SELECT MAX(words) FROM pages )")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the page content by title
@app.get("/v1/language_corpus/page_content_by_title", operation_id="get_page_content_by_title", summary="Retrieves the full content of a specific page based on its title. The operation searches for a page with the provided title and returns its content.")
async def get_page_content_by_title(title: str = Query(..., description="The title of the page to retrieve the content")):
    cursor.execute("SELECT page FROM pages WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"page_content": []}
    return {"page_content": result[0]}

# Endpoint to get the word ID and occurrences for the most frequent word in a specific page
@app.get("/v1/language_corpus/most_frequent_word_in_page", operation_id="get_most_frequent_word_in_page", summary="Retrieves the ID and occurrences of the most frequently appearing word in a specified page. The page is identified by its title.")
async def get_most_frequent_word_in_page(title: str = Query(..., description="The title of the page to find the most frequent word")):
    cursor.execute("SELECT T2.wid, T2.occurrences FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T1.title = ? ORDER BY T2.occurrences DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"word_id": [], "occurrences": []}
    return {"word_id": result[0], "occurrences": result[1]}

# Endpoint to get the titles of pages ordered by the number of words with a specified limit
@app.get("/v1/language_corpus/page_titles_ordered_by_words", operation_id="get_page_titles_ordered_by_words", summary="Retrieve a specified number of page titles, sorted by the total word count in each page. The 'limit' parameter determines the maximum number of titles to return.")
async def get_page_titles_ordered_by_words(limit: int = Query(..., description="The limit of the number of titles to return")):
    cursor.execute("SELECT T1.title FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid ORDER BY T1.words LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the total occurrences of a specific biword pair
@app.get("/v1/language_corpus/total_occurrences_of_biword_pair", operation_id="get_total_occurrences_of_biword_pair", summary="Retrieves the total number of occurrences of a specific biword pair, identified by the provided first and second word IDs.")
async def get_total_occurrences_of_biword_pair(w1st: int = Query(..., description="The first word ID of the biword pair"), w2nd: int = Query(..., description="The second word ID of the biword pair")):
    cursor.execute("SELECT SUM(occurrences) FROM biwords WHERE w1st = ? AND w2nd = ?", (w1st, w2nd))
    result = cursor.fetchone()
    if not result:
        return {"total_occurrences": []}
    return {"total_occurrences": result[0]}

# Endpoint to get the word pairs with a specific number of occurrences
@app.get("/v1/language_corpus/word_pairs_by_occurrences", operation_id="get_word_pairs_by_occurrences", summary="Retrieve word pairs that appear together a specified number of times in the corpus. The operation filters word pairs based on the provided occurrence count, returning the words that form each pair.")
async def get_word_pairs_by_occurrences(occurrences: int = Query(..., description="The number of occurrences to filter word pairs")):
    cursor.execute("SELECT T1.word, T3.word FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st INNER JOIN words AS T3 ON T3.wid = T2.w2nd WHERE T2.occurrences = ?", (occurrences,))
    result = cursor.fetchall()
    if not result:
        return {"word_pairs": []}
    return {"word_pairs": result}

# Endpoint to get the most frequent biword pair for a given page title
@app.get("/v1/language_corpus/most_frequent_biword_pair", operation_id="get_most_frequent_biword_pair", summary="Retrieves the most frequently occurring pair of words (biword pair) for a specified page. The page is identified by its title. The response includes the two words in the pair and the number of times they appear together on the page.")
async def get_most_frequent_biword_pair(title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT T3.w1st, T3.w2nd, T3.occurrences FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid INNER JOIN biwords AS T3 ON T2.wid = T3.w1st OR T2.wid = T3.w2nd WHERE T1.title = ? ORDER BY T3.occurrences DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"biword_pair": []}
    return {"biword_pair": {"w1st": result[0], "w2nd": result[1], "occurrences": result[2]}}

# Endpoint to get the total number of words for pages with specific titles
@app.get("/v1/language_corpus/total_words_for_titles", operation_id="get_total_words_for_titles", summary="Retrieves the total count of words from pages that match the provided titles. The operation accepts two titles as input parameters, which are used to filter the pages and calculate the sum of words.")
async def get_total_words_for_titles(title1: str = Query(..., description="First title of the page"), title2: str = Query(..., description="Second title of the page")):
    cursor.execute("SELECT SUM(words) FROM pages WHERE title IN (?, ?)", (title1, title2))
    result = cursor.fetchone()
    if not result:
        return {"total_words": []}
    return {"total_words": result[0]}

# Endpoint to get the revision of a page by its title
@app.get("/v1/language_corpus/page_revision_by_title", operation_id="get_page_revision_by_title", summary="Get the revision of a page by its title")
async def get_page_revision_by_title(title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT revision FROM pages WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"revision": []}
    return {"revision": result[0]}

# Endpoint to get the average number of words for pages with a minimum word count
@app.get("/v1/language_corpus/average_words_above_threshold", operation_id="get_average_words_above_threshold", summary="Retrieves the average number of words from pages that contain at least a specified minimum number of words. This operation calculates the total number of words from qualifying pages and divides it by the count of those pages to provide a meaningful average.")
async def get_average_words_above_threshold(min_words: int = Query(..., description="Minimum number of words in a page")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN words >= ? THEN words ELSE 0 END) AS REAL) / SUM(CASE WHEN words >= ? THEN 1 ELSE 0 END) FROM pages", (min_words, min_words))
    result = cursor.fetchone()
    if not result:
        return {"average_words": []}
    return {"average_words": result[0]}

# Endpoint to get the revisions of pages with fewer than a specified number of words
@app.get("/v1/language_corpus/revisions_below_word_count", operation_id="get_revisions_below_word_count", summary="Retrieves a limited number of revisions for pages containing fewer than the specified number of words. This operation is useful for identifying and analyzing shorter pages that may require further editing or expansion.")
async def get_revisions_below_word_count(max_words: int = Query(..., description="Maximum number of words in a page"), limit: int = Query(..., description="Number of results to return")):
    cursor.execute("SELECT revision FROM pages WHERE words < ? LIMIT ?", (max_words, limit))
    results = cursor.fetchall()
    if not results:
        return {"revisions": []}
    return {"revisions": [result[0] for result in results]}

# Endpoint to get page IDs for titles starting with a specific prefix
@app.get("/v1/language_corpus/page_ids_by_title_prefix", operation_id="get_page_ids_by_title_prefix", summary="Retrieves the unique identifiers (page IDs) of pages whose titles begin with the specified prefix. This operation allows for efficient filtering of pages based on the initial characters of their titles.")
async def get_page_ids_by_title_prefix(prefix: str = Query(..., description="Prefix of the title")):
    cursor.execute("SELECT pid FROM pages WHERE title LIKE ?", (prefix + '%',))
    results = cursor.fetchall()
    if not results:
        return {"page_ids": []}
    return {"page_ids": [result[0] for result in results]}

# Endpoint to get page titles containing a specific word
@app.get("/v1/language_corpus/page_titles_by_word", operation_id="get_page_titles_by_word", summary="Retrieves a list of page titles that contain a specified word. The operation searches for the word in the page titles and returns the matching titles. The input parameter is used to specify the word to search for.")
async def get_page_titles_by_word(word: str = Query(..., description="Word to search for in page titles")):
    cursor.execute("SELECT T1.title FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid INNER JOIN words AS T3 ON T2.wid = T3.wid WHERE T3.word = ?", (word,))
    results = cursor.fetchall()
    if not results:
        return {"titles": []}
    return {"titles": [result[0] for result in results]}

# Endpoint to get word IDs for a specific page title
@app.get("/v1/language_corpus/word_ids_by_page_title", operation_id="get_word_ids_by_page_title", summary="Retrieves the unique identifiers of all words associated with a specific page, based on the provided page title. This operation allows you to obtain a comprehensive list of word IDs for a given page, enabling further analysis or processing of the page's content.")
async def get_word_ids_by_page_title(title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT T2.wid FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T1.title = ?", (title,))
    results = cursor.fetchall()
    if not results:
        return {"word_ids": []}
    return {"word_ids": [result[0] for result in results]}

# Endpoint to check if a specific word ID exists in a page with a given title
@app.get("/v1/language_corpus/check_word_id_in_page", operation_id="check_word_id_in_page", summary="This operation verifies the presence of a specific word within a page identified by its title. It returns 'YES' if the word is found in the page, and 'NO' otherwise. The word is identified by its unique ID, and the page is determined by its title.")
async def check_word_id_in_page(word_id: int = Query(..., description="Word ID to check"), title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT CASE WHEN COUNT(T1.pid) > 0 THEN 'YES' ELSE 'NO' END AS YORN FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T2.wid = ? AND T1.title = ?", (word_id, title))
    result = cursor.fetchone()
    if not result:
        return {"exists": []}
    return {"exists": result[0]}

# Endpoint to get the occurrences of a specific word in a page with a given title
@app.get("/v1/language_corpus/word_occurrences_in_page", operation_id="get_word_occurrences_in_page", summary="Get the occurrences of a specific word in a page with a given title")
async def get_word_occurrences_in_page(word: str = Query(..., description="Word to search for"), title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT T2.occurrences FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ? AND T3.title = ?", (word, title))
    result = cursor.fetchone()
    if not result:
        return {"occurrences": []}
    return {"occurrences": result[0]}

# Endpoint to get the sum of occurrences for a given word
@app.get("/v1/language_corpus/sum_occurrences_by_word", operation_id="get_sum_occurrences_by_word", summary="Get the sum of occurrences for a given word")
async def get_sum_occurrences_by_word(word: str = Query(..., description="Word to filter by")):
    cursor.execute("SELECT SUM(T2.occurrences) FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st OR T1.wid = T2.w2nd WHERE T2.w1st IN (SELECT wid FROM words WHERE word = ?) OR T2.w2nd IN (SELECT wid FROM words WHERE word = ?)", (word, word))
    result = cursor.fetchone()
    if not result:
        return {"sum_occurrences": []}
    return {"sum_occurrences": result[0]}

# Endpoint to get the second word in biwords based on the first word and limit
@app.get("/v1/language_corpus/second_word_by_first_word", operation_id="get_second_word_by_first_word", summary="Retrieves the second word from a list of biwords that match the provided first word, limiting the results to the specified number. This operation is useful for finding common second words associated with a given first word in a corpus of language data.")
async def get_second_word_by_first_word(word: str = Query(..., description="First word to filter by"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT w2nd FROM biwords WHERE w1st = (SELECT wid FROM words WHERE word = ?) LIMIT ?", (word, limit))
    result = cursor.fetchall()
    if not result:
        return {"second_words": []}
    return {"second_words": result}

# Endpoint to get the revision of pages based on a specific word
@app.get("/v1/language_corpus/page_revisions_by_word", operation_id="get_page_revisions_by_word", summary="Retrieves the revisions of pages that contain a specified word. The operation filters pages based on the provided word and returns the corresponding revisions.")
async def get_page_revisions_by_word(word: str = Query(..., description="Word to filter by")):
    cursor.execute("SELECT T3.revision FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ?", (word,))
    result = cursor.fetchall()
    if not result:
        return {"revisions": []}
    return {"revisions": result}

# Endpoint to get word pairs based on word ID limit
@app.get("/v1/language_corpus/word_pairs_by_wid", operation_id="get_word_pairs_by_wid", summary="Retrieves word pairs where the first word's ID is less than or equal to the provided limit. The response includes the first and second words of each pair.")
async def get_word_pairs_by_wid(wid_limit: int = Query(..., description="Limit for word ID")):
    cursor.execute("SELECT T1.word AS W1, T3.word AS W2 FROM words AS T1 LEFT JOIN biwords AS T2 ON T1.wid = T2.w1st LEFT JOIN words AS T3 ON T3.wid = T2.w2nd WHERE T1.wid <= ? GROUP BY T1.wid", (wid_limit,))
    result = cursor.fetchall()
    if not result:
        return {"word_pairs": []}
    return {"word_pairs": result}

# Endpoint to get words and their occurrences based on page title and limit
@app.get("/v1/language_corpus/words_occurrences_by_page_title", operation_id="get_words_occurrences_by_page_title", summary="Retrieves a list of words and their respective occurrences from a specific page, as identified by its title. The results are limited to a specified number of entries. This operation is useful for analyzing word frequency within a particular page.")
async def get_words_occurrences_by_page_title(title: str = Query(..., description="Title of the page"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.word, T1.occurrences FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid WHERE T2.pid = (SELECT pid FROM pages WHERE title = ?) LIMIT ?", (title, limit))
    result = cursor.fetchall()
    if not result:
        return {"words_occurrences": []}
    return {"words_occurrences": result}

# Endpoint to get the word with the maximum occurrences
@app.get("/v1/language_corpus/word_with_max_occurrences", operation_id="get_word_with_max_occurrences", summary="Retrieves the word that appears most frequently in the corpus. This operation scans the entire corpus to identify the word with the highest occurrence count. The result is a single word that has the maximum number of appearances in the corpus.")
async def get_word_with_max_occurrences():
    cursor.execute("SELECT word FROM words WHERE occurrences = (SELECT MAX(occurrences) FROM words)")
    result = cursor.fetchall()
    if not result:
        return {"word": []}
    return {"word": result}

# Endpoint to get pages based on title containing any digit
@app.get("/v1/language_corpus/pages_by_title_digit", operation_id="get_pages_by_title_digit", summary="Retrieves pages with titles containing a specified digit (0-9). The operation filters the pages based on the provided digit and returns the page ID and title.")
async def get_pages_by_title_digit(digit: str = Query(..., description="Digit to filter by (0-9)")):
    cursor.execute("SELECT pid, title FROM pages WHERE title LIKE ? OR ? OR ? OR ? OR ? OR ? OR ? OR ? OR ? OR ?", (f'%{digit}%', f'%{digit}%', f'%{digit}%', f'%{digit}%', f'%{digit}%', f'%{digit}%', f'%{digit}%', f'%{digit}%', f'%{digit}%', f'%{digit}%'))
    result = cursor.fetchall()
    if not result:
        return {"pages": []}
    return {"pages": result}

# Endpoint to get the page with the minimum number of words
@app.get("/v1/language_corpus/page_with_min_words", operation_id="get_page_with_min_words", summary="Retrieves the title of the page with the least number of words from the corpus. This operation does not require any input parameters and returns the title of the page with the minimum word count.")
async def get_page_with_min_words():
    cursor.execute("SELECT title FROM pages WHERE title = (SELECT MIN(words) FROM pages)")
    result = cursor.fetchall()
    if not result:
        return {"title": []}
    return {"title": result}

# Endpoint to get the sum of occurrences for words of a specific length
@app.get("/v1/language_corpus/sum_occurrences_by_word_length", operation_id="get_sum_occurrences_by_word_length", summary="Retrieves the total count of words of a specified length from the language corpus.")
async def get_sum_occurrences_by_word_length(word_length: int = Query(..., description="Length of the word")):
    cursor.execute("SELECT SUM(occurrences) FROM words WHERE LENGTH(word) = ?", (word_length,))
    result = cursor.fetchone()
    if not result:
        return {"sum_occurrences": []}
    return {"sum_occurrences": result[0]}

# Endpoint to get the average number of words in pages with titles starting with a specific letter
@app.get("/v1/language_corpus/average_words_by_title_prefix", operation_id="get_average_words_by_title_prefix", summary="Retrieves the average number of words in pages whose titles begin with a specified prefix. The prefix is provided as a parameter, allowing for targeted analysis of word count averages based on the initial characters of a page's title.")
async def get_average_words_by_title_prefix(title_prefix: str = Query(..., description="Prefix of the title (e.g., 'A%')")):
    cursor.execute("SELECT AVG(words) FROM pages WHERE title LIKE ?", (title_prefix,))
    result = cursor.fetchone()
    if not result:
        return {"average_words": []}
    return {"average_words": result[0]}

# Endpoint to get the proportion of biwords with a specific first word
@app.get("/v1/language_corpus/proportion_biwords_by_first_word", operation_id="get_proportion_biwords_by_first_word", summary="Retrieves the proportion of biwords (two-word phrases) that start with a specific first word. The proportion is calculated by dividing the count of biwords with the specified first word by the total count of biwords in the corpus. The first word is identified by its unique ID.")
async def get_proportion_biwords_by_first_word(w1st: int = Query(..., description="First word ID")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN w1st = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(w1st) FROM biwords", (w1st,))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the percentage of pages with a specific word count
@app.get("/v1/language_corpus/percentage_pages_by_word_count", operation_id="get_percentage_pages_by_word_count", summary="Retrieves the percentage of pages that contain a specific number of words, given a minimum word count threshold. The operation filters pages based on the provided word count and minimum word count, and returns the percentage of pages that meet these criteria. The number of results can be limited using the 'limit' parameter.")
async def get_percentage_pages_by_word_count(word_count: int = Query(..., description="Word count"), min_words: int = Query(..., description="Minimum number of words"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN words = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(page) FROM pages WHERE words > ? LIMIT ?", (word_count, min_words, limit))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of biwords where the first and second words are the same
@app.get("/v1/language_corpus/percentage_biwords_same_words", operation_id="get_percentage_biwords_same_words", summary="Retrieves the percentage of biwords in the corpus where the first and second words are identical. This operation calculates the ratio of biwords with matching first and second words to the total number of biwords in the corpus.")
async def get_percentage_biwords_same_words():
    cursor.execute("SELECT CAST(COUNT(CASE WHEN w1st = w2nd THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(w1st) FROM biwords")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the pages containing a specific word
@app.get("/v1/language_corpus/pages_by_word", operation_id="get_pages_by_word", summary="Retrieves the pages that contain a specified word. The operation scans the database for the input word and returns the corresponding pages where it is found.")
async def get_pages_by_word(word: str = Query(..., description="Word to search for")):
    cursor.execute("SELECT T3.page FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ?", (word,))
    result = cursor.fetchall()
    if not result:
        return {"pages": []}
    return {"pages": [row[0] for row in result]}

# Endpoint to get the occurrences of words in a page with a specific title
@app.get("/v1/language_corpus/word_occurrences_by_page_title", operation_id="get_word_occurrences_by_page_title", summary="Retrieves the total number of occurrences of words in a page identified by its title. The page title is provided as an input parameter.")
async def get_word_occurrences_by_page_title(title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT occurrences FROM pages_words WHERE pid = ( SELECT pid FROM pages WHERE title = ? )", (title,))
    result = cursor.fetchall()
    if not result:
        return {"occurrences": []}
    return {"occurrences": [row[0] for row in result]}

# Endpoint to check if a specific word pair exists in biwords
@app.get("/v1/language_corpus/check_word_pair_in_biwords", operation_id="check_word_pair_in_biwords", summary="Check if a specific word pair exists in biwords")
async def check_word_pair_in_biwords(word1: str = Query(..., description="First word"), word2_pattern: str = Query(..., description="Pattern for the second word (e.g., 'd%egees')")):
    cursor.execute("SELECT CASE WHEN COUNT(T1.wid) > 0 THEN 'yes' ELSE 'no' END FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st OR T1.wid = T2.w2nd WHERE T2.w1st = ( SELECT wid FROM words WHERE word = ? ) AND T2.w2nd = ( SELECT wid FROM words WHERE word LIKE ? )", (word1, word2_pattern))
    result = cursor.fetchone()
    if not result:
        return {"exists": []}
    return {"exists": result[0]}

# Endpoint to get the average occurrences of words in pages with a specific word count
@app.get("/v1/language_corpus/average_word_occurrences_by_word_count", operation_id="get_average_word_occurrences_by_word_count", summary="Get the average occurrences of words in pages with a specific word count")
async def get_average_word_occurrences_by_word_count(word_count: int = Query(..., description="Word count")):
    cursor.execute("SELECT CAST(SUM(T2.occurrences) AS REAL) / COUNT(T1.page) FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T1.words = ?", (word_count,))
    result = cursor.fetchone()
    if not result:
        return {"average_occurrences": []}
    return {"average_occurrences": result[0]}

# Endpoint to get pages with word count greater than a specified number
@app.get("/v1/language_corpus/pages_with_word_count_greater_than", operation_id="get_pages_with_word_count_greater_than", summary="Get pages with word count greater than a specified number")
async def get_pages_with_word_count_greater_than(min_words: int = Query(..., description="Minimum number of words"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT page FROM pages WHERE words > ? LIMIT ?", (min_words, limit))
    result = cursor.fetchall()
    if not result:
        return {"pages": []}
    return {"pages": [row[0] for row in result]}

# Endpoint to get occurrences of a word by its ID
@app.get("/v1/language_corpus/word_occurrences_by_id", operation_id="get_word_occurrences_by_id", summary="Get the occurrences of a word by its ID")
async def get_word_occurrences_by_id(wid: int = Query(..., description="Word ID")):
    cursor.execute("SELECT occurrences FROM words WHERE wid = ?", (wid,))
    result = cursor.fetchone()
    if not result:
        return {"occurrences": []}
    return {"occurrences": result[0]}

# Endpoint to get words ordered by occurrences
@app.get("/v1/language_corpus/words_ordered_by_occurrences", operation_id="get_words_ordered_by_occurrences", summary="Get words ordered by their occurrences in descending order")
async def get_words_ordered_by_occurrences(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT word, wid FROM words ORDER BY occurrences DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"words": []}
    return {"words": [{"word": row[0], "wid": row[1]} for row in result]}

# Endpoint to get occurrences of biwords based on first and second word IDs
@app.get("/v1/language_corpus/biword_occurrences", operation_id="get_biword_occurrences", summary="Get the occurrences of biwords based on the first and second word IDs")
async def get_biword_occurrences(w1st: int = Query(..., description="First word ID"), w2nd: int = Query(..., description="Second word ID")):
    cursor.execute("SELECT occurrences FROM biwords WHERE w1st = ? AND w2nd = ?", (w1st, w2nd))
    result = cursor.fetchone()
    if not result:
        return {"occurrences": []}
    return {"occurrences": result[0]}

# Endpoint to get words from a page by its revision number
@app.get("/v1/language_corpus/page_words_by_revision", operation_id="get_page_words_by_revision", summary="Get the words from a page by its revision number")
async def get_page_words_by_revision(revision: int = Query(..., description="Revision number")):
    cursor.execute("SELECT words FROM pages WHERE revision = ?", (revision,))
    result = cursor.fetchone()
    if not result:
        return {"words": []}
    return {"words": result[0]}

# Endpoint to get the percentage of language words with occurrences greater than a specified number
@app.get("/v1/language_corpus/percentage_langs_words_occurrences_greater_than", operation_id="get_percentage_langs_words_occurrences_greater_than", summary="Get the percentage of language words with occurrences greater than a specified number")
async def get_percentage_langs_words_occurrences_greater_than(min_occurrences: int = Query(..., description="Minimum number of occurrences")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN occurrences > ? THEN lid ELSE NULL END) AS REAL) * 100 / COUNT(lid) FROM langs_words", (min_occurrences,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the page with the maximum number of words
@app.get("/v1/language_corpus/page_with_max_words", operation_id="get_page_with_max_words", summary="Get the page with the maximum number of words")
async def get_page_with_max_words():
    cursor.execute("SELECT page FROM pages WHERE words = ( SELECT MAX(words) FROM pages )")
    result = cursor.fetchone()
    if not result:
        return {"page": []}
    return {"page": result[0]}

# Endpoint to get the percentage of biwords with occurrences less than a specified number
@app.get("/v1/language_corpus/percentage_biwords_occurrences_less_than", operation_id="get_percentage_biwords_occurrences_less_than", summary="Get the percentage of biwords with occurrences less than a specified number")
async def get_percentage_biwords_occurrences_less_than(max_occurrences: int = Query(..., description="Maximum number of occurrences")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN occurrences < ? THEN lid ELSE NULL END) AS REAL) * 100 / COUNT(lid) FROM biwords", (max_occurrences,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get page titles and revisions by language ID
@app.get("/v1/language_corpus/page_titles_revisions_by_language", operation_id="get_page_titles_revisions_by_language", summary="Get page titles and revisions by language ID")
async def get_page_titles_revisions_by_language(lid: int = Query(..., description="Language ID"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT title, revision FROM pages WHERE lid = ? LIMIT ?", (lid, limit))
    result = cursor.fetchall()
    if not result:
        return {"pages": []}
    return {"pages": [{"title": row[0], "revision": row[1]} for row in result]}

# Endpoint to get languages of biwords based on first and second word IDs
@app.get("/v1/language_corpus/biword_languages", operation_id="get_biword_languages", summary="Get the languages of biwords based on the first and second word IDs")
async def get_biword_languages(w1st: int = Query(..., description="First word ID"), w2nd: int = Query(..., description="Second word ID")):
    cursor.execute("SELECT T2.lang FROM biwords AS T1 INNER JOIN langs AS T2 ON T1.lid = T2.lid WHERE T1.w1st = ? AND T1.w2nd = ?", (w1st, w2nd))
    result = cursor.fetchall()
    if not result:
        return {"languages": []}
    return {"languages": [row[0] for row in result]}

# Endpoint to get the occurrences of a specific word
@app.get("/v1/language_corpus/word_occurrences", operation_id="get_word_occurrences", summary="Get the occurrences of a specific word")
async def get_word_occurrences(word: str = Query(..., description="The word to search for")):
    cursor.execute("SELECT T1.occurrences FROM langs_words AS T1 INNER JOIN words AS T2 ON T1.wid = T2.wid WHERE T2.word = ?", (word,))
    result = cursor.fetchone()
    if not result:
        return {"occurrences": []}
    return {"occurrences": result[0]}

# Endpoint to get the word by its ID
@app.get("/v1/language_corpus/word_by_id", operation_id="get_word_by_id", summary="Get the word by its ID")
async def get_word_by_id(wid: int = Query(..., description="The ID of the word")):
    cursor.execute("SELECT word FROM words WHERE wid = ?", (wid,))
    result = cursor.fetchone()
    if not result:
        return {"word": []}
    return {"word": result[0]}

# Endpoint to get the occurrences of a word on a specific page
@app.get("/v1/language_corpus/word_occurrences_on_page", operation_id="get_word_occurrences_on_page", summary="Get the occurrences of a word on a specific page")
async def get_word_occurrences_on_page(word: str = Query(..., description="The word to search for"), pid: int = Query(..., description="The ID of the page")):
    cursor.execute("SELECT T2.occurrences FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid WHERE T1.word = ? AND T2.pid = ?", (word, pid))
    result = cursor.fetchone()
    if not result:
        return {"occurrences": []}
    return {"occurrences": result[0]}

# Endpoint to get the sum of occurrences of a biword on a specific page
@app.get("/v1/language_corpus/sum_occurrences_biword_on_page", operation_id="get_sum_occurrences_biword_on_page", summary="Get the sum of occurrences of a biword on a specific page")
async def get_sum_occurrences_biword_on_page(w2nd: int = Query(..., description="The second word ID of the biword"), w1st: int = Query(..., description="The first word ID of the biword"), pid: int = Query(..., description="The ID of the page")):
    cursor.execute("SELECT SUM(T1.occurrences) FROM pages_words AS T1 INNER JOIN biwords AS T2 ON T2.w2nd = T1.wid WHERE T2.w2nd = ? AND T2.w1st = ? AND T1.pid = ?", (w2nd, w1st, pid))
    result = cursor.fetchone()
    if not result:
        return {"sum_occurrences": []}
    return {"sum_occurrences": result[0]}

# Endpoint to get the percentage of words with occurrences less than a specified number in a specific language
@app.get("/v1/language_corpus/percentage_words_less_occurrences", operation_id="get_percentage_words_less_occurrences", summary="Get the percentage of words with occurrences less than a specified number in a specific language")
async def get_percentage_words_less_occurrences(occurrences_threshold: int = Query(..., description="The threshold for occurrences"), lang: str = Query(..., description="The language code")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.occurrences < ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.lid) FROM langs AS T1 INNER JOIN langs_words AS T2 ON T1.lid = T2.lid WHERE T1.lang = ?", (occurrences_threshold, lang))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of pages with more than a specified number of words in a specific language
@app.get("/v1/language_corpus/percentage_pages_more_words", operation_id="get_percentage_pages_more_words", summary="Get the percentage of pages with more than a specified number of words in a specific language")
async def get_percentage_pages_more_words(words_threshold: int = Query(..., description="The threshold for the number of words"), lang: str = Query(..., description="The language code")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.words > ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.page) FROM langs AS T1 INNER JOIN pages AS T2 ON T1.lid = T2.lid WHERE T1.lang = ?", (words_threshold, lang))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the occurrences of a specific word
@app.get("/v1/language_corpus/word_occurrences_by_word", operation_id="get_word_occurrences_by_word", summary="Get the occurrences of a specific word")
async def get_word_occurrences_by_word(word: str = Query(..., description="The word to search for")):
    cursor.execute("SELECT occurrences FROM words WHERE word = ?", (word,))
    result = cursor.fetchone()
    if not result:
        return {"occurrences": []}
    return {"occurrences": result[0]}

# Endpoint to get the count of biwords with occurrences greater than a specified number
@app.get("/v1/language_corpus/count_biwords_greater_occurrences", operation_id="get_count_biwords_greater_occurrences", summary="Get the count of biwords with occurrences greater than a specified number")
async def get_count_biwords_greater_occurrences(occurrences_threshold: int = Query(..., description="The threshold for occurrences")):
    cursor.execute("SELECT COUNT(w1st) AS countwords FROM biwords WHERE occurrences > ?", (occurrences_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"countwords": []}
    return {"countwords": result[0]}

# Endpoint to get the list of pages from the language corpus
@app.get("/v1/language_corpus/list_pages", operation_id="get_list_pages", summary="Get the list of pages from the language corpus")
async def get_list_pages():
    cursor.execute("SELECT pages FROM langs")
    result = cursor.fetchall()
    if not result:
        return {"pages": []}
    return {"pages": [row[0] for row in result]}

# Endpoint to get the count of words based on occurrence range
@app.get("/v1/language_corpus/word_count_by_occurrences", operation_id="get_word_count_by_occurrences", summary="Get the count of words with occurrences between a specified range")
async def get_word_count_by_occurrences(min_occurrences: int = Query(..., description="Minimum number of occurrences"), max_occurrences: int = Query(..., description="Maximum number of occurrences")):
    cursor.execute("SELECT COUNT(wid) FROM langs_words WHERE occurrences BETWEEN ? AND ?", (min_occurrences, max_occurrences))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of occurrences for a pair of words
@app.get("/v1/language_corpus/sum_occurrences_for_word_pair", operation_id="get_sum_occurrences_for_word_pair", summary="Get the sum of occurrences for a pair of words")
async def get_sum_occurrences_for_word_pair(word1: str = Query(..., description="First word"), word2: str = Query(..., description="Second word")):
    cursor.execute("SELECT SUM(occurrences) FROM biwords WHERE w1st = ( SELECT wid FROM words WHERE word = ? ) AND w2nd = ( SELECT wid FROM words WHERE word = ? )", (word1, word2))
    result = cursor.fetchone()
    if not result:
        return {"sum_occurrences": []}
    return {"sum_occurrences": result[0]}

# Endpoint to get the locale of a page based on its title
@app.get("/v1/language_corpus/locale_by_page_title", operation_id="get_locale_by_page_title", summary="Get the locale of a page with a specific title")
async def get_locale_by_page_title(title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT T1.locale FROM langs AS T1 INNER JOIN pages AS T2 ON T1.lid = T2.lid WHERE T2.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"locale": []}
    return {"locale": result[0]}

# Endpoint to get the word based on the number of occurrences
@app.get("/v1/language_corpus/word_by_occurrences", operation_id="get_word_by_occurrences", summary="Get the word with a specific number of occurrences")
async def get_word_by_occurrences(occurrences: int = Query(..., description="Number of occurrences")):
    cursor.execute("SELECT T1.word FROM words AS T1 INNER JOIN langs_words AS T2 ON T1.wid = T2.wid WHERE T2.occurrences = ?", (occurrences,))
    result = cursor.fetchone()
    if not result:
        return {"word": []}
    return {"word": result[0]}

# Endpoint to get the words from a language based on a pair of word IDs
@app.get("/v1/language_corpus/words_by_word_ids", operation_id="get_words_by_word_ids", summary="Get the words from a language based on a pair of word IDs")
async def get_words_by_word_ids(w1st: int = Query(..., description="First word ID"), w2nd: int = Query(..., description="Second word ID")):
    cursor.execute("SELECT words FROM langs WHERE lid = ( SELECT lid FROM biwords WHERE w1st = ? AND w2nd = ? )", (w1st, w2nd))
    result = cursor.fetchone()
    if not result:
        return {"words": []}
    return {"words": result[0]}

# Endpoint to get the count of pages based on the number of occurrences
@app.get("/v1/language_corpus/page_count_by_occurrences", operation_id="get_page_count_by_occurrences", summary="Get the count of pages with a specific number of occurrences")
async def get_page_count_by_occurrences(occurrences: int = Query(..., description="Number of occurrences")):
    cursor.execute("SELECT COUNT(T1.pages) FROM langs AS T1 INNER JOIN langs_words AS T2 ON T1.lid = T2.lid WHERE T2.occurrences = ?", (occurrences,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct page titles based on word ID threshold
@app.get("/v1/language_corpus/distinct_page_titles_by_word_id", operation_id="get_distinct_page_titles_by_word_id", summary="Get distinct page titles where the word ID is less than a specified threshold")
async def get_distinct_page_titles_by_word_id(word_id_threshold: int = Query(..., description="Threshold for word ID")):
    cursor.execute("SELECT DISTINCT T1.title FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T2.wid < ?", (word_id_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of word occurrences in pages with a specific number of words
@app.get("/v1/language_corpus/count_word_occurrences_by_words", operation_id="get_count_word_occurrences_by_words", summary="Get the count of word occurrences in pages with a specific number of words")
async def get_count_word_occurrences_by_words(words: int = Query(..., description="Number of words in the page")):
    cursor.execute("SELECT COUNT(T2.wid) FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T1.words = ?", (words,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of word occurrences and language ID for pages with a specific number of occurrences
@app.get("/v1/language_corpus/count_word_occurrences_by_occurrences", operation_id="get_count_word_occurrences_by_occurrences", summary="Get the count of word occurrences and language ID for pages with a specific number of occurrences")
async def get_count_word_occurrences_by_occurrences(occurrences: int = Query(..., description="Number of occurrences of the word")):
    cursor.execute("SELECT COUNT(T2.wid), T1.lid FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T2.occurrences = ?", (occurrences,))
    result = cursor.fetchall()
    if not result:
        return {"count": []}
    return {"count": result}

# Endpoint to get the percentage of words to occurrences for pages with a revision less than a specific value
@app.get("/v1/language_corpus/percentage_words_to_occurrences_by_revision", operation_id="get_percentage_words_to_occurrences_by_revision", summary="Get the percentage of words to occurrences for pages with a revision less than a specific value")
async def get_percentage_words_to_occurrences_by_revision(revision: int = Query(..., description="Revision number of the page")):
    cursor.execute("SELECT CAST(SUM(T1.words) AS REAL) * 100 / SUM(T2.occurrences) FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T1.revision < ?", (revision,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the pages grouped by language with word occurrences greater than a specific value
@app.get("/v1/language_corpus/pages_by_language_with_occurrences", operation_id="get_pages_by_language_with_occurrences", summary="Get the pages grouped by language with word occurrences greater than a specific value")
async def get_pages_by_language_with_occurrences(occurrences: int = Query(..., description="Number of occurrences of the word")):
    cursor.execute("SELECT T1.pages FROM langs AS T1 INNER JOIN langs_words AS T2 ON T1.lid = T2.lid WHERE T2.occurrences > ? GROUP BY T1.pages", (occurrences,))
    result = cursor.fetchall()
    if not result:
        return {"pages": []}
    return {"pages": result}

# Endpoint to get word pairs based on specific word IDs
@app.get("/v1/language_corpus/word_pairs_by_word_ids", operation_id="get_word_pairs_by_word_ids", summary="Get word pairs based on specific word IDs")
async def get_word_pairs_by_word_ids(w1st: int = Query(..., description="First word ID"), w2nd: int = Query(..., description="Second word ID")):
    cursor.execute("SELECT T1.word, T3.word FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st INNER JOIN words AS T3 ON T3.wid = T2.w2nd WHERE T2.w1st = ? AND T2.w2nd = ?", (w1st, w2nd))
    result = cursor.fetchall()
    if not result:
        return {"word_pairs": []}
    return {"word_pairs": result}

# Endpoint to get page titles based on language ID and minimum number of words
@app.get("/v1/language_corpus/page_titles_by_language_and_words", operation_id="get_page_titles_by_language_and_words", summary="Get page titles based on language ID and minimum number of words")
async def get_page_titles_by_language_and_words(lid: int = Query(..., description="Language ID"), words: int = Query(..., description="Minimum number of words in the page")):
    cursor.execute("SELECT title FROM pages WHERE lid = ? AND words > ?", (lid, words))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the number of words in a page by title
@app.get("/v1/language_corpus/words_by_page_title", operation_id="get_words_by_page_title", summary="Get the number of words in a page by title")
async def get_words_by_page_title(title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT words FROM pages WHERE title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"words": []}
    return {"words": result[0]}

# Endpoint to compare the number of words between two pages by title
@app.get("/v1/language_corpus/compare_words_by_page_titles", operation_id="compare_words_by_page_titles", summary="Compare the number of words between two pages by title")
async def compare_words_by_page_titles(title1: str = Query(..., description="Title of the first page"), title2: str = Query(..., description="Title of the second page")):
    cursor.execute("SELECT CASE WHEN ( SELECT words FROM pages WHERE title = ? ) > ( SELECT words FROM pages WHERE title = ? ) THEN ? ELSE ? END", (title1, title2, title1, title2))
    result = cursor.fetchone()
    if not result:
        return {"comparison": []}
    return {"comparison": result[0]}

# Endpoint to get words with occurrences greater than a specific value
@app.get("/v1/language_corpus/words_by_occurrences", operation_id="get_words_by_occurrences", summary="Get words with occurrences greater than a specific value")
async def get_words_by_occurrences(occurrences: int = Query(..., description="Minimum number of occurrences")):
    cursor.execute("SELECT word FROM words WHERE occurrences > ?", (occurrences,))
    result = cursor.fetchall()
    if not result:
        return {"words": []}
    return {"words": result}

# Endpoint to get the locale of a page based on its title
@app.get("/v1/language_corpus/get_locale_by_title", operation_id="get_locale_by_title", summary="Get the locale of a page based on its title")
async def get_locale_by_title(title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT T2.locale FROM pages AS T1 INNER JOIN langs AS T2 ON T1.lid = T2.lid WHERE T1.title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"locale": []}
    return {"locale": result[0]}

# Endpoint to get the titles of pages where a word occurs more than a specified number of times
@app.get("/v1/language_corpus/get_page_titles_by_word_occurrences", operation_id="get_page_titles_by_word_occurrences", summary="Get the titles of pages where a word occurs more than a specified number of times")
async def get_page_titles_by_word_occurrences(word: str = Query(..., description="Word to search for"), min_occurrences: int = Query(..., description="Minimum number of occurrences")):
    cursor.execute("SELECT T3.title FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ? AND T2.occurrences > ?", (word, min_occurrences))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the total number of words in pages where a specific word occurs a specified number of times
@app.get("/v1/language_corpus/get_total_words_by_word_occurrences", operation_id="get_total_words_by_word_occurrences", summary="Get the total number of words in pages where a specific word occurs a specified number of times")
async def get_total_words_by_word_occurrences(word: str = Query(..., description="Word to search for"), occurrences: int = Query(..., description="Number of occurrences")):
    cursor.execute("SELECT SUM(T3.words) FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ? AND T2.occurrences = ?", (word, occurrences))
    result = cursor.fetchone()
    if not result:
        return {"total_words": []}
    return {"total_words": result[0]}

# Endpoint to compare occurrences of two biwords and return the more frequent one
@app.get("/v1/language_corpus/compare_biword_occurrences", operation_id="compare_biword_occurrences", summary="Compare occurrences of two biwords and return the more frequent one")
async def compare_biword_occurrences(word1: str = Query(..., description="First word of the first biword"), word2: str = Query(..., description="Second word of the first biword"), word3: str = Query(..., description="First word of the second biword"), word4: str = Query(..., description="Second word of the second biword"), biword1: str = Query(..., description="First biword"), biword2: str = Query(..., description="Second biword")):
    cursor.execute("SELECT CASE WHEN ( SELECT occurrences FROM biwords WHERE w1st = ( SELECT wid FROM words WHERE word = ? ) AND w2nd = ( SELECT wid FROM words WHERE word = ? ) ) > ( SELECT occurrences FROM biwords WHERE w1st = ( SELECT wid FROM words WHERE word = ? ) AND w2nd = ( SELECT wid FROM words WHERE word = ? ) ) THEN ? ELSE ? END AS CALUS FROM words LIMIT 1", (word1, word2, word3, word4, biword1, biword2))
    result = cursor.fetchone()
    if not result:
        return {"biword": []}
    return {"biword": result[0]}

# Endpoint to get the difference in occurrences between a word and a biword
@app.get("/v1/language_corpus/get_occurrences_difference", operation_id="get_occurrences_difference", summary="Get the difference in occurrences between a word and a biword")
async def get_occurrences_difference(word1: str = Query(..., description="First word of the biword"), word2: str = Query(..., description="Second word of the biword"), word: str = Query(..., description="Word to compare")):
    cursor.execute("SELECT occurrences - ( SELECT occurrences FROM biwords WHERE w1st = ( SELECT wid FROM words WHERE word = ? ) AND w2nd = ( SELECT wid FROM words WHERE word = ? ) ) AS CALUS FROM words WHERE word = ?", (word1, word2, word))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the biwords starting with a specific word
@app.get("/v1/language_corpus/get_biwords_starting_with", operation_id="get_biwords_starting_with", summary="Get the biwords starting with a specific word")
async def get_biwords_starting_with(word: str = Query(..., description="Word to search for")):
    cursor.execute("SELECT T1.word AS W1, T3.word AS W2 FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st INNER JOIN words AS T3 ON T3.wid = T2.w2nd WHERE T1.word = ?", (word,))
    result = cursor.fetchall()
    if not result:
        return {"biwords": []}
    return {"biwords": [{"W1": row[0], "W2": row[1]} for row in result]}

# Endpoint to get the count of biwords starting with a specific word
@app.get("/v1/language_corpus/get_biword_count_starting_with", operation_id="get_biword_count_starting_with", summary="Get the count of biwords starting with a specific word")
async def get_biword_count_starting_with(word: str = Query(..., description="Word to search for")):
    cursor.execute("SELECT COUNT(T2.w1st) FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st INNER JOIN words AS T3 ON T3.wid = T2.w2nd WHERE T1.word = ?", (word,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of pages in a language that contain a specific biword
@app.get("/v1/language_corpus/get_page_count_by_biword", operation_id="get_page_count_by_biword", summary="Get the count of pages in a language that contain a specific biword")
async def get_page_count_by_biword(word1: str = Query(..., description="First word of the biword"), word2: str = Query(..., description="Second word of the biword")):
    cursor.execute("SELECT COUNT(T1.pages) FROM langs AS T1 INNER JOIN biwords AS T2 ON T1.lid = T2.lid WHERE T2.w1st = ( SELECT wid FROM words WHERE word = ? ) AND T2.w2nd = ( SELECT wid FROM words WHERE word = ? )", (word1, word2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to calculate the percentage difference in occurrences of a word between two page titles
@app.get("/v1/language_corpus/percentage_difference_occurrences", operation_id="get_percentage_difference_occurrences", summary="Calculate the percentage difference in occurrences of a word between two page titles")
async def get_percentage_difference_occurrences(title1: str = Query(..., description="First page title"), title2: str = Query(..., description="Second page title"), word: str = Query(..., description="Word to compare")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T3.title = ? THEN T2.occurrences END) - SUM(CASE WHEN T3.title = ? THEN T2.occurrences END)) AS REAL) * 100 / SUM(CASE WHEN T3.title = ? THEN T2.occurrences END) FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ?", (title1, title2, title2, word))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the word ID with the maximum occurrences in langs_words
@app.get("/v1/language_corpus/max_occurrences_word_id", operation_id="get_max_occurrences_word_id", summary="Get the word ID with the maximum occurrences in langs_words")
async def get_max_occurrences_word_id():
    cursor.execute("SELECT wid FROM langs_words WHERE occurrences = ( SELECT MAX(occurrences) FROM langs_words )")
    result = cursor.fetchone()
    if not result:
        return {"word_id": []}
    return {"word_id": result[0]}

# Endpoint to get the second word ID with the maximum occurrences in biwords
@app.get("/v1/language_corpus/max_occurrences_second_word_id", operation_id="get_max_occurrences_second_word_id", summary="Get the second word ID with the maximum occurrences in biwords")
async def get_max_occurrences_second_word_id():
    cursor.execute("SELECT w2nd FROM biwords WHERE occurrences = ( SELECT MAX(occurrences) FROM biwords )")
    result = cursor.fetchone()
    if not result:
        return {"second_word_id": []}
    return {"second_word_id": result[0]}

# Endpoint to get the word ID of a specific word
@app.get("/v1/language_corpus/word_id", operation_id="get_word_id", summary="Get the word ID of a specific word")
async def get_word_id(word: str = Query(..., description="Word to get the ID for")):
    cursor.execute("SELECT wid FROM words WHERE word = ?", (word,))
    result = cursor.fetchone()
    if not result:
        return {"word_id": []}
    return {"word_id": result[0]}

# Endpoint to get the word with a specific number of occurrences in biwords
@app.get("/v1/language_corpus/word_by_biword_occurrences", operation_id="get_word_by_biword_occurrences", summary="Get the word with a specific number of occurrences in biwords")
async def get_word_by_biword_occurrences(occurrences: int = Query(..., description="Number of occurrences in biwords")):
    cursor.execute("SELECT T1.word FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w2nd WHERE T2.occurrences = ?", (occurrences,))
    result = cursor.fetchone()
    if not result:
        return {"word": []}
    return {"word": result[0]}

# Endpoint to get the count of word IDs for a specific word in biwords
@app.get("/v1/language_corpus/count_word_ids_in_biwords", operation_id="get_count_word_ids_in_biwords", summary="Get the count of word IDs for a specific word in biwords")
async def get_count_word_ids_in_biwords(word: str = Query(..., description="Word to count IDs for")):
    cursor.execute("SELECT COUNT(T1.wid) FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st INNER JOIN words AS T3 ON T3.wid = T2.w2nd WHERE T1.word = ?", (word,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the word ID with the highest occurrences for a specific page title
@app.get("/v1/language_corpus/max_occurrences_word_id_by_page_title", operation_id="get_max_occurrences_word_id_by_page_title", summary="Get the word ID with the highest occurrences for a specific page title")
async def get_max_occurrences_word_id_by_page_title(title: str = Query(..., description="Page title")):
    cursor.execute("SELECT T2.wid FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T1.title = ? ORDER BY T2.occurrences DESC LIMIT 1", (title,))
    result = cursor.fetchone()
    if not result:
        return {"word_id": []}
    return {"word_id": result[0]}

# Endpoint to get the total occurrences of a word in a specific page title
@app.get("/v1/language_corpus/total_occurrences_by_page_title_and_word_id", operation_id="get_total_occurrences_by_page_title_and_word_id", summary="Get the total occurrences of a word in a specific page title")
async def get_total_occurrences_by_page_title_and_word_id(title: str = Query(..., description="Page title"), wid: int = Query(..., description="Word ID")):
    cursor.execute("SELECT SUM(T2.occurrences) FROM pages AS T1 INNER JOIN pages_words AS T2 ON T1.pid = T2.pid WHERE T1.title = ? AND T2.wid = ?", (title, wid))
    result = cursor.fetchone()
    if not result:
        return {"total_occurrences": []}
    return {"total_occurrences": result[0]}

# Endpoint to get the title of a page based on word ID
@app.get("/v1/language_corpus/page_title_by_word_id", operation_id="get_page_title_by_word_id", summary="Get the title of a page based on the word ID")
async def get_page_title_by_word_id(wid: int = Query(..., description="Word ID")):
    cursor.execute("SELECT title FROM pages WHERE pid = ( SELECT pid FROM pages_words WHERE wid = ? ORDER BY occurrences DESC LIMIT 1 )", (wid,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of occurrences of a word in pages
@app.get("/v1/language_corpus/word_occurrences_count", operation_id="get_word_occurrences_count", summary="Get the count of occurrences of a word in pages")
async def get_word_occurrences_count(word: str = Query(..., description="Word to search")):
    cursor.execute("SELECT COUNT(T2.occurrences) FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid WHERE T1.word = ?", (word,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the page ID based on revision number
@app.get("/v1/language_corpus/page_id_by_revision", operation_id="get_page_id_by_revision", summary="Get the page ID based on the revision number")
async def get_page_id_by_revision(revision: int = Query(..., description="Revision number")):
    cursor.execute("SELECT pid FROM pages_words WHERE pid = ( SELECT pid FROM pages WHERE revision = ? ) ORDER BY occurrences DESC LIMIT 1", (revision,))
    result = cursor.fetchone()
    if not result:
        return {"pid": []}
    return {"pid": result[0]}

# Endpoint to get the count of first words in biwords based on the second word
@app.get("/v1/language_corpus/biword_first_word_count", operation_id="get_biword_first_word_count", summary="Get the count of first words in biwords based on the second word")
async def get_biword_first_word_count(word: str = Query(..., description="Second word")):
    cursor.execute("SELECT COUNT(w1st) FROM biwords WHERE w2nd = ( SELECT wid FROM words WHERE word = ? )", (word,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the occurrences of a word in a specific language
@app.get("/v1/language_corpus/word_occurrences_in_language", operation_id="get_word_occurrences_in_language", summary="Get the occurrences of a word in a specific language")
async def get_word_occurrences_in_language(word: str = Query(..., description="Word to search"), lid: int = Query(..., description="Language ID")):
    cursor.execute("SELECT T2.occurrences FROM words AS T1 INNER JOIN langs_words AS T2 ON T1.wid = T2.wid WHERE T1.word = ? AND T2.lid = ?", (word, lid))
    result = cursor.fetchone()
    if not result:
        return {"occurrences": []}
    return {"occurrences": result[0]}

# Endpoint to get words based on occurrences and language ID
@app.get("/v1/language_corpus/words_by_occurrences_and_language", operation_id="get_words_by_occurrences_and_language", summary="Get words based on occurrences and language ID")
async def get_words_by_occurrences_and_language(occurrences: int = Query(..., description="Number of occurrences"), lid: int = Query(..., description="Language ID")):
    cursor.execute("SELECT T1.word FROM words AS T1 INNER JOIN langs_words AS T2 ON T1.wid = T2.wid WHERE T2.occurrences = ? AND T2.lid = ?", (occurrences, lid))
    result = cursor.fetchone()
    if not result:
        return {"word": []}
    return {"word": result[0]}

# Endpoint to get the ratio of occurrences of two biwords
@app.get("/v1/language_corpus/biword_occurrences_ratio", operation_id="get_biword_occurrences_ratio", summary="Get the ratio of occurrences of two biwords")
async def get_biword_occurrences_ratio(word1a: str = Query(..., description="First word of the first biword"), word1b: str = Query(..., description="Second word of the first biword"), word2a: str = Query(..., description="First word of the second biword"), word2b: str = Query(..., description="Second word of the second biword")):
    cursor.execute("SELECT CAST(occurrences AS REAL) / ( SELECT occurrences FROM biwords WHERE w1st = ( SELECT wid FROM words WHERE word = ? ) AND w2nd = ( SELECT wid FROM words WHERE word = ? ) ) FROM biwords WHERE w1st = ( SELECT wid FROM words WHERE word = ? ) AND w2nd = ( SELECT wid FROM words WHERE word = ? )", (word1a, word1b, word2a, word2b))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of occurrences of a word in pages with a specific title
@app.get("/v1/language_corpus/word_occurrences_percentage_by_title", operation_id="get_word_occurrences_percentage_by_title", summary="Get the percentage of occurrences of a word in pages with a specific title")
async def get_word_occurrences_percentage_by_title(title: str = Query(..., description="Title of the page"), word: str = Query(..., description="Word to search")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.title = ? THEN T2.occurrences ELSE 0 END) AS REAL) * 100 / SUM(T2.occurrences) FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ?", (title, word))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of pages based on language ID and word count
@app.get("/v1/language_corpus/page_count_by_language_and_word_count", operation_id="get_page_count_by_language_and_word_count", summary="Get the count of pages based on language ID and word count")
async def get_page_count_by_language_and_word_count(lid: int = Query(..., description="Language ID"), word_count: int = Query(..., description="Number of words")):
    cursor.execute("SELECT COUNT(lid) FROM pages WHERE lid = ? AND words > ?", (lid, word_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get titles of pages with specific lid and word count
@app.get("/v1/language_corpus/page_titles_by_lid_and_words", operation_id="get_page_titles", summary="Get titles of pages with a specific lid and word count, limited by a specified number")
async def get_page_titles(lid: int = Query(..., description="Language ID"), words: int = Query(..., description="Number of words"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT title FROM pages WHERE lid = ? AND words = ? LIMIT ?", (lid, words, limit))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get titles of pages with specific lid ordered by word count
@app.get("/v1/language_corpus/page_titles_by_lid_ordered_by_words", operation_id="get_page_titles_ordered", summary="Get titles of pages with a specific lid, ordered by word count in descending order, limited by a specified number")
async def get_page_titles_ordered(lid: int = Query(..., description="Language ID"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT title FROM pages WHERE lid = ? ORDER BY words DESC LIMIT ?", (lid, limit))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get revisions of pages with specific lid and title
@app.get("/v1/language_corpus/page_revisions_by_lid_and_title", operation_id="get_page_revisions", summary="Get revisions of pages with a specific lid and title")
async def get_page_revisions(lid: int = Query(..., description="Language ID"), title: str = Query(..., description="Title of the page")):
    cursor.execute("SELECT revision FROM pages WHERE lid = ? AND title = ?", (lid, title))
    result = cursor.fetchall()
    if not result:
        return {"revisions": []}
    return {"revisions": [row[0] for row in result]}

# Endpoint to get the count of pages with specific lid, word count, and revision
@app.get("/v1/language_corpus/page_count_by_lid_words_revision", operation_id="get_page_count", summary="Get the count of pages with a specific lid, word count greater than a specified number, and revision greater than a specified number")
async def get_page_count(lid: int = Query(..., description="Language ID"), words: int = Query(..., description="Minimum number of words"), revision: int = Query(..., description="Minimum revision number")):
    cursor.execute("SELECT COUNT(lid) FROM pages WHERE lid = ? AND words > ? AND revision > ?", (lid, words, revision))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of pages containing a specific word with occurrences greater than a specified number
@app.get("/v1/language_corpus/page_count_by_word_and_occurrences", operation_id="get_page_count_by_word_and_occurrences", summary="Get the count of pages containing a specific word with occurrences greater than a specified number")
async def get_page_count_by_word_and_occurrences(word: str = Query(..., description="Word to search for"), occurrences: int = Query(..., description="Minimum number of occurrences")):
    cursor.execute("SELECT COUNT(T2.pid) FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid WHERE T1.word = ? AND T2.occurrences > ?", (word, occurrences))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of biwords starting with a specific word
@app.get("/v1/language_corpus/biword_count_by_word", operation_id="get_biword_count_by_word", summary="Get the count of biwords starting with a specific word")
async def get_biword_count_by_word(word: str = Query(..., description="Word to search for")):
    cursor.execute("SELECT COUNT(T2.w1st) FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w2nd WHERE T1.word = ?", (word,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get titles of pages containing a specific word with a specific number of occurrences
@app.get("/v1/language_corpus/page_titles_by_word_and_occurrences", operation_id="get_page_titles_by_word_and_occurrences", summary="Get titles of pages containing a specific word with a specific number of occurrences")
async def get_page_titles_by_word_and_occurrences(word: str = Query(..., description="Word to search for"), occurrences: int = Query(..., description="Number of occurrences")):
    cursor.execute("SELECT T3.title FROM words AS T1 INNER JOIN pages_words AS T2 ON T1.wid = T2.wid INNER JOIN pages AS T3 ON T2.pid = T3.pid WHERE T1.word = ? AND T2.occurrences = ?", (word, occurrences))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of biwords ending with a specific word and occurrences greater than a specified number
@app.get("/v1/language_corpus/biword_count_by_word_and_occurrences", operation_id="get_biword_count_by_word_and_occurrences", summary="Get the count of biwords ending with a specific word and occurrences greater than a specified number")
async def get_biword_count_by_word_and_occurrences(word: str = Query(..., description="Word to search for"), occurrences: int = Query(..., description="Minimum number of occurrences")):
    cursor.execute("SELECT COUNT(T2.w2nd) FROM words AS T1 INNER JOIN biwords AS T2 ON T1.wid = T2.w1st WHERE T1.word = ? AND T2.occurrences > ?", (word, occurrences))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

api_calls = [
    "/v1/language_corpus/longest_title_page",
    "/v1/language_corpus/titles_fewer_words?max_words=10",
    "/v1/language_corpus/pages_with_substring?substring=Art",
    "/v1/language_corpus/title_by_revision?revision=16203226",
    "/v1/language_corpus/titles_by_revision_range?min_revision=106600&max_revision=106700",
    "/v1/language_corpus/count_pages_by_word_range?min_words=1000&max_words=2000",
    "/v1/language_corpus/page_ids_by_word?word=decimal",
    "/v1/language_corpus/word_max_occurrences",
    "/v1/language_corpus/bigram_words?word=antic",
    "/v1/language_corpus/titles_occurrences_by_word?word=quipu",
    "/v1/language_corpus/average_occurrences_by_word?word=sistema",
    "/v1/language_corpus/pages_by_language?lang=ca",
    "/v1/language_corpus/biwords_with_max_occurrences",
    "/v1/language_corpus/word_ids_by_occurrences?max_occurrences=10",
    "/v1/language_corpus/page_title_with_max_words",
    "/v1/language_corpus/page_content_by_title?title=Arqueozoologia",
    "/v1/language_corpus/most_frequent_word_in_page?title=Abadia",
    "/v1/language_corpus/page_titles_ordered_by_words?limit=5",
    "/v1/language_corpus/total_occurrences_of_biword_pair?w1st=86&w2nd=109",
    "/v1/language_corpus/word_pairs_by_occurrences?occurrences=2",
    "/v1/language_corpus/most_frequent_biword_pair?title=Addicio",
    "/v1/language_corpus/total_words_for_titles?title1=Adam&title2=Acampada",
    "/v1/language_corpus/page_revision_by_title?title=Aigua%20dol%C3%A7a",
    "/v1/language_corpus/average_words_above_threshold?min_words=10",
    "/v1/language_corpus/revisions_below_word_count?max_words=10&limit=5",
    "/v1/language_corpus/page_ids_by_title_prefix?prefix=b",
    "/v1/language_corpus/page_titles_by_word?word=desena",
    "/v1/language_corpus/word_ids_by_page_title?title=Sometent",
    "/v1/language_corpus/check_word_id_in_page?word_id=88&title=Animals",
    "/v1/language_corpus/word_occurrences_in_page?word=del&title=Any%20anomal%C3%ADstic",
    "/v1/language_corpus/sum_occurrences_by_word?word=nombre",
    "/v1/language_corpus/second_word_by_first_word?word=john&limit=10",
    "/v1/language_corpus/page_revisions_by_word?word=fresc",
    "/v1/language_corpus/word_pairs_by_wid?wid_limit=10",
    "/v1/language_corpus/words_occurrences_by_page_title?title=Atomium&limit=3",
    "/v1/language_corpus/word_with_max_occurrences",
    "/v1/language_corpus/pages_by_title_digit?digit=0",
    "/v1/language_corpus/page_with_min_words",
    "/v1/language_corpus/sum_occurrences_by_word_length?word_length=3",
    "/v1/language_corpus/average_words_by_title_prefix?title_prefix=A%25",
    "/v1/language_corpus/proportion_biwords_by_first_word?w1st=34",
    "/v1/language_corpus/percentage_pages_by_word_count?word_count=1500&min_words=300&limit=3",
    "/v1/language_corpus/percentage_biwords_same_words",
    "/v1/language_corpus/pages_by_word?word=ripoll",
    "/v1/language_corpus/word_occurrences_by_page_title?title=Llista%20de%20conflictes%20armats",
    "/v1/language_corpus/check_word_pair_in_biwords?word1=fukunaga&word2_pattern=d%25egees",
    "/v1/language_corpus/average_word_occurrences_by_word_count?word_count=100",
    "/v1/language_corpus/pages_with_word_count_greater_than?min_words=300&limit=3",
    "/v1/language_corpus/word_occurrences_by_id?wid=8",
    "/v1/language_corpus/words_ordered_by_occurrences?limit=3",
    "/v1/language_corpus/biword_occurrences?w1st=1&w2nd=25",
    "/v1/language_corpus/page_words_by_revision?revision=27457362",
    "/v1/language_corpus/percentage_langs_words_occurrences_greater_than?min_occurrences=16000",
    "/v1/language_corpus/page_with_max_words",
    "/v1/language_corpus/percentage_biwords_occurrences_less_than?max_occurrences=80",
    "/v1/language_corpus/page_titles_revisions_by_language?lid=1&limit=3",
    "/v1/language_corpus/biword_languages?w1st=1&w2nd=616",
    "/v1/language_corpus/word_occurrences?word=nombre",
    "/v1/language_corpus/word_by_id?wid=8968",
    "/v1/language_corpus/word_occurrences_on_page?word=votives&pid=44",
    "/v1/language_corpus/sum_occurrences_biword_on_page?w2nd=109&w1st=1&pid=16",
    "/v1/language_corpus/percentage_words_less_occurrences?occurrences_threshold=180&lang=ca",
    "/v1/language_corpus/percentage_pages_more_words?words_threshold=10000&lang=ca",
    "/v1/language_corpus/word_occurrences_by_word?word=desena",
    "/v1/language_corpus/count_biwords_greater_occurrences?occurrences_threshold=10",
    "/v1/language_corpus/list_pages",
    "/v1/language_corpus/word_count_by_occurrences?min_occurrences=2000&max_occurrences=5000",
    "/v1/language_corpus/sum_occurrences_for_word_pair?word1=barcelona&word2=precolombina",
    "/v1/language_corpus/locale_by_page_title?title=Anys%2090",
    "/v1/language_corpus/word_by_occurrences?occurrences=71303",
    "/v1/language_corpus/words_by_word_ids?w1st=100&w2nd=317",
    "/v1/language_corpus/page_count_by_occurrences?occurrences=2593",
    "/v1/language_corpus/distinct_page_titles_by_word_id?word_id_threshold=20",
    "/v1/language_corpus/count_word_occurrences_by_words?words=3",
    "/v1/language_corpus/count_word_occurrences_by_occurrences?occurrences=8",
    "/v1/language_corpus/percentage_words_to_occurrences_by_revision?revision=106680",
    "/v1/language_corpus/pages_by_language_with_occurrences?occurrences=3000",
    "/v1/language_corpus/word_pairs_by_word_ids?w1st=20&w2nd=50",
    "/v1/language_corpus/page_titles_by_language_and_words?lid=1&words=4000",
    "/v1/language_corpus/words_by_page_title?title=Asclepi",
    "/v1/language_corpus/compare_words_by_page_titles?title1=Asclepi&title2=Afluent",
    "/v1/language_corpus/words_by_occurrences?occurrences=200000",
    "/v1/language_corpus/get_locale_by_title?title=Asclepi",
    "/v1/language_corpus/get_page_titles_by_word_occurrences?word=grec&min_occurrences=20",
    "/v1/language_corpus/get_total_words_by_word_occurrences?word=grec&occurrences=52",
    "/v1/language_corpus/compare_biword_occurrences?word1=\u00e0bac&word2=xin\u00e8s&word3=\u00e0bac&word4=grec&biword1=\u00e0bac-xin\u00e8s&biword2=\u00e0bac-grec",
    "/v1/language_corpus/get_occurrences_difference?word1=\u00e0bac&word2=xin\u00e8s&word=\u00e0bac",
    "/v1/language_corpus/get_biwords_starting_with?word=\u00e0bac",
    "/v1/language_corpus/get_biword_count_starting_with?word=\u00e0bac",
    "/v1/language_corpus/get_page_count_by_biword?word1=\u00e0bac&word2=xin\u00e8s",
    "/v1/language_corpus/percentage_difference_occurrences?title1=\u00c0bac&title2=Astronomia&word=grec",
    "/v1/language_corpus/max_occurrences_word_id",
    "/v1/language_corpus/max_occurrences_second_word_id",
    "/v1/language_corpus/word_id?word=periodograma",
    "/v1/language_corpus/word_by_biword_occurrences?occurrences=116430",
    "/v1/language_corpus/count_word_ids_in_biwords?word=riu",
    "/v1/language_corpus/max_occurrences_word_id_by_page_title?title=Agricultura",
    "/v1/language_corpus/total_occurrences_by_page_title_and_word_id?title=Astre&wid=2823",
    "/v1/language_corpus/page_title_by_word_id?wid=174",
    "/v1/language_corpus/word_occurrences_count?word=her\u00f2dot",
    "/v1/language_corpus/page_id_by_revision?revision=28278070",
    "/v1/language_corpus/biword_first_word_count?word=base",
    "/v1/language_corpus/word_occurrences_in_language?word=exemple&lid=1",
    "/v1/language_corpus/words_by_occurrences_and_language?occurrences=274499&lid=1",
    "/v1/language_corpus/biword_occurrences_ratio?word1a=a&word1b=decimal&word2a=a&word2b=base",
    "/v1/language_corpus/word_occurrences_percentage_by_title?title=Art&word=grec",
    "/v1/language_corpus/page_count_by_language_and_word_count?lid=1&word_count=4000",
    "/v1/language_corpus/page_titles_by_lid_and_words?lid=1&words=10&limit=10",
    "/v1/language_corpus/page_titles_by_lid_ordered_by_words?lid=1&limit=3",
    "/v1/language_corpus/page_revisions_by_lid_and_title?lid=1&title=Arqueologia",
    "/v1/language_corpus/page_count_by_lid_words_revision?lid=1&words=300&revision=28330000",
    "/v1/language_corpus/page_count_by_word_and_occurrences?word=nombre&occurrences=5",
    "/v1/language_corpus/biword_count_by_word?word=grec",
    "/v1/language_corpus/page_titles_by_word_and_occurrences?word=grec&occurrences=52",
    "/v1/language_corpus/biword_count_by_word_and_occurrences?word=\u00e0bac&occurrences=10"
]
