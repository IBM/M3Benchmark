from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/codebase_community/codebase_community.sqlite')
cursor = conn.cursor()

# Endpoint to get the display name of users with the highest reputation among a given set of display names
@app.get("/v1/codebase_community/users/highest_reputation_display_names", operation_id="get_highest_reputation_display_names", summary="Retrieve the display names of users who have the highest reputation among the provided set of display names. This operation requires at least two display names as input parameters. The display names are used to filter the users and identify those with the maximum reputation. The operation returns the display names of the users who meet the criteria.")
async def get_highest_reputation_display_names(display_name1: str = Query(..., description="First display name"), display_name2: str = Query(..., description="Second display name")):
    cursor.execute("SELECT DisplayName FROM users WHERE DisplayName IN (?, ?) AND Reputation = ( SELECT MAX(Reputation) FROM users WHERE DisplayName IN (?, ?) )", (display_name1, display_name2, display_name1, display_name2))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the display name of users created in a specific year
@app.get("/v1/codebase_community/users/display_names_by_creation_year", operation_id="get_display_names_by_creation_year", summary="Retrieves the display names of users who were created in a specific year. The year must be provided in the 'YYYY' format.")
async def get_display_names_by_creation_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DisplayName FROM users WHERE STRFTIME('%Y', CreationDate) = ?", (year,))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the count of users who last accessed after a specific date
@app.get("/v1/codebase_community/users/count_last_access_after_date", operation_id="get_count_last_access_after_date", summary="Retrieves the total number of users who last accessed the system after the specified date. The date should be provided in 'YYYY-MM-DD' format.")
async def get_count_last_access_after_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(Id) FROM users WHERE date(LastAccessDate) > ?", (date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display name of users with the highest views
@app.get("/v1/codebase_community/users/highest_views_display_names", operation_id="get_highest_views_display_names", summary="Retrieves the display names of users who have accumulated the highest number of views. This operation returns a list of display names that correspond to the users with the maximum view count in the system.")
async def get_highest_views_display_names():
    cursor.execute("SELECT DisplayName FROM users WHERE Views = ( SELECT MAX(Views) FROM users )")
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the count of users with upvotes greater than a specific number and downvotes greater than a specific number
@app.get("/v1/codebase_community/users/count_upvotes_downvotes", operation_id="get_count_upvotes_downvotes", summary="Retrieves the count of users who have received more than the specified number of upvotes and downvotes. This operation allows you to understand the distribution of user engagement based on the provided upvote and downvote thresholds.")
async def get_count_upvotes_downvotes(upvotes: int = Query(..., description="Number of upvotes"), downvotes: int = Query(..., description="Number of downvotes")):
    cursor.execute("SELECT COUNT(Id) FROM users WHERE Upvotes > ? AND Downvotes > ?", (upvotes, downvotes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users created after a specific year with views greater than a specific number
@app.get("/v1/codebase_community/users/count_creation_year_views", operation_id="get_count_creation_year_views", summary="Retrieves the total number of users who were created after the specified year and have accumulated more views than the provided threshold. The year should be provided in 'YYYY' format, and the views threshold is a numerical value.")
async def get_count_creation_year_views(year: str = Query(..., description="Year in 'YYYY' format"), views: int = Query(..., description="Number of views")):
    cursor.execute("SELECT COUNT(id) FROM users WHERE STRFTIME('%Y', CreationDate) > ? AND Views > ?", (year, views))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of posts by a specific user
@app.get("/v1/codebase_community/posts/count_posts_by_user", operation_id="get_count_posts_by_user", summary="Retrieves the total number of posts created by a user identified by their display name. This operation provides a quantitative measure of a user's activity within the community.")
async def get_count_posts_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT COUNT(T1.id) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the titles of posts by a specific user
@app.get("/v1/codebase_community/posts/titles_by_user", operation_id="get_titles_by_user", summary="Retrieves the titles of all posts authored by a specific user, identified by their display name.")
async def get_titles_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T1.Title FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the display name of users who posted a specific title
@app.get("/v1/codebase_community/posts/display_names_by_title", operation_id="get_display_names_by_title", summary="Retrieves the display names of users who have posted a specific title. The operation filters posts by the provided title and returns the corresponding user display names.")
async def get_display_names_by_title(title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T2.DisplayName FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the most viewed post title by a specific user
@app.get("/v1/codebase_community/posts/most_viewed_post_by_user", operation_id="get_most_viewed_post_by_user", summary="Retrieves the title of the post with the highest view count for a given user. The user is identified by their display name.")
async def get_most_viewed_post_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T1.Title FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = ? ORDER BY T1.ViewCount DESC LIMIT 1", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the display name of the user with the highest favorite count post
@app.get("/v1/codebase_community/top_favorite_post_user", operation_id="get_top_favorite_post_user", summary="Retrieves the display name of the user who has authored the post with the highest number of favorites, as determined by the 'FavoriteCount' field. The 'limit' parameter can be used to restrict the number of results returned.")
async def get_top_favorite_post_user(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.DisplayName FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id ORDER BY T1.FavoriteCount DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the total comment count for posts by a specific user
@app.get("/v1/codebase_community/total_comment_count_by_user", operation_id="get_total_comment_count_by_user", summary="Retrieves the cumulative count of comments on posts authored by a user identified by their display name.")
async def get_total_comment_count_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT SUM(T1.CommentCount) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_comment_count": []}
    return {"total_comment_count": result[0]}

# Endpoint to get the maximum answer count for posts by a specific user
@app.get("/v1/codebase_community/max_answer_count_by_user", operation_id="get_max_answer_count_by_user", summary="Retrieves the highest number of answers associated with a single post created by a specific user. The user is identified by their display name.")
async def get_max_answer_count_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT MAX(T1.AnswerCount) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_answer_count": []}
    return {"max_answer_count": result[0]}

# Endpoint to get the display name of the last editor of a post with a specific title
@app.get("/v1/codebase_community/last_editor_by_post_title", operation_id="get_last_editor_by_post_title", summary="Retrieves the display name of the user who last edited a post with the specified title. The title parameter is used to identify the post.")
async def get_last_editor_by_post_title(title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T2.DisplayName FROM posts AS T1 INNER JOIN users AS T2 ON T1.LastEditorUserId = T2.Id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the count of top-level posts by a specific user
@app.get("/v1/codebase_community/top_level_post_count_by_user", operation_id="get_top_level_post_count_by_user", summary="Retrieves the total number of top-level posts created by a specific user. The user is identified by their display name.")
async def get_top_level_post_count_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT COUNT(T1.Id) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = ? AND T1.ParentId IS NULL", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display names of users who have closed posts
@app.get("/v1/codebase_community/users_with_closed_posts", operation_id="get_users_with_closed_posts", summary="Retrieves the display names of users who have closed posts. This operation fetches the user display names from the users table that have corresponding closed posts in the posts table. The closed posts are identified by a non-null ClosedDate value.")
async def get_users_with_closed_posts():
    cursor.execute("SELECT T2.DisplayName FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T1.ClosedDate IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the count of posts by users with a specific score and age
@app.get("/v1/codebase_community/post_count_by_score_and_age", operation_id="get_post_count_by_score_and_age", summary="Retrieves the total number of posts made by users who have a score equal to or greater than the specified minimum score and an age greater than the specified minimum age. This operation provides insights into the activity of users based on their score and age.")
async def get_post_count_by_score_and_age(min_score: int = Query(..., description="Minimum score of the post"), min_age: int = Query(..., description="Minimum age of the user")):
    cursor.execute("SELECT COUNT(T1.Id) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T1.Score >= ? AND T2.Age > ?", (min_score, min_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the location of the user who posted a specific title
@app.get("/v1/codebase_community/user_location_by_post_title", operation_id="get_user_location_by_post_title", summary="Retrieves the location of the user who authored a post with a specific title. The operation uses the provided post title to search for the corresponding post and then identifies the user who created it. The location of this user is then returned.")
async def get_user_location_by_post_title(title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T2.Location FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": result[0]}

# Endpoint to get the body of posts with a specific tag
@app.get("/v1/codebase_community/post_body_by_tag", operation_id="get_post_body_by_tag", summary="Retrieves the content of posts associated with a specified tag. The operation filters posts based on the provided tag name and returns the body of the matching posts.")
async def get_post_body_by_tag(tag_name: str = Query(..., description="Name of the tag")):
    cursor.execute("SELECT T2.Body FROM tags AS T1 INNER JOIN posts AS T2 ON T2.Id = T1.ExcerptPostId WHERE T1.TagName = ?", (tag_name,))
    result = cursor.fetchall()
    if not result:
        return {"bodies": []}
    return {"bodies": [row[0] for row in result]}

# Endpoint to get the body of the post with the highest tag count
@app.get("/v1/codebase_community/top_tagged_post_body", operation_id="get_top_tagged_post_body", summary="Retrieves the content of the post that has been tagged the most. The number of top-tagged posts to consider can be limited by the provided input parameter.")
async def get_top_tagged_post_body(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Body FROM posts WHERE id = ( SELECT ExcerptPostId FROM tags ORDER BY Count DESC LIMIT ? )", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"body": []}
    return {"body": result[0]}

# Endpoint to get the count of badges for a user by display name
@app.get("/v1/codebase_community/badges/count_by_display_name", operation_id="get_badge_count_by_display_name", summary="Retrieves the total number of badges associated with a specific user, identified by their display name.")
async def get_badge_count_by_display_name(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT COUNT(T1.Id) FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of badges for a user by display name
@app.get("/v1/codebase_community/badges/names_by_display_name", operation_id="get_badge_names_by_display_name", summary="Retrieves the names of badges associated with a user, identified by their display name. The display name is used to locate the user's record and subsequently fetch the corresponding badge names.")
async def get_badge_names_by_display_name(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T1.`Name` FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of badges for a user by year and display name
@app.get("/v1/codebase_community/badges/count_by_year_and_display_name", operation_id="get_badge_count_by_year_and_display_name", summary="Retrieves the total number of badges earned by a user in a specific year, identified by their display name. The response provides a count of badges awarded to the user during the given year.")
async def get_badge_count_by_year_and_display_name(year: str = Query(..., description="Year in 'YYYY' format"), display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT COUNT(T1.Id) FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE STRFTIME('%Y', T1.Date) = ? AND T2.DisplayName = ?", (year, display_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display name of the user with the most badges
@app.get("/v1/codebase_community/badges/top_user_by_badge_count", operation_id="get_top_user_by_badge_count", summary="Retrieves the display name of the user who has earned the highest number of badges. This operation ranks users based on their badge count and returns the top-ranked user's display name.")
async def get_top_user_by_badge_count():
    cursor.execute("SELECT T2.DisplayName FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id GROUP BY T2.DisplayName ORDER BY COUNT(T1.Id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the average score of posts for a user by display name
@app.get("/v1/codebase_community/posts/average_score_by_display_name", operation_id="get_average_score_by_display_name", summary="Retrieves the average score of posts associated with a specific user, identified by their display name.")
async def get_average_score_by_display_name(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT AVG(T1.Score) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get the ratio of badges to distinct users with views greater than a specified number
@app.get("/v1/codebase_community/badges/ratio_badges_to_users_by_views", operation_id="get_ratio_badges_to_users_by_views", summary="Retrieves the ratio of badges to distinct users who have more than the specified minimum number of views. This operation calculates the ratio by dividing the total count of badges by the count of unique users with views exceeding the provided minimum threshold.")
async def get_ratio_badges_to_users_by_views(min_views: int = Query(..., description="Minimum number of views")):
    cursor.execute("SELECT CAST(COUNT(T1.Id) AS REAL) / COUNT(DISTINCT T2.DisplayName) FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T2.Views > ?", (min_views,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of posts by users older than a specified age with a score greater than a specified value
@app.get("/v1/codebase_community/posts/percentage_by_age_and_score", operation_id="get_percentage_by_age_and_score", summary="Retrieves the percentage of posts created by users who are older than the specified minimum age and have a score greater than the specified minimum value. This operation calculates the percentage by comparing the count of posts that meet the age and score criteria to the total number of posts.")
async def get_percentage_by_age_and_score(min_age: int = Query(..., description="Minimum age of the user"), min_score: int = Query(..., description="Minimum score of the post")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Age > ?, 1, 0)) AS REAL) * 100 / COUNT(T1.Id) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T1.Score > ?", (min_age, min_score))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of votes for a user on a specific date
@app.get("/v1/codebase_community/votes/count_by_user_and_date", operation_id="get_vote_count_by_user_and_date", summary="Retrieves the total number of votes cast by a specific user on a given date. The user is identified by their unique User ID, and the date is provided in the 'YYYY-MM-DD' format.")
async def get_vote_count_by_user_and_date(user_id: int = Query(..., description="User ID"), creation_date: str = Query(..., description="Creation date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(Id) FROM votes WHERE UserId = ? AND CreationDate = ?", (user_id, creation_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the date with the most votes
@app.get("/v1/codebase_community/votes/most_votes_date", operation_id="get_most_votes_date", summary="Retrieves the date on which the highest number of votes were recorded. The operation returns the date with the most votes, based on the count of unique vote IDs.")
async def get_most_votes_date():
    cursor.execute("SELECT CreationDate FROM votes GROUP BY CreationDate ORDER BY COUNT(Id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the count of badges by name
@app.get("/v1/codebase_community/badges/count_by_name", operation_id="get_badge_count_by_name", summary="Retrieves the total count of badges that match the specified name. This operation allows you to determine the frequency of a particular badge within the system.")
async def get_badge_count_by_name(name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT COUNT(Id) FROM badges WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the title of the post with the highest-scoring comment
@app.get("/v1/codebase_community/top_scoring_comment_post_title", operation_id="get_top_scoring_comment_post_title", summary="Retrieves the title of the post that has the comment with the highest score. The score of a comment is determined by the number of upvotes it has received. This operation does not require any input parameters.")
async def get_top_scoring_comment_post_title():
    cursor.execute("SELECT Title FROM posts WHERE Id = ( SELECT PostId FROM comments ORDER BY Score DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the count of posts with a specific view count
@app.get("/v1/codebase_community/post_count_by_view_count", operation_id="get_post_count_by_view_count", summary="Retrieves the total number of posts that have received a specific number of views. This operation considers the view count of each post and returns the corresponding count. The input parameter specifies the view count to be considered for this calculation.")
async def get_post_count_by_view_count(view_count: int = Query(..., description="View count of the post")):
    cursor.execute("SELECT COUNT(T1.Id) FROM posts AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.PostId WHERE T1.ViewCount = ?", (view_count,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the favorite count of posts based on comment creation date and user ID
@app.get("/v1/codebase_community/favorite_count_by_comment_date_user", operation_id="get_favorite_count_by_comment_date_user", summary="Retrieves the total number of favorites for posts that have comments created on a specific date by a particular user. The response is based on the provided comment creation date and user ID.")
async def get_favorite_count_by_comment_date_user(creation_date: str = Query(..., description="Creation date of the comment in 'YYYY-MM-DD HH:MM:SS.SSS' format"), user_id: int = Query(..., description="User ID of the commenter")):
    cursor.execute("SELECT T1.FavoriteCount FROM posts AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.PostId WHERE T2.CreationDate = ? AND T2.UserId = ?", (creation_date, user_id))
    result = cursor.fetchone()
    if not result:
        return {"favorite_count": []}
    return {"favorite_count": result[0]}

# Endpoint to get the text of comments based on post parent ID and comment count
@app.get("/v1/codebase_community/comment_text_by_parent_id_comment_count", operation_id="get_comment_text_by_parent_id_comment_count", summary="Retrieves the text of comments associated with a specific post, identified by its parent ID and comment count. This operation filters comments based on the provided parent ID and comment count, and returns the corresponding comment text.")
async def get_comment_text_by_parent_id_comment_count(parent_id: int = Query(..., description="Parent ID of the post"), comment_count: int = Query(..., description="Comment count of the post")):
    cursor.execute("SELECT T2.Text FROM posts AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.PostId WHERE T1.ParentId = ? AND T1.CommentCount = ?", (parent_id, comment_count))
    result = cursor.fetchone()
    if not result:
        return {"text": []}
    return {"text": result[0]}

# Endpoint to determine if a post is well-finished based on user ID and comment creation date
@app.get("/v1/codebase_community/post_finish_status_by_user_comment_date", operation_id="get_post_finish_status_by_user_comment_date", summary="This operation evaluates the completion status of a post based on the user who commented and the date the comment was created. It returns a 'well-finished' status if the post has been closed, and a 'NOT well-finished' status otherwise. The user ID and comment creation date are required as input parameters.")
async def get_post_finish_status_by_user_comment_date(user_id: int = Query(..., description="User ID of the commenter"), creation_date: str = Query(..., description="Creation date of the comment in 'YYYY-MM-DD HH:MM:SS.SSS' format")):
    cursor.execute("SELECT IIF(T2.ClosedDate IS NULL, 'NOT well-finished', 'well-finished') AS result FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T1.UserId = ? AND T1.CreationDate = ?", (user_id, creation_date))
    result = cursor.fetchone()
    if not result:
        return {"result": []}
    return {"result": result[0]}

# Endpoint to get the reputation of a user based on post ID
@app.get("/v1/codebase_community/user_reputation_by_post_id", operation_id="get_user_reputation_by_post_id", summary="Retrieves the reputation score of the user who owns the post identified by the provided post ID. This operation fetches the user's reputation from the database by joining the users and posts tables using the post's owner user ID.")
async def get_user_reputation_by_post_id(post_id: int = Query(..., description="ID of the post")):
    cursor.execute("SELECT T1.Reputation FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE T2.Id = ?", (post_id,))
    result = cursor.fetchone()
    if not result:
        return {"reputation": []}
    return {"reputation": result[0]}

# Endpoint to get the count of users with a specific display name
@app.get("/v1/codebase_community/user_count_by_display_name", operation_id="get_user_count_by_display_name", summary="Retrieves the total number of users who have the specified display name and have authored at least one post.")
async def get_user_count_by_display_name(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE T1.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display name of a user based on vote ID
@app.get("/v1/codebase_community/user_display_name_by_vote_id", operation_id="get_user_display_name_by_vote_id", summary="Retrieves the display name of the user associated with the specified vote ID. This operation fetches the user's display name from the users table by joining it with the votes table using the UserId field and filtering based on the provided vote ID.")
async def get_user_display_name_by_vote_id(vote_id: int = Query(..., description="ID of the vote")):
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.UserId WHERE T2.Id = ?", (vote_id,))
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the count of posts with titles containing a specific phrase
@app.get("/v1/codebase_community/post_count_by_title_phrase", operation_id="get_post_count_by_title_phrase", summary="Retrieves the total number of posts that have a specific phrase in their titles. The phrase is provided as an input parameter, and the count is determined by searching through the post titles in the database.")
async def get_post_count_by_title_phrase(title_phrase: str = Query(..., description="Phrase to search in the post titles")):
    cursor.execute("SELECT COUNT(T1.Id) FROM posts AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.PostId WHERE T1.Title LIKE ?", ('%' + title_phrase + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the badge names of a user based on display name
@app.get("/v1/codebase_community/user_badge_names_by_display_name", operation_id="get_user_badge_names_by_display_name", summary="Retrieves the names of badges associated with a user, identified by their display name. The display name is used to locate the user in the database and subsequently fetch the corresponding badge names.")
async def get_user_badge_names_by_display_name(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T2.Name FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T1.DisplayName = ?", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"badge_names": []}
    return {"badge_names": [row[0] for row in result]}

# Endpoint to get the ratio of posts to votes for a specific user
@app.get("/v1/codebase_community/posts_to_votes_ratio", operation_id="get_posts_to_votes_ratio", summary="Retrieves the ratio of posts to votes for a specific user. This operation calculates the ratio by dividing the total number of posts by the total number of votes made by the user. The user is identified by the provided user_id.")
async def get_posts_to_votes_ratio(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT CAST(COUNT(T2.Id) AS REAL) / COUNT(DISTINCT T1.Id) FROM votes AS T1 INNER JOIN posts AS T2 ON T1.UserId = T2.OwnerUserId WHERE T1.UserId = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the view count of a post by title
@app.get("/v1/codebase_community/post_view_count", operation_id="get_post_view_count", summary="Retrieves the total number of views for a specific post, identified by its title.")
async def get_post_view_count(title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT ViewCount FROM posts WHERE Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"view_count": []}
    return {"view_count": result[0]}

# Endpoint to get the text of comments by score
@app.get("/v1/codebase_community/comment_text_by_score", operation_id="get_comment_text_by_score", summary="Retrieves the text content of comments that have a specified score. The score parameter is used to filter the comments and return only those with the given score.")
async def get_comment_text_by_score(score: int = Query(..., description="Score of the comment")):
    cursor.execute("SELECT Text FROM comments WHERE Score = ?", (score,))
    result = cursor.fetchall()
    if not result:
        return {"text": []}
    return {"text": [row[0] for row in result]}

# Endpoint to get the display name of users by website URL
@app.get("/v1/codebase_community/user_display_name_by_website", operation_id="get_user_display_name_by_website", summary="Retrieves the display name of a user associated with the provided website URL. The input parameter is the website URL of the user, which is used to identify the corresponding user record in the database.")
async def get_user_display_name_by_website(website_url: str = Query(..., description="Website URL of the user")):
    cursor.execute("SELECT DisplayName FROM users WHERE WebsiteUrl = ?", (website_url,))
    result = cursor.fetchall()
    if not result:
        return {"display_name": []}
    return {"display_name": [row[0] for row in result]}

# Endpoint to get the display name of users who made a specific comment
@app.get("/v1/codebase_community/user_display_name_by_comment", operation_id="get_user_display_name_by_comment", summary="Retrieves the display name of the user who authored a specific comment. The comment is identified by its text content, which is provided as an input parameter.")
async def get_user_display_name_by_comment(comment_text: str = Query(..., description="Text of the comment")):
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.UserId WHERE T2.Text = ?", (comment_text,))
    result = cursor.fetchall()
    if not result:
        return {"display_name": []}
    return {"display_name": [row[0] for row in result]}

# Endpoint to get the comments made by a specific user
@app.get("/v1/codebase_community/comments_by_user", operation_id="get_comments_by_user", summary="Retrieves all comments made by a user identified by their display name. The display name is a unique identifier for the user.")
async def get_comments_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T2.Text FROM users AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.UserId WHERE T1.DisplayName = ?", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": [row[0] for row in result]}

# Endpoint to get the display name and reputation of users who posted a specific post
@app.get("/v1/codebase_community/user_info_by_post", operation_id="get_user_info_by_post", summary="Retrieves the display name and reputation of users who authored a post with a specified title. The post title is used to identify the relevant users.")
async def get_user_info_by_post(post_title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T1.DisplayName, T1.Reputation FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE T2.Title = ?", (post_title,))
    result = cursor.fetchall()
    if not result:
        return {"user_info": []}
    return {"user_info": [{"display_name": row[0], "reputation": row[1]} for row in result]}

# Endpoint to get the comments of a specific post
@app.get("/v1/codebase_community/comments_by_post", operation_id="get_comments_by_post", summary="Retrieves the comments associated with a specific post, identified by its title. The operation returns the text content of each comment.")
async def get_comments_by_post(post_title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T1.Text FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T2.Title = ?", (post_title,))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": [row[0] for row in result]}

# Endpoint to get the display names of users with a specific badge
@app.get("/v1/codebase_community/user_display_names_by_badge", operation_id="get_user_display_names_by_badge", summary="Retrieves the display names of up to 10 users who have been awarded a specific badge. The badge name is used to filter the users.")
async def get_user_display_names_by_badge(badge_name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.Name = ? LIMIT 10", (badge_name,))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the sum of scores and website URLs of posts edited by a specific user
@app.get("/v1/codebase_community/sum_scores_website_urls_by_editor", operation_id="get_sum_scores_website_urls_by_editor", summary="Retrieves the total sum of scores and associated website URLs for posts edited by a specific user. The user is identified by their display name.")
async def get_sum_scores_website_urls_by_editor(display_name: str = Query(..., description="Display name of the editor")):
    cursor.execute("SELECT SUM(T1.Score), T2.WebsiteUrl FROM posts AS T1 INNER JOIN users AS T2 ON T1.LastEditorUserId = T2.Id WHERE T2.DisplayName = ? GROUP BY T2.WebsiteUrl", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"sum_scores_website_urls": []}
    return {"sum_scores_website_urls": [{"sum_score": row[0], "website_url": row[1]} for row in result]}

# Endpoint to get the comments of a post with a specific title
@app.get("/v1/codebase_community/comments_by_post_title", operation_id="get_comments_by_post_title", summary="Retrieves all comments associated with a post that has a specified title. The post title is used to identify the relevant post and its corresponding comments.")
async def get_comments_by_post_title(post_title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T2.Comment FROM posts AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.PostId WHERE T1.Title = ?", (post_title,))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": [row[0] for row in result]}

# Endpoint to get the sum of bounty amounts for posts with titles containing a specific keyword
@app.get("/v1/codebase_community/sum_bounty_amounts_by_post_title_keyword", operation_id="get_sum_bounty_amounts_by_post_title_keyword", summary="Retrieves the total bounty amount for posts containing a specified keyword in their titles. The keyword is used to filter the posts, and the sum of their associated bounty amounts is calculated.")
async def get_sum_bounty_amounts_by_post_title_keyword(title_keyword: str = Query(..., description="Keyword to search in post titles")):
    cursor.execute("SELECT SUM(T2.BountyAmount) FROM posts AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.PostId WHERE T1.Title LIKE ?", (f'%{title_keyword}%',))
    result = cursor.fetchone()
    if not result:
        return {"sum_bounty_amount": []}
    return {"sum_bounty_amount": result[0]}

# Endpoint to get the display names and titles of posts with a specific bounty amount and title containing a specific keyword
@app.get("/v1/codebase_community/display_names_titles_by_bounty_amount_title_keyword", operation_id="get_display_names_titles_by_bounty_amount_title_keyword", summary="Retrieve the display names of users and the titles of their posts that have a specified bounty amount and contain a given keyword in the title. This operation filters posts based on the provided bounty amount and keyword, then returns the display names of users who have posted them along with the corresponding post titles.")
async def get_display_names_titles_by_bounty_amount_title_keyword(bounty_amount: int = Query(..., description="Bounty amount"), title_keyword: str = Query(..., description="Keyword to search in post titles")):
    cursor.execute("SELECT T3.DisplayName, T1.Title FROM posts AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.PostId INNER JOIN users AS T3 ON T3.Id = T2.UserId WHERE T2.BountyAmount = ? AND T1.Title LIKE ?", (bounty_amount, f'%{title_keyword}%'))
    result = cursor.fetchall()
    if not result:
        return {"display_names_titles": []}
    return {"display_names_titles": [{"display_name": row[0], "title": row[1]} for row in result]}

# Endpoint to get the average view count, titles, and text of comments for posts with a specific tag
@app.get("/v1/codebase_community/avg_view_count_titles_text_by_tag", operation_id="get_avg_view_count_titles_text_by_tag", summary="Retrieves the average view count, associated post titles, and comment text for posts containing a specified tag. This operation groups the results by post title and comment text.")
async def get_avg_view_count_titles_text_by_tag(tag: str = Query(..., description="Tag to search in posts")):
    cursor.execute("SELECT AVG(T2.ViewCount), T2.Title, T1.Text FROM comments AS T1 INNER JOIN posts AS T2 ON T2.Id = T1.PostId WHERE T2.Tags = ? GROUP BY T2.Title, T1.Text", (tag,))
    result = cursor.fetchall()
    if not result:
        return {"avg_view_count_titles_text": []}
    return {"avg_view_count_titles_text": [{"avg_view_count": row[0], "title": row[1], "text": row[2]} for row in result]}

# Endpoint to get the count of comments by a specific user
@app.get("/v1/codebase_community/comment_count_by_user", operation_id="get_comment_count_by_user", summary="Retrieves the total number of comments made by a specific user. The operation requires the user's unique identifier as input to accurately determine the comment count.")
async def get_comment_count_by_user(user_id: int = Query(..., description="ID of the user")):
    cursor.execute("SELECT COUNT(Id) FROM comments WHERE UserId = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the IDs of users with the maximum reputation
@app.get("/v1/codebase_community/user_ids_with_max_reputation", operation_id="get_user_ids_with_max_reputation", summary="Retrieves the unique identifiers of users who have achieved the highest reputation score within the community. This operation does not return any user details or reputation scores, only the identifiers of the top-ranked users.")
async def get_user_ids_with_max_reputation():
    cursor.execute("SELECT Id FROM users WHERE Reputation = ( SELECT MAX(Reputation) FROM users )")
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the IDs of users with the minimum views
@app.get("/v1/codebase_community/user_ids_with_min_views", operation_id="get_user_ids_with_min_views", summary="Retrieves the unique identifiers of users who have the least number of views. This operation returns a list of user IDs that have the minimum view count in the system.")
async def get_user_ids_with_min_views():
    cursor.execute("SELECT Id FROM users WHERE Views = ( SELECT MIN(Views) FROM users )")
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the count of badges awarded in a specific year and with a specific name
@app.get("/v1/codebase_community/badge_count_by_year_and_name", operation_id="get_badge_count_by_year_and_name", summary="Retrieves the total number of badges awarded in a given year with a specific name. The year should be provided in 'YYYY' format, and the name should correspond to the desired badge. This operation is useful for tracking the distribution of badges over time and by name.")
async def get_badge_count_by_year_and_name(year: str = Query(..., description="Year in 'YYYY' format"), name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT COUNT(Id) FROM badges WHERE STRFTIME('%Y', Date) = ? AND Name = ?", (year, name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users who have more than a specified number of badges
@app.get("/v1/codebase_community/user_count_by_badge_count", operation_id="get_user_count_by_badge_count", summary="Retrieves the total number of users who possess more than the specified minimum number of badges.")
async def get_user_count_by_badge_count(min_badges: int = Query(..., description="Minimum number of badges a user must have")):
    cursor.execute("SELECT COUNT(UserId) FROM ( SELECT UserId, COUNT(Name) AS num FROM badges GROUP BY UserId ) T WHERE T.num > ?", (min_badges,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct badges awarded to users in a specific location with specific badge names
@app.get("/v1/codebase_community/badge_count_by_location_and_names", operation_id="get_badge_count_by_location_and_names", summary="Retrieves the total count of unique badges awarded to users in a specified location, filtered by the provided badge names.")
async def get_badge_count_by_location_and_names(name1: str = Query(..., description="First badge name"), name2: str = Query(..., description="Second badge name"), location: str = Query(..., description="Location of the user")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Id) FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.Name IN (?, ?) AND T2.Location = ?", (name1, name2, location))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user details who commented on a specific post
@app.get("/v1/codebase_community/user_details_by_post_id", operation_id="get_user_details_by_post_id", summary="Retrieves the user details of individuals who have commented on a specific post. The user details include the user's unique identifier and reputation score. The post is identified by its unique ID.")
async def get_user_details_by_post_id(post_id: int = Query(..., description="ID of the post")):
    cursor.execute("SELECT T2.Id, T2.Reputation FROM comments AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.PostId = ?", (post_id,))
    result = cursor.fetchall()
    if not result:
        return {"users": []}
    return {"users": result}

# Endpoint to get user IDs who have posts with a minimum view count and a specific number of distinct post history types
@app.get("/v1/codebase_community/user_ids_by_view_count_and_history_types", operation_id="get_user_ids_by_view_count_and_history_types", summary="Retrieves the user IDs of users who have posts with a minimum view count and a specific number of distinct post history types. The operation filters posts based on the provided minimum view count and identifies users who have a specified number of distinct post history types associated with their posts.")
async def get_user_ids_by_view_count_and_history_types(min_view_count: int = Query(..., description="Minimum view count of the post"), distinct_history_types: int = Query(..., description="Number of distinct post history types")):
    cursor.execute("SELECT T2.UserId FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id WHERE T3.ViewCount >= ? GROUP BY T2.UserId HAVING COUNT(DISTINCT T2.PostHistoryTypeId) = ?", (min_view_count, distinct_history_types))
    result = cursor.fetchall()
    if not result:
        return {"user_ids": []}
    return {"user_ids": [row[0] for row in result]}

# Endpoint to get the most common badge name among users who have commented, limited by a specified number
@app.get("/v1/codebase_community/most_common_badge_name", operation_id="get_most_common_badge_name", summary="Retrieves the most frequently awarded badge name among users who have commented on the platform. The number of results returned can be limited by specifying a value for the limit parameter.")
async def get_most_common_badge_name(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Name FROM badges AS T1 INNER JOIN comments AS T2 ON T1.UserId = T2.UserId GROUP BY T2.UserId ORDER BY COUNT(T2.UserId) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"badge_names": []}
    return {"badge_names": [row[0] for row in result]}

# Endpoint to get the count of badges awarded to users in a specific location with a specific badge name
@app.get("/v1/codebase_community/badge_count_by_location_and_name", operation_id="get_badge_count_by_location_and_name", summary="Retrieves the total number of badges awarded to users in a specified location with a specified badge name. The operation requires the location of the users and the name of the badge as input parameters.")
async def get_badge_count_by_location_and_name(location: str = Query(..., description="Location of the user"), name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT COUNT(T1.Id) FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T2.Location = ? AND T1.Name = ?", (location, name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage difference in badges awarded between two years for a specific badge name
@app.get("/v1/codebase_community/badge_percentage_difference_by_years", operation_id="get_badge_percentage_difference_by_years", summary="Retrieve the percentage difference in the number of badges awarded for a specific badge between two given years. The calculation is based on the total count of badges awarded in each year and the total count of badges awarded for the specific badge in each year.")
async def get_badge_percentage_difference_by_years(year1: str = Query(..., description="First year in 'YYYY' format"), year2: str = Query(..., description="Second year in 'YYYY' format"), name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT CAST(SUM(IIF(STRFTIME('%Y', Date) = ?, 1, 0)) AS REAL) * 100 / COUNT(Id) - CAST(SUM(IIF(STRFTIME('%Y', Date) = ?, 1, 0)) AS REAL) * 100 / COUNT(Id) FROM badges WHERE Name = ?", (year1, year2, name))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the post history type ID and the number of distinct users who commented on a specific post
@app.get("/v1/codebase_community/post_history_type_and_user_count", operation_id="get_post_history_type_and_user_count", summary="Retrieves the post history type ID and the count of unique users who have commented on a specific post. The post is identified by its unique ID.")
async def get_post_history_type_and_user_count(post_id: int = Query(..., description="ID of the post")):
    cursor.execute("SELECT T1.PostHistoryTypeId, (SELECT COUNT(DISTINCT UserId) FROM comments WHERE PostId = ?) AS NumberOfUsers FROM postHistory AS T1 WHERE T1.PostId = ?", (post_id, post_id))
    result = cursor.fetchall()
    if not result:
        return {"post_history": []}
    return {"post_history": result}

# Endpoint to get the score and link type ID of a post based on post ID
@app.get("/v1/codebase_community/post_score_link_type", operation_id="get_post_score_link_type", summary="Retrieves the score and associated link type ID of a specific post, identified by its unique post ID.")
async def get_post_score_link_type(post_id: int = Query(..., description="ID of the post")):
    cursor.execute("SELECT T1.Score, T2.LinkTypeId FROM posts AS T1 INNER JOIN postLinks AS T2 ON T1.Id = T2.PostId WHERE T2.PostId = ?", (post_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get post history based on minimum score
@app.get("/v1/codebase_community/post_history_min_score", operation_id="get_post_history_min_score", summary="Retrieves a list of post histories for posts with a score greater than the specified minimum score. The response includes the post ID and the user ID associated with each post history.")
async def get_post_history_min_score(min_score: int = Query(..., description="Minimum score of the post")):
    cursor.execute("SELECT PostId, UserId FROM postHistory WHERE PostId IN ( SELECT Id FROM posts WHERE Score > ? )", (min_score,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the sum of distinct favorite counts for posts edited by a user in a specific year
@app.get("/v1/codebase_community/sum_favorite_counts", operation_id="get_sum_favorite_counts", summary="Retrieves the total of unique favorite counts for posts edited by a specific user during a given year. The user is identified by their unique ID, and the year is specified in 'YYYY' format.")
async def get_sum_favorite_counts(user_id: int = Query(..., description="ID of the user"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(DISTINCT FavoriteCount) FROM posts WHERE Id IN ( SELECT PostId FROM postHistory WHERE UserId = ? AND STRFTIME('%Y', CreationDate) = ? )", (user_id, year))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the average upvotes and age of users with more than a specified number of posts
@app.get("/v1/codebase_community/avg_upvotes_age", operation_id="get_avg_upvotes_age", summary="Retrieves the average number of upvotes and the average age of users who have published more than a specified minimum number of posts. This operation provides insights into the engagement and demographics of active community members.")
async def get_avg_upvotes_age(min_posts: int = Query(..., description="Minimum number of posts")):
    cursor.execute("SELECT AVG(T1.UpVotes), AVG(T1.Age) FROM users AS T1 INNER JOIN ( SELECT OwnerUserId, COUNT(*) AS post_count FROM posts GROUP BY OwnerUserId HAVING post_count > ?) AS T2 ON T1.Id = T2.OwnerUserId", (min_posts,))
    result = cursor.fetchone()
    if not result:
        return {"avg_upvotes": [], "avg_age": []}
    return {"avg_upvotes": result[0], "avg_age": result[1]}

# Endpoint to get the names of badges awarded on a specific date
@app.get("/v1/codebase_community/badge_names_by_date", operation_id="get_badge_names_by_date", summary="Retrieve the names of badges awarded on a specific date. The date must be provided in the 'YYYY-MM-DD HH:MM:SS.SSS' format. This operation returns a list of badge names that were awarded on the specified date.")
async def get_badge_names_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD HH:MM:SS.SSS' format")):
    cursor.execute("SELECT Name FROM badges WHERE Date = ?", (date,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of comments based on minimum score
@app.get("/v1/codebase_community/comment_count_min_score", operation_id="get_comment_count_min_score", summary="Retrieves the total number of comments that have a score greater than the specified minimum score. The input parameter determines the minimum score threshold for the count.")
async def get_comment_count_min_score(min_score: int = Query(..., description="Minimum score of the comment")):
    cursor.execute("SELECT COUNT(id) FROM comments WHERE score > ?", (min_score,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the text of comments created on a specific date
@app.get("/v1/codebase_community/comment_text_by_date", operation_id="get_comment_text_by_date", summary="Retrieves the text content of comments created on a specified date. The date should be provided in the 'YYYY-MM-DD HH:MM:SS.SSS' format.")
async def get_comment_text_by_date(creation_date: str = Query(..., description="Creation date in 'YYYY-MM-DD HH:MM:SS.SSS' format")):
    cursor.execute("SELECT Text FROM comments WHERE CreationDate = ?", (creation_date,))
    result = cursor.fetchall()
    if not result:
        return {"texts": []}
    return {"texts": [row[0] for row in result]}

# Endpoint to get the count of posts based on score
@app.get("/v1/codebase_community/post_count_by_score", operation_id="get_post_count_by_score", summary="Retrieves the total number of posts that have a specified score. The score is a measure of the post's popularity or relevance.")
async def get_post_count_by_score(score: int = Query(..., description="Score of the post")):
    cursor.execute("SELECT COUNT(id) FROM posts WHERE Score = ?", (score,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the badge of the user with the highest reputation
@app.get("/v1/codebase_community/top_reputation_user_badge", operation_id="get_top_reputation_user_badge", summary="Retrieves the name of the badge associated with the user who has the highest reputation in the community. The operation ranks users by their reputation scores and identifies the top-ranked user. It then fetches the name of the badge linked to this user.")
async def get_top_reputation_user_badge():
    cursor.execute("SELECT T2.name FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId ORDER BY T1.Reputation DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the reputation of users based on badge date
@app.get("/v1/codebase_community/user_reputation_by_badge_date", operation_id="get_user_reputation_by_badge_date", summary="Retrieves the reputation scores of users who were awarded a badge on a specific date. The date of badge awarding is provided as input in the 'YYYY-MM-DD HH:MM:SS.SSS' format.")
async def get_user_reputation_by_badge_date(badge_date: str = Query(..., description="Date when the badge was awarded in 'YYYY-MM-DD HH:MM:SS.SSS' format")):
    cursor.execute("SELECT T1.Reputation FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.Date = ?", (badge_date,))
    result = cursor.fetchall()
    if not result:
        return {"reputations": []}
    return {"reputations": [row[0] for row in result]}

# Endpoint to get badge dates based on user location
@app.get("/v1/codebase_community/badge_dates_by_user_location", operation_id="get_badge_dates_by_user_location", summary="Retrieves the dates when badges were awarded to users from a specified location. The location parameter is used to filter the users and return the corresponding badge award dates.")
async def get_badge_dates_by_user_location(location: str = Query(..., description="Location of the user")):
    cursor.execute("SELECT T2.Date FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T1.Location = ?", (location,))
    result = cursor.fetchall()
    if not result:
        return {"badge_dates": []}
    return {"badge_dates": [row[0] for row in result]}

# Endpoint to get the percentage of users with a specific badge
@app.get("/v1/codebase_community/percentage_users_with_badge", operation_id="get_percentage_users_with_badge", summary="Retrieves the percentage of users who possess a specific badge. The calculation is based on the total number of users and the count of users with the specified badge. The badge name is provided as an input parameter.")
async def get_percentage_users_with_badge(badge_name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT CAST(COUNT(T1.Id) AS REAL) * 100 / (SELECT COUNT(Id) FROM users) FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.Name = ?", (badge_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of users with a specific badge within an age range
@app.get("/v1/codebase_community/percentage_users_with_badge_age_range", operation_id="get_percentage_users_with_badge_age_range", summary="Retrieves the percentage of users who possess a specific badge and are within a defined age range. The calculation is based on the total number of users with the given badge and the count of users within the specified age range.")
async def get_percentage_users_with_badge_age_range(badge_name: str = Query(..., description="Name of the badge"), min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Age BETWEEN ? AND ?, 1, 0)) AS REAL) * 100 / COUNT(T1.Id) FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.Name = ?", (min_age, max_age, badge_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get comment scores based on creation date
@app.get("/v1/codebase_community/comment_scores_by_creation_date", operation_id="get_comment_scores_by_creation_date", summary="Retrieve the scores of comments created on a specific date. The operation filters comments based on the provided creation date and returns their respective scores. The input parameter specifies the creation date in 'YYYY-MM-DD HH:MM:SS.SSS' format.")
async def get_comment_scores_by_creation_date(creation_date: str = Query(..., description="Creation date of the comment in 'YYYY-MM-DD HH:MM:SS.SSS' format")):
    cursor.execute("SELECT T1.Score FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T1.CreationDate = ?", (creation_date,))
    result = cursor.fetchall()
    if not result:
        return {"scores": []}
    return {"scores": [row[0] for row in result]}

# Endpoint to get comment texts based on creation date
@app.get("/v1/codebase_community/comment_texts_by_creation_date", operation_id="get_comment_texts_by_creation_date", summary="Retrieve the texts of comments created on a specific date. The operation filters comments based on the provided creation date and returns their respective texts. The input parameter specifies the creation date in 'YYYY-MM-DD HH:MM:SS.SSS' format.")
async def get_comment_texts_by_creation_date(creation_date: str = Query(..., description="Creation date of the comment in 'YYYY-MM-DD HH:MM:SS.SSS' format")):
    cursor.execute("SELECT T1.Text FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T1.CreationDate = ?", (creation_date,))
    result = cursor.fetchall()
    if not result:
        return {"texts": []}
    return {"texts": [row[0] for row in result]}

# Endpoint to get user ages based on location
@app.get("/v1/codebase_community/user_ages_by_location", operation_id="get_user_ages_by_location", summary="Retrieves the ages of users who have earned badges and are located in the specified location. This operation provides a snapshot of the age distribution of active users in a particular area.")
async def get_user_ages_by_location(location: str = Query(..., description="Location of the user")):
    cursor.execute("SELECT T1.Age FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T1.Location = ?", (location,))
    result = cursor.fetchall()
    if not result:
        return {"ages": []}
    return {"ages": [row[0] for row in result]}

# Endpoint to get the count of users with a specific badge and age range
@app.get("/v1/codebase_community/count_users_with_badge_age_range", operation_id="get_count_users_with_badge_age_range", summary="Retrieves the number of users who possess a specific badge and are within a defined age range. The operation considers the provided badge name and age range to calculate the count.")
async def get_count_users_with_badge_age_range(badge_name: str = Query(..., description="Name of the badge"), min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.Name = ? AND T1.Age BETWEEN ? AND ?", (badge_name, min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get user views based on badge date
@app.get("/v1/codebase_community/user_views_by_badge_date", operation_id="get_user_views_by_badge_date", summary="Retrieves the total views of users who were awarded a badge on a specific date. The date must be provided in the 'YYYY-MM-DD HH:MM:SS.SSS' format.")
async def get_user_views_by_badge_date(badge_date: str = Query(..., description="Date when the badge was awarded in 'YYYY-MM-DD HH:MM:SS.SSS' format")):
    cursor.execute("SELECT T1.Views FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.Date = ?", (badge_date,))
    result = cursor.fetchall()
    if not result:
        return {"views": []}
    return {"views": [row[0] for row in result]}

# Endpoint to get the names of badges for users with the minimum reputation
@app.get("/v1/codebase_community/badges_min_reputation", operation_id="get_badges_min_reputation", summary="Retrieves the names of badges associated with users who have the lowest reputation in the community. This operation provides a quick overview of the badges held by users with the least reputation points.")
async def get_badges_min_reputation():
    cursor.execute("SELECT T2.Name FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T1.Reputation = (SELECT MIN(Reputation) FROM users)")
    result = cursor.fetchall()
    if not result:
        return {"badges": []}
    return {"badges": [row[0] for row in result]}

# Endpoint to get the count of users with a specific badge and age greater than a specified value
@app.get("/v1/codebase_community/user_count_age_badge", operation_id="get_user_count_age_badge", summary="Retrieves the number of users who have a specified badge and are older than a given age. The age and badge name are provided as input parameters.")
async def get_user_count_age_badge(min_age: int = Query(..., description="Minimum age of the user"), badge_name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T1.Age > ? AND T2.Name = ?", (min_age, badge_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display name of a user by user ID
@app.get("/v1/codebase_community/user_display_name_by_id", operation_id="get_user_display_name_by_id", summary="Retrieves the display name associated with the provided user ID. This operation allows you to look up a user's display name using their unique identifier.")
async def get_user_display_name_by_id(user_id: int = Query(..., description="ID of the user")):
    cursor.execute("SELECT DisplayName FROM users WHERE Id = ?", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the count of users from a specific location
@app.get("/v1/codebase_community/user_count_by_location", operation_id="get_user_count_by_location", summary="Retrieves the total number of users associated with a given location. The location is specified as an input parameter, allowing for a targeted count of users from a particular area.")
async def get_user_count_by_location(location: str = Query(..., description="Location of the user")):
    cursor.execute("SELECT COUNT(Id) FROM users WHERE Location = ?", (location,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of votes created in a specific year
@app.get("/v1/codebase_community/vote_count_by_year", operation_id="get_vote_count_by_year", summary="Retrieves the total number of votes created in a specified year. The year must be provided in the 'YYYY' format.")
async def get_vote_count_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(id) FROM votes WHERE STRFTIME('%Y', CreationDate) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of users within a specific age range
@app.get("/v1/codebase_community/user_count_by_age_range", operation_id="get_user_count_by_age_range", summary="Retrieves the total number of users whose ages fall within the specified range. The range is defined by a minimum and maximum age, both of which are inclusive.")
async def get_user_count_by_age_range(min_age: int = Query(..., description="Minimum age of the user"), max_age: int = Query(..., description="Maximum age of the user")):
    cursor.execute("SELECT COUNT(id) FROM users WHERE Age BETWEEN ? AND ?", (min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ID and display name of users with the maximum views
@app.get("/v1/codebase_community/users_max_views", operation_id="get_users_max_views", summary="Retrieves the unique identifier and display name of users who have the highest view count in the community.")
async def get_users_max_views():
    cursor.execute("SELECT Id, DisplayName FROM users WHERE Views = ( SELECT MAX(Views) FROM users )")
    result = cursor.fetchall()
    if not result:
        return {"users": []}
    return {"users": [{"id": row[0], "display_name": row[1]} for row in result]}

# Endpoint to get the ratio of votes created in two specific years
@app.get("/v1/codebase_community/vote_ratio_by_years", operation_id="get_vote_ratio_by_years", summary="Retrieves the ratio of votes created in two specified years. The operation calculates the proportion of votes from the first year against the total votes from the second year. The input parameters define the two years to compare.")
async def get_vote_ratio_by_years(year1: str = Query(..., description="First year in 'YYYY' format"), year2: str = Query(..., description="Second year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(IIF(STRFTIME('%Y', CreationDate) = ?, 1, 0)) AS REAL) / SUM(IIF(STRFTIME('%Y', CreationDate) = ?, 1, 0)) FROM votes", (year1, year2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the tags of posts edited by a user with a specific display name
@app.get("/v1/codebase_community/post_tags_by_user", operation_id="get_post_tags_by_user", summary="Retrieve the tags associated with posts that a user with a specific display name has edited. The display name of the user is required as an input parameter.")
async def get_post_tags_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T3.Tags FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id WHERE T1.DisplayName = ?", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"tags": []}
    return {"tags": [row[0] for row in result]}

# Endpoint to get the count of users based on display name and votes
@app.get("/v1/codebase_community/user_count_by_display_name_and_votes", operation_id="get_user_count_by_display_name_and_votes", summary="Retrieves the total number of users who have a specific display name and have received votes on their posts. The display name is used to filter the users.")
async def get_user_count_by_display_name_and_votes(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN votes AS T3 ON T3.PostId = T2.PostId WHERE T1.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the post ID with the highest answer count for a specific user
@app.get("/v1/codebase_community/top_post_by_answer_count", operation_id="get_top_post_by_answer_count", summary="Retrieves the post ID of the user's top post, ranked by the highest number of answers. The user is identified by their display name.")
async def get_top_post_by_answer_count(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T2.PostId FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id WHERE T1.DisplayName = ? ORDER BY T3.AnswerCount DESC LIMIT 1", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"post_id": []}
    return {"post_id": result[0]}

# Endpoint to get the display name with the highest view count between two users
@app.get("/v1/codebase_community/top_display_name_by_view_count", operation_id="get_top_display_name_by_view_count", summary="Retrieves the display name of the user with the highest cumulative view count between two specified users. The operation compares the total view count of posts made by the two users and returns the display name of the user with the highest sum.")
async def get_top_display_name_by_view_count(display_name1: str = Query(..., description="First display name of the user"), display_name2: str = Query(..., description="Second display name of the user")):
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id WHERE T1.DisplayName = ? OR T1.DisplayName = ? GROUP BY T1.DisplayName ORDER BY SUM(T3.ViewCount) DESC LIMIT 1", (display_name1, display_name2))
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the count of users with a specific display name and more than a certain number of votes
@app.get("/v1/codebase_community/user_count_by_display_name_and_vote_count", operation_id="get_user_count_by_display_name_and_vote_count", summary="Get the count of users with a specific display name and more than a certain number of votes")
async def get_user_count_by_display_name_and_vote_count(display_name: str = Query(..., description="Display name of the user"), min_votes: int = Query(..., description="Minimum number of votes")):
    cursor.execute("SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id INNER JOIN votes AS T4 ON T4.PostId = T3.Id WHERE T1.DisplayName = ? GROUP BY T2.PostId, T4.Id HAVING COUNT(T4.Id) > ?", (display_name, min_votes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of comments with a specific user display name and score less than a certain value
@app.get("/v1/codebase_community/comment_count_by_display_name_and_score", operation_id="get_comment_count_by_display_name_and_score", summary="Retrieves the total number of comments authored by a specific user with a score less than a given value. The user is identified by their display name, and the score threshold is defined by the max_score parameter. This operation does not return the comments themselves, but rather the count of comments that meet the specified criteria.")
async def get_comment_count_by_display_name_and_score(display_name: str = Query(..., description="Display name of the user"), max_score: int = Query(..., description="Maximum score of the comment")):
    cursor.execute("SELECT COUNT(T3.Id) FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId INNER JOIN comments AS T3 ON T2.Id = T3.PostId WHERE T1.DisplayName = ? AND T3.Score < ?", (display_name, max_score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the tags of posts with a specific user display name and comment count
@app.get("/v1/codebase_community/post_tags_by_display_name_and_comment_count", operation_id="get_post_tags_by_display_name_and_comment_count", summary="Retrieve the tags associated with posts authored by a user with a specific display name and a specified comment count. This operation allows you to filter posts based on the author's display name and the number of comments they have received, providing a targeted list of tags for further analysis or categorization.")
async def get_post_tags_by_display_name_and_comment_count(display_name: str = Query(..., description="Display name of the user"), comment_count: int = Query(..., description="Comment count of the post")):
    cursor.execute("SELECT T3.Tags FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T3.Id = T2.PostId WHERE T1.DisplayName = ? AND T3.CommentCount = ?", (display_name, comment_count))
    result = cursor.fetchall()
    if not result:
        return {"tags": []}
    return {"tags": [row[0] for row in result]}

# Endpoint to get the display names of users with a specific badge name
@app.get("/v1/codebase_community/user_display_names_by_badge_name", operation_id="get_user_display_names_by_badge_name", summary="Retrieves the display names of users who have been awarded a specific badge. The badge name is provided as an input parameter.")
async def get_user_display_names_by_badge_name(badge_name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.`Name` = ?", (badge_name,))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the percentage of posts with a specific tag name for a specific user
@app.get("/v1/codebase_community/percentage_posts_by_tag_and_display_name", operation_id="get_percentage_posts_by_tag_and_display_name", summary="Retrieves the percentage of posts associated with a specific tag for a given user. This operation calculates the proportion of posts with the provided tag name out of the total posts made by the specified user.")
async def get_percentage_posts_by_tag_and_display_name(tag_name: str = Query(..., description="Name of the tag"), display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.TagName = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.Id) FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN tags AS T3 ON T3.ExcerptPostId = T2.PostId WHERE T1.DisplayName = ?", (tag_name, display_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in view counts between two users
@app.get("/v1/codebase_community/view_count_difference_between_users", operation_id="get_view_count_difference_between_users", summary="Retrieves the difference in total view counts for posts authored by two specified users. The operation compares the view counts of posts written by the first user with those of the second user, and returns the net difference.")
async def get_view_count_difference_between_users(display_name1: str = Query(..., description="First display name of the user"), display_name2: str = Query(..., description="Second display name of the user")):
    cursor.execute("SELECT SUM(IIF(T1.DisplayName = ?, T3.ViewCount, 0)) - SUM(IIF(T1.DisplayName = ?, T3.ViewCount, 0)) AS diff FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T3.Id = T2.PostId", (display_name1, display_name2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of badges based on name and year
@app.get("/v1/codebase_community/badges_count_by_name_and_year", operation_id="get_badges_count", summary="Retrieves the total count of badges with a specified name and year. The name and year are provided as input parameters to filter the badge records and calculate the count.")
async def get_badges_count(name: str = Query(..., description="Name of the badge"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(Id) FROM badges WHERE Name = ? AND STRFTIME('%Y', Date) = ?", (name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of post history entries by creation date
@app.get("/v1/codebase_community/post_history_count_by_creation_date", operation_id="get_post_history_count", summary="Retrieves the total number of post history entries created on a specified date. The date should be provided in 'YYYY-MM-DD' format.")
async def get_post_history_count(creation_date: str = Query(..., description="Creation date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(id) FROM postHistory WHERE date(CreationDate) = ?", (creation_date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display name and age of users with the maximum views
@app.get("/v1/codebase_community/users_with_max_views", operation_id="get_users_with_max_views", summary="Retrieves the display names and ages of users who have accumulated the highest number of views. This operation provides a snapshot of the most popular users based on view count.")
async def get_users_with_max_views():
    cursor.execute("SELECT DisplayName, Age FROM users WHERE Views = ( SELECT MAX(Views) FROM users )")
    result = cursor.fetchall()
    if not result:
        return {"users": []}
    return {"users": result}

# Endpoint to get the last edit date and editor user ID of posts by title
@app.get("/v1/codebase_community/posts_last_edit_by_title", operation_id="get_posts_last_edit", summary="Retrieves the date and user ID of the most recent edit for posts that match the specified title.")
async def get_posts_last_edit(title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT LastEditDate, LastEditorUserId FROM posts WHERE Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"posts": []}
    return {"posts": result}

# Endpoint to get the count of comments based on user ID and score
@app.get("/v1/codebase_community/comments_count_by_user_and_score", operation_id="get_comments_count", summary="Retrieves the total number of comments made by a particular user that have a score below a specified threshold. The user is identified by their unique ID, and the score threshold is a numerical value.")
async def get_comments_count(user_id: int = Query(..., description="User ID"), score: int = Query(..., description="Score threshold")):
    cursor.execute("SELECT COUNT(Id) FROM comments WHERE UserId = ? AND Score < ?", (user_id, score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get post titles and comment user display names based on post score
@app.get("/v1/codebase_community/posts_comments_by_score", operation_id="get_posts_comments_by_score", summary="Retrieves the titles of posts and the display names of users who commented on those posts, where the post score surpasses a specified threshold. The threshold is defined by the 'score' input parameter.")
async def get_posts_comments_by_score(score: int = Query(..., description="Score threshold")):
    cursor.execute("SELECT T1.Title, T2.UserDisplayName FROM posts AS T1 INNER JOIN comments AS T2 ON T2.PostId = T2.Id WHERE T1.Score > ?", (score,))
    result = cursor.fetchall()
    if not result:
        return {"posts_comments": []}
    return {"posts_comments": result}

# Endpoint to get badge names based on year and user location
@app.get("/v1/codebase_community/badges_by_year_and_location", operation_id="get_badges_by_year_and_location", summary="Retrieves the names of badges awarded to users in a specific location during a given year. The operation filters badges based on the provided year and location parameters.")
async def get_badges_by_year_and_location(year: str = Query(..., description="Year in 'YYYY' format"), location: str = Query(..., description="Location of the user")):
    cursor.execute("SELECT T2.Name FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE STRFTIME('%Y', T2.Date) = ? AND T1.Location = ?", (year, location))
    result = cursor.fetchall()
    if not result:
        return {"badges": []}
    return {"badges": result}

# Endpoint to get user display names and website URLs based on favorite count
@app.get("/v1/codebase_community/users_by_favorite_count", operation_id="get_users_by_favorite_count", summary="Retrieve the display names and website URLs of users who have authored posts with a favorite count exceeding the provided threshold. This operation allows you to filter users based on the popularity of their posts, as indicated by the favorite count.")
async def get_users_by_favorite_count(favorite_count: int = Query(..., description="Favorite count threshold")):
    cursor.execute("SELECT T1.DisplayName, T1.WebsiteUrl FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE T2.FavoriteCount > ?", (favorite_count,))
    result = cursor.fetchall()
    if not result:
        return {"users": []}
    return {"users": result}

# Endpoint to get post history IDs and last edit dates based on post title
@app.get("/v1/codebase_community/post_history_by_title", operation_id="get_post_history_by_title", summary="Retrieves the unique identifiers and most recent edit dates of post history entries associated with a post bearing a specific title. The title of the post is provided as an input parameter.")
async def get_post_history_by_title(title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T1.Id, T2.LastEditDate FROM postHistory AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T2.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"post_history": []}
    return {"post_history": result}

# Endpoint to get user last access dates and locations based on badge name
@app.get("/v1/codebase_community/users_by_badge_name", operation_id="get_users_by_badge_name", summary="Retrieves the most recent access dates and locations of users who possess a specific badge. The badge name is used to filter the users.")
async def get_users_by_badge_name(badge_name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT T1.LastAccessDate, T1.Location FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.Name = ?", (badge_name,))
    result = cursor.fetchall()
    if not result:
        return {"users": []}
    return {"users": result}

# Endpoint to get related post titles based on a given post title
@app.get("/v1/codebase_community/related_post_titles", operation_id="get_related_post_titles", summary="Retrieves a list of titles for posts related to a specified post title. The operation identifies related posts by examining linkages between posts and returns the titles of those posts.")
async def get_related_post_titles(post_title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T3.Title FROM postLinks AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id INNER JOIN posts AS T3 ON T1.RelatedPostId = T3.Id WHERE T2.Title = ?", (post_title,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get post IDs and badge names for a user in a specific year
@app.get("/v1/codebase_community/user_posts_badges_by_year", operation_id="get_user_posts_badges_by_year", summary="Retrieves the post IDs and associated badge names for a specific user, filtered by a given year. The user is identified by their display name, and the year is provided in 'YYYY' format. The data returned reflects the user's post history and badge awards within the specified year.")
async def get_user_posts_badges_by_year(user_display_name: str = Query(..., description="Display name of the user"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.PostId, T2.Name FROM postHistory AS T1 INNER JOIN badges AS T2 ON T1.UserId = T2.UserId WHERE T1.UserDisplayName = ? AND STRFTIME('%Y', T1.CreationDate) = ? AND STRFTIME('%Y', T2.Date) = ?", (user_display_name, year, year))
    result = cursor.fetchall()
    if not result:
        return {"posts_badges": []}
    return {"posts_badges": [{"post_id": row[0], "badge_name": row[1]} for row in result]}

# Endpoint to get the display name of the user with the highest viewed post
@app.get("/v1/codebase_community/top_viewed_post_user", operation_id="get_top_viewed_post_user", summary="Retrieves the display name of the user who authored the post with the highest view count. This operation provides a quick way to identify the most popular content creator in the community.")
async def get_top_viewed_post_user():
    cursor.execute("SELECT DisplayName FROM users WHERE Id = ( SELECT OwnerUserId FROM posts ORDER BY ViewCount DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get user display names and locations based on a tag name
@app.get("/v1/codebase_community/user_info_by_tag", operation_id="get_user_info_by_tag", summary="Retrieves the display names and locations of users who have posted content associated with the specified tag. The tag name is used to identify relevant posts and their respective authors.")
async def get_user_info_by_tag(tag_name: str = Query(..., description="Name of the tag")):
    cursor.execute("SELECT T3.DisplayName, T3.Location FROM tags AS T1 INNER JOIN posts AS T2 ON T1.ExcerptPostId = T2.Id INNER JOIN users AS T3 ON T3.Id = T2.OwnerUserId WHERE T1.TagName = ?", (tag_name,))
    result = cursor.fetchall()
    if not result:
        return {"user_info": []}
    return {"user_info": [{"display_name": row[0], "location": row[1]} for row in result]}

# Endpoint to get related post titles and link types based on a given post title
@app.get("/v1/codebase_community/related_post_titles_link_types", operation_id="get_related_post_titles_link_types", summary="Retrieves the titles of posts related to a specified post, along with the corresponding link types. The operation uses the provided post title to search for related posts and their associated link types.")
async def get_related_post_titles_link_types(post_title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T3.Title, T2.LinkTypeId FROM posts AS T1 INNER JOIN postLinks AS T2 ON T1.Id = T2.PostId INNER JOIN posts AS T3 ON T2.RelatedPostId = T3.Id WHERE T1.Title = ?", (post_title,))
    result = cursor.fetchall()
    if not result:
        return {"related_posts": []}
    return {"related_posts": [{"title": row[0], "link_type_id": row[1]} for row in result]}

# Endpoint to get the display name of the user with the highest scored post that has a parent
@app.get("/v1/codebase_community/top_scored_post_user", operation_id="get_top_scored_post_user", summary="Retrieves the display name of the user who authored the highest-scored post with a parent post. The score of a post is determined by the number of upvotes minus the number of downvotes it has received.")
async def get_top_scored_post_user():
    cursor.execute("SELECT DisplayName FROM users WHERE Id = ( SELECT OwnerUserId FROM posts WHERE ParentId IS NOT NULL ORDER BY Score DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the display name and website URL of the user with the highest bounty amount for a specific vote type
@app.get("/v1/codebase_community/top_bounty_user", operation_id="get_top_bounty_user", summary="Retrieves the display name and website URL of the user who has the highest bounty amount for a specified vote type. The vote type is identified by its unique ID.")
async def get_top_bounty_user(vote_type_id: int = Query(..., description="Vote type ID")):
    cursor.execute("SELECT DisplayName, WebsiteUrl FROM users WHERE Id = ( SELECT UserId FROM votes WHERE VoteTypeId = ? ORDER BY BountyAmount DESC LIMIT 1 )", (vote_type_id,))
    result = cursor.fetchone()
    if not result:
        return {"user_info": []}
    return {"user_info": {"display_name": result[0], "website_url": result[1]}}

# Endpoint to get the top 5 post titles by view count
@app.get("/v1/codebase_community/top_viewed_posts", operation_id="get_top_viewed_posts", summary="Retrieves the titles of the top five most viewed posts in the community. The posts are ranked by their view count in descending order.")
async def get_top_viewed_posts():
    cursor.execute("SELECT Title FROM posts ORDER BY ViewCount DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the count of tags within a specific range
@app.get("/v1/codebase_community/tag_count_range", operation_id="get_tag_count_range", summary="Retrieves the total number of tags that have a count value within the specified range. The range is defined by the minimum and maximum count values provided as input parameters.")
async def get_tag_count_range(min_count: int = Query(..., description="Minimum count"), max_count: int = Query(..., description="Maximum count")):
    cursor.execute("SELECT COUNT(Id) FROM tags WHERE Count BETWEEN ? AND ?", (min_count, max_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the user ID of the post with the maximum favorite count
@app.get("/v1/codebase_community/max_favorite_post_user", operation_id="get_max_favorite_post_user", summary="Retrieves the unique identifier of the user who owns the post with the highest number of favorites in the community.")
async def get_max_favorite_post_user():
    cursor.execute("SELECT OwnerUserId FROM posts WHERE FavoriteCount = ( SELECT MAX(FavoriteCount) FROM posts )")
    result = cursor.fetchone()
    if not result:
        return {"user_id": []}
    return {"user_id": result[0]}

# Endpoint to get the age of the user with the highest reputation
@app.get("/v1/codebase_community/user_age_highest_reputation", operation_id="get_user_age_highest_reputation", summary="Retrieves the age of the user who has the highest reputation in the community. This operation does not require any input parameters and returns the age of the top-ranked user based on their reputation score.")
async def get_user_age_highest_reputation():
    cursor.execute("SELECT Age FROM users WHERE Reputation = ( SELECT MAX(Reputation) FROM users )")
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the count of posts with a specific bounty amount and creation year
@app.get("/v1/codebase_community/count_posts_bounty_amount_year", operation_id="get_count_posts_bounty_amount_year", summary="Retrieves the total number of posts that have a specified bounty amount and were created in a given year. The bounty amount and year are provided as input parameters.")
async def get_count_posts_bounty_amount_year(bounty_amount: int = Query(..., description="Bounty amount"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.Id) FROM posts AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.PostId WHERE T2.BountyAmount = ? AND STRFTIME('%Y', T2.CreationDate) = ?", (bounty_amount, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ID of the youngest user
@app.get("/v1/codebase_community/user_id_youngest", operation_id="get_user_id_youngest", summary="Retrieves the unique identifier of the user with the lowest age value in the users table.")
async def get_user_id_youngest():
    cursor.execute("SELECT Id FROM users WHERE Age = ( SELECT MIN(Age) FROM users )")
    result = cursor.fetchone()
    if not result:
        return {"id": []}
    return {"id": result[0]}

# Endpoint to get the sum of scores for posts with a specific last activity date
@app.get("/v1/codebase_community/sum_scores_last_activity_date", operation_id="get_sum_scores_last_activity_date", summary="Get the sum of scores for posts with a specific last activity date")
async def get_sum_scores_last_activity_date(last_activity_date: str = Query(..., description="Last activity date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT SUM(Score) FROM posts WHERE LastActivityDate LIKE ?", (last_activity_date,))
    result = cursor.fetchone()
    if not result:
        return {"sum_score": []}
    return {"sum_score": result[0]}

# Endpoint to get the average monthly count of post links with a specific answer count and creation year
@app.get("/v1/codebase_community/avg_monthly_post_links_answer_count_year", operation_id="get_avg_monthly_post_links_answer_count_year", summary="Retrieves the average monthly count of post links associated with posts that have an answer count equal to or less than the specified value, and were created in the provided year. The result is calculated by dividing the total count of post links by 12.")
async def get_avg_monthly_post_links_answer_count_year(answer_count: int = Query(..., description="Answer count"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(T1.Id) AS REAL) / 12 FROM postLinks AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T2.AnswerCount <= ? AND STRFTIME('%Y', T1.CreationDate) = ?", (answer_count, year))
    result = cursor.fetchone()
    if not result:
        return {"avg_monthly_count": []}
    return {"avg_monthly_count": result[0]}

# Endpoint to get the post ID with the highest favorite count for a specific user
@app.get("/v1/codebase_community/post_id_highest_favorite_count_user", operation_id="get_post_id_highest_favorite_count_user", summary="Retrieves the post ID of the user's post with the highest number of favorites. The user is identified by the provided user_id.")
async def get_post_id_highest_favorite_count_user(user_id: int = Query(..., description="User ID")):
    cursor.execute("SELECT T2.Id FROM votes AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T1.UserId = ? ORDER BY T2.FavoriteCount DESC LIMIT 1", (user_id,))
    result = cursor.fetchone()
    if not result:
        return {"post_id": []}
    return {"post_id": result[0]}

# Endpoint to get the title of the earliest post
@app.get("/v1/codebase_community/earliest_post_title", operation_id="get_earliest_post_title", summary="Get the title of the earliest post")
async def get_earliest_post_title():
    cursor.execute("SELECT T1.Title FROM posts AS T1 INNER JOIN postLinks AS T2 ON T2.PostId = T1.Id ORDER BY T1.CreationDate LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the display name of the user with the most badges
@app.get("/v1/codebase_community/user_most_badges", operation_id="get_user_most_badges", summary="Retrieves the display name of the user who has earned the highest number of badges. This operation returns the top user based on the count of badges they have acquired.")
async def get_user_most_badges():
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId GROUP BY T1.DisplayName ORDER BY COUNT(T1.Id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the earliest vote creation date for a specific user
@app.get("/v1/codebase_community/earliest_vote_creation_date_user", operation_id="get_earliest_vote_creation_date_user", summary="Retrieves the date of the earliest vote cast by a user identified by their display name. The operation returns the creation date of the first vote associated with the specified user.")
async def get_earliest_vote_creation_date_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T2.CreationDate FROM users AS T1 INNER JOIN votes AS T2 ON T1.Id = T2.UserId WHERE T1.DisplayName = ? ORDER BY T2.CreationDate LIMIT 1", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"creation_date": []}
    return {"creation_date": result[0]}

# Endpoint to get the creation date of the earliest post by the youngest user
@app.get("/v1/codebase_community/earliest_post_creation_date_youngest_user", operation_id="get_earliest_post_creation_date_youngest_user", summary="Get the creation date of the earliest post by the youngest user")
async def get_earliest_post_creation_date_youngest_user():
    cursor.execute("SELECT T2.CreationDate FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE T1.Age IS NOT NULL ORDER BY T1.Age, T2.CreationDate LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"creation_date": []}
    return {"creation_date": result[0]}

# Endpoint to get the display name of a user with a specific badge
@app.get("/v1/codebase_community/user_display_name_by_badge", operation_id="get_user_display_name_by_badge", summary="Retrieves the display name of the most recent user who earned a specific badge. The badge name is required as input to identify the user.")
async def get_user_display_name_by_badge(badge_name: str = Query(..., description="Name of the badge")):
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T2.`Name` = ? ORDER BY T2.Date LIMIT 1", (badge_name,))
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the count of users from a specific location with a minimum number of favorite posts
@app.get("/v1/codebase_community/user_count_by_location_and_favorite_count", operation_id="get_user_count_by_location_and_favorite_count", summary="Retrieves the number of users from a specified location who have published posts with a minimum defined count of favorites. The location and minimum favorite count are provided as input parameters.")
async def get_user_count_by_location_and_favorite_count(location: str = Query(..., description="Location of the user"), min_favorite_count: int = Query(..., description="Minimum number of favorites on posts")):
    cursor.execute("SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE T1.Location = ? AND T2.FavoriteCount >= ?", (location, min_favorite_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average post ID of votes by the oldest user
@app.get("/v1/codebase_community/average_post_id_voted_by_oldest_user", operation_id="get_average_post_id_voted_by_oldest_user", summary="Retrieves the average post ID of votes cast by the oldest user in the community. This operation provides a statistical overview of the voting behavior of the most senior community member.")
async def get_average_post_id_voted_by_oldest_user():
    cursor.execute("SELECT AVG(PostId) FROM votes WHERE UserId IN ( SELECT Id FROM users WHERE Age = ( SELECT MAX(Age) FROM users ) )")
    result = cursor.fetchone()
    if not result:
        return {"average_post_id": []}
    return {"average_post_id": result[0]}

# Endpoint to get the display name of the user with the highest reputation
@app.get("/v1/codebase_community/user_with_highest_reputation", operation_id="get_user_with_highest_reputation", summary="Retrieves the display name of the user who has the highest reputation in the community. This operation does not require any input parameters and returns the display name of the top-ranked user based on their reputation score.")
async def get_user_with_highest_reputation():
    cursor.execute("SELECT DisplayName FROM users WHERE Reputation = ( SELECT MAX(Reputation) FROM users )")
    result = cursor.fetchone()
    if not result:
        return {"display_name": []}
    return {"display_name": result[0]}

# Endpoint to get the count of users with reputation and views above specific thresholds
@app.get("/v1/codebase_community/user_count_by_reputation_and_views", operation_id="get_user_count_by_reputation_and_views", summary="Retrieves the number of users who have a reputation and view count exceeding the provided thresholds. The input parameters specify the minimum reputation and view count required for a user to be included in the count.")
async def get_user_count_by_reputation_and_views(min_reputation: int = Query(..., description="Minimum reputation"), min_views: int = Query(..., description="Minimum views")):
    cursor.execute("SELECT COUNT(id) FROM users WHERE Reputation > ? AND Views > ?", (min_reputation, min_views))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display names of users within a specific age range
@app.get("/v1/codebase_community/user_display_names_by_age_range", operation_id="get_user_display_names_by_age_range", summary="Retrieves the display names of users who fall within the specified age range. The operation accepts a minimum and maximum age as input parameters, and returns a list of user display names for those who are within this age range.")
async def get_user_display_names_by_age_range(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT DisplayName FROM users WHERE Age BETWEEN ? AND ?", (min_age, max_age))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the count of posts by a specific user in a specific year
@app.get("/v1/codebase_community/post_count_by_user_and_year", operation_id="get_post_count_by_user_and_year", summary="Retrieves the total number of posts created by a specific user during a given year. The operation requires the year in 'YYYY' format and the display name of the user as input parameters. The result is a single integer value representing the count of posts.")
async def get_post_count_by_user_and_year(year: str = Query(..., description="Year in 'YYYY' format"), display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT COUNT(T1.Id) FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE STRFTIME('%Y', T2.CreaionDate) = ? AND T1.DisplayName = ?", (year, display_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest scored post
@app.get("/v1/codebase_community/highest_scored_post", operation_id="get_highest_scored_post", summary="Retrieves the post with the highest score from the community. The post is identified by its unique ID and associated with its title. The score is determined by the community's engagement with the post.")
async def get_highest_scored_post():
    cursor.execute("SELECT T1.Id, T2.Title FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId ORDER BY T2.Score DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"post_id": [], "post_title": []}
    return {"post_id": result[0], "post_title": result[1]}

# Endpoint to get the average score of posts by a specific user
@app.get("/v1/codebase_community/average_post_score_by_user", operation_id="get_average_post_score_by_user", summary="Retrieves the average score of posts authored by a specific user, identified by their display name.")
async def get_average_post_score_by_user(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT AVG(T2.Score) FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE T1.DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_score": []}
    return {"average_score": result[0]}

# Endpoint to get user display names based on post creation year and view count
@app.get("/v1/codebase_community/user_display_names_by_post_year_view_count", operation_id="get_user_display_names", summary="Retrieves the display names of users who have posted in a specified year and whose posts have received more views than a given minimum threshold. The year is provided in 'YYYY' format, and the minimum view count is a positive integer.")
async def get_user_display_names(post_year: str = Query(..., description="Year of post creation in 'YYYY' format"), min_view_count: int = Query(..., description="Minimum view count")):
    cursor.execute("SELECT T1.DisplayName FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE STRFTIME('%Y', T2.CreaionDate) = ? AND T2.ViewCount > ?", (post_year, min_view_count))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": [row[0] for row in result]}

# Endpoint to get the top user by favorite count for a specific user creation year
@app.get("/v1/codebase_community/top_user_by_favorite_count_year", operation_id="get_top_user_by_favorite_count", summary="Retrieves the user with the highest number of favorites for posts created in a specific year. The user's unique identifier and display name are returned. The year of user creation is used to filter the results.")
async def get_top_user_by_favorite_count(user_creation_year: str = Query(..., description="Year of user creation in 'YYYY' format")):
    cursor.execute("SELECT T2.OwnerUserId, T1.DisplayName FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId WHERE STRFTIME('%Y', T1.CreationDate) = ? ORDER BY T2.FavoriteCount DESC LIMIT 1", (user_creation_year,))
    result = cursor.fetchone()
    if not result:
        return {"top_user": []}
    return {"top_user": {"owner_user_id": result[0], "display_name": result[1]}}

# Endpoint to get the percentage of users with high reputation who posted in a specific year
@app.get("/v1/codebase_community/percentage_high_reputation_users_posted_year", operation_id="get_percentage_high_reputation_users", summary="Retrieves the percentage of users who have a reputation score exceeding a specified threshold and have posted in a given year. This operation calculates the ratio of users who meet the reputation and post year criteria to the total number of users.")
async def get_percentage_high_reputation_users(post_year: str = Query(..., description="Year of post creation in 'YYYY' format"), min_reputation: int = Query(..., description="Minimum reputation")):
    cursor.execute("SELECT CAST(SUM(IIF(STRFTIME('%Y', T2.CreaionDate) = ? AND T1.Reputation > ?, 1, 0)) AS REAL) * 100 / COUNT(T1.Id) FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId", (post_year, min_reputation))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of users within a specific age range
@app.get("/v1/codebase_community/percentage_users_age_range", operation_id="get_percentage_users_age_range", summary="Retrieves the percentage of users within a specified age range from the user database. The age range is defined by the provided minimum and maximum ages.")
async def get_percentage_users_age_range(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age")):
    cursor.execute("SELECT CAST(SUM(IIF(Age BETWEEN ? AND ?, 1, 0)) AS REAL) * 100 / COUNT(Id) FROM users", (min_age, max_age))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get view count and display name of the last editor for posts with specific text in post history
@app.get("/v1/codebase_community/view_count_last_editor_by_post_history_text", operation_id="get_view_count_last_editor", summary="Retrieves the total view count and the display name of the last user who edited a post, given a specific text in the post's history. This operation is useful for tracking the popularity and recent editors of posts containing certain content.")
async def get_view_count_last_editor(post_history_text: str = Query(..., description="Text in post history")):
    cursor.execute("SELECT T2.ViewCount, T3.DisplayName FROM postHistory AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id INNER JOIN users AS T3 ON T2.LastEditorUserId = T3.Id WHERE T1.Text = ?", (post_history_text,))
    result = cursor.fetchall()
    if not result:
        return {"view_counts": []}
    return {"view_counts": [{"view_count": row[0], "display_name": row[1]} for row in result]}

# Endpoint to get post IDs with view count greater than the average view count
@app.get("/v1/codebase_community/post_ids_above_average_view_count", operation_id="get_post_ids_above_average_view_count", summary="Retrieves the identifiers of posts that have received more views than the average view count across all posts.")
async def get_post_ids_above_average_view_count():
    cursor.execute("SELECT Id FROM posts WHERE ViewCount > ( SELECT AVG(ViewCount) FROM posts )")
    result = cursor.fetchall()
    if not result:
        return {"post_ids": []}
    return {"post_ids": [row[0] for row in result]}

# Endpoint to get the post with the highest score sum and its comment count
@app.get("/v1/codebase_community/top_post_by_score_sum", operation_id="get_top_post_by_score_sum", summary="Retrieves the post with the highest cumulative score and its associated comment count. The post is identified by its unique ID, which is used to join with the comments table and calculate the total score and comment count.")
async def get_top_post_by_score_sum():
    cursor.execute("SELECT COUNT(T2.Id) FROM posts AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.PostId GROUP BY T1.Id ORDER BY SUM(T1.Score) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"comment_count": []}
    return {"comment_count": result[0]}

# Endpoint to get the count of posts with view count greater than a specified number and no comments
@app.get("/v1/codebase_community/post_count_view_count_no_comments", operation_id="get_post_count_view_count_no_comments", summary="Retrieves the total number of posts that have surpassed a specified view count threshold and have no comments. The input parameters define the minimum view count and the comment count, which should be set to zero for posts with no comments.")
async def get_post_count_view_count_no_comments(min_view_count: int = Query(..., description="Minimum view count"), comment_count: int = Query(..., description="Number of comments")):
    cursor.execute("SELECT COUNT(Id) FROM posts WHERE ViewCount > ? AND CommentCount = ?", (min_view_count, comment_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the display name and location of the user who last edited a specific post
@app.get("/v1/codebase_community/user_info_last_editor_post", operation_id="get_user_info_last_editor_post", summary="Retrieves the display name and location of the user who most recently edited a specific post. The post is identified by its unique ID.")
async def get_user_info_last_editor_post(post_id: int = Query(..., description="Post ID")):
    cursor.execute("SELECT T2.DisplayName, T2.Location FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T1.Id = ? ORDER BY T1.LastEditDate DESC LIMIT 1", (post_id,))
    result = cursor.fetchone()
    if not result:
        return {"user_info": []}
    return {"user_info": {"display_name": result[0], "location": result[1]}}

# Endpoint to get the most recent badge name for a user by display name
@app.get("/v1/codebase_community/recent_badge_by_user_display_name", operation_id="get_recent_badge_by_user_display_name", summary="Retrieves the name of the most recent badge earned by a user, identified by their display name. The badge is selected based on the latest date it was awarded.")
async def get_recent_badge_by_user_display_name(display_name: str = Query(..., description="User display name")):
    cursor.execute("SELECT T1.Name FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T2.DisplayName = ? ORDER BY T1.Date DESC LIMIT 1", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"badge_name": []}
    return {"badge_name": result[0]}

# Endpoint to get the count of users within a specific age range and minimum upvotes
@app.get("/v1/codebase_community/user_count_age_upvotes", operation_id="get_user_count_age_upvotes", summary="Retrieves the total number of users who fall within a specified age range and have more than a certain number of upvotes. The age range is defined by a minimum and maximum age, and the minimum number of upvotes is also provided as input.")
async def get_user_count_age_upvotes(min_age: int = Query(..., description="Minimum age"), max_age: int = Query(..., description="Maximum age"), min_upvotes: int = Query(..., description="Minimum upvotes")):
    cursor.execute("SELECT COUNT(Id) FROM users WHERE Age BETWEEN ? AND ? AND UpVotes > ?", (min_age, max_age, min_upvotes))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the date difference between badges and user creation date for a specific user
@app.get("/v1/codebase_community/badge_user_date_difference", operation_id="get_badge_user_date_difference", summary="Retrieves the time difference between the creation dates of a user's badges and the user's own creation date. The operation requires the user's display name as input to identify the relevant user and their associated badges.")
async def get_badge_user_date_difference(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T1.Date - T2.CreationDate FROM badges AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"date_differences": []}
    return {"date_differences": result}

# Endpoint to get the count of posts with comments for the most recent user
@app.get("/v1/codebase_community/recent_user_post_comment_count", operation_id="get_recent_user_post_comment_count", summary="Retrieves the total number of posts with comments made by the most recent user. The user is determined by the creation date, and only the latest user's post and comment count is returned.")
async def get_recent_user_post_comment_count():
    cursor.execute("SELECT COUNT(T2.Id) FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId INNER JOIN comments AS T3 ON T3.PostId = T2.Id ORDER BY T1.CreationDate DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the text of comments and display name of users for a specific post title
@app.get("/v1/codebase_community/comments_display_name_post_title", operation_id="get_comments_display_name_post_title", summary="Retrieves the most recent comments and corresponding user display names for a specific post title. The comments are ordered by the user's creation date in descending order, and only the top 10 results are returned.")
async def get_comments_display_name_post_title(post_title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T3.Text, T1.DisplayName FROM users AS T1 INNER JOIN posts AS T2 ON T1.Id = T2.OwnerUserId INNER JOIN comments AS T3 ON T2.Id = T3.PostId WHERE T2.Title = ? ORDER BY T1.CreationDate DESC LIMIT 10", (post_title,))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": result}

# Endpoint to get the count of tags with a specific name
@app.get("/v1/codebase_community/tag_count_by_name", operation_id="get_tag_count_by_name", summary="Retrieves the total count of tags that match the provided tag name. This operation allows you to determine the frequency of a specific tag within the system.")
async def get_tag_count_by_name(tag_name: str = Query(..., description="Name of the tag")):
    cursor.execute("SELECT COUNT(Id) FROM tags WHERE TagName = ?", (tag_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the reputation and views of a user by display name
@app.get("/v1/codebase_community/user_reputation_views", operation_id="get_user_reputation_views", summary="Retrieves the reputation score and view count associated with a specific user, identified by their display name.")
async def get_user_reputation_views(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT Reputation, Views FROM users WHERE DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"reputation": [], "views": []}
    return {"reputation": result[0], "views": result[1]}

# Endpoint to get the comment count and answer count of a post by title
@app.get("/v1/codebase_community/post_comment_answer_count", operation_id="get_post_comment_answer_count", summary="Retrieves the total number of comments and answers associated with a specific post, identified by its title.")
async def get_post_comment_answer_count(post_title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT CommentCount, AnswerCount FROM posts WHERE Title = ?", (post_title,))
    result = cursor.fetchone()
    if not result:
        return {"comment_count": [], "answer_count": []}
    return {"comment_count": result[0], "answer_count": result[1]}

# Endpoint to get the creation date of a user by display name
@app.get("/v1/codebase_community/user_creation_date", operation_id="get_user_creation_date", summary="Retrieves the creation date of a user identified by their display name. The display name is a unique identifier for the user.")
async def get_user_creation_date(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT CreationDate FROM users WHERE DisplayName = ?", (display_name,))
    result = cursor.fetchone()
    if not result:
        return {"creation_date": []}
    return {"creation_date": result[0]}

# Endpoint to get the count of votes with a minimum bounty amount
@app.get("/v1/codebase_community/vote_count_bounty_amount", operation_id="get_vote_count_bounty_amount", summary="Retrieves the total count of votes that have a bounty amount equal to or greater than the specified minimum bounty amount.")
async def get_vote_count_bounty_amount(min_bounty_amount: int = Query(..., description="Minimum bounty amount")):
    cursor.execute("SELECT COUNT(id) FROM votes WHERE BountyAmount >= ?", (min_bounty_amount,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of posts with a score greater than a specified value from users with the maximum reputation
@app.get("/v1/codebase_community/percentage_posts_score_max_reputation", operation_id="get_percentage_posts_score_max_reputation", summary="Retrieves the percentage of posts from users with the highest reputation that have a score surpassing a specified threshold. This operation calculates the proportion of posts meeting the score criterion out of the total posts made by these high-reputation users.")
async def get_percentage_posts_score_max_reputation(score: int = Query(..., description="Score threshold")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Score > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.Id) FROM users T1 INNER JOIN posts T2 ON T1.Id = T2.OwnerUserId INNER JOIN ( SELECT MAX(Reputation) AS max_reputation FROM users ) T3 ON T1.Reputation = T3.max_reputation", (score,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of posts with a score less than a specified value
@app.get("/v1/codebase_community/count_posts_score_less_than", operation_id="get_count_posts_score_less_than", summary="Retrieves the total number of posts that have a score below the provided threshold. This operation is useful for understanding the distribution of post scores and identifying posts that may require attention or improvement.")
async def get_count_posts_score_less_than(score: int = Query(..., description="Score threshold")):
    cursor.execute("SELECT COUNT(id) FROM posts WHERE Score < ?", (score,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of tags with a count less than or equal to a specified value and ID less than a specified value
@app.get("/v1/codebase_community/count_tags_count_id_less_than", operation_id="get_count_tags_count_id_less_than", summary="Retrieves the number of tags that have a count less than or equal to a specified count threshold and an ID less than a specified ID threshold.")
async def get_count_tags_count_id_less_than(count: int = Query(..., description="Count threshold"), id: int = Query(..., description="ID threshold")):
    cursor.execute("SELECT COUNT(id) FROM tags WHERE Count <= ? AND Id < ?", (count, id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ExcerptPostId and WikiPostId for a given tag name
@app.get("/v1/codebase_community/excerpt_wiki_post_id_by_tag_name", operation_id="get_excerpt_wiki_post_id_by_tag_name", summary="Retrieves the unique identifiers of the excerpt and wiki posts associated with a specific tag. The tag is identified by its name.")
async def get_excerpt_wiki_post_id_by_tag_name(tag_name: str = Query(..., description="Tag name")):
    cursor.execute("SELECT ExcerptPostId, WikiPostId FROM tags WHERE TagName = ?", (tag_name,))
    result = cursor.fetchall()
    if not result:
        return {"posts": []}
    return {"posts": result}

# Endpoint to get the reputation and upvotes of users who commented with a specific text
@app.get("/v1/codebase_community/reputation_upvotes_by_comment_text", operation_id="get_reputation_upvotes_by_comment_text", summary="Retrieve the reputation and upvote count of users who have commented with a specific text. The provided comment text is used to filter the results.")
async def get_reputation_upvotes_by_comment_text(text: str = Query(..., description="Comment text")):
    cursor.execute("SELECT T2.Reputation, T2.UpVotes FROM comments AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.Text = ?", (text,))
    result = cursor.fetchall()
    if not result:
        return {"users": []}
    return {"users": result}

# Endpoint to get the text of comments from posts with titles containing a specific phrase
@app.get("/v1/codebase_community/comment_text_by_post_title", operation_id="get_comment_text_by_post_title", summary="Retrieves the text of comments from posts containing a specified phrase in their titles. The phrase is used to filter the posts, and the comment texts from the matching posts are returned.")
async def get_comment_text_by_post_title(title_phrase: str = Query(..., description="Phrase to search in post titles")):
    cursor.execute("SELECT T1.Text FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T2.Title LIKE ?", (f'%{title_phrase}%',))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": result}

# Endpoint to get the top comment text from posts with view counts within a specified range
@app.get("/v1/codebase_community/top_comment_by_view_count_range", operation_id="get_top_comment_by_view_count_range", summary="Retrieves the highest-scoring comment text from posts that have view counts within the provided range. The range is defined by the minimum and maximum view count parameters.")
async def get_top_comment_by_view_count_range(min_view_count: int = Query(..., description="Minimum view count"), max_view_count: int = Query(..., description="Maximum view count")):
    cursor.execute("SELECT Text FROM comments WHERE PostId IN ( SELECT Id FROM posts WHERE ViewCount BETWEEN ? AND ? ) ORDER BY Score DESC LIMIT 1", (min_view_count, max_view_count))
    result = cursor.fetchone()
    if not result:
        return {"comment": []}
    return {"comment": result[0]}

# Endpoint to get the creation date and age of users who commented with text containing a specific phrase
@app.get("/v1/codebase_community/user_creation_age_by_comment_text", operation_id="get_user_creation_age_by_comment_text", summary="Retrieve the creation date and age of users who have included a specific phrase in their comments. The phrase to search for is provided as an input parameter.")
async def get_user_creation_age_by_comment_text(text_phrase: str = Query(..., description="Phrase to search in comment text")):
    cursor.execute("SELECT T2.CreationDate, T2.Age FROM comments AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.text LIKE ?", (f'%{text_phrase}%',))
    result = cursor.fetchall()
    if not result:
        return {"users": []}
    return {"users": result}

# Endpoint to get the count of comments from posts with view counts less than a specified value and a specific score
@app.get("/v1/codebase_community/count_comments_view_count_score", operation_id="get_count_comments_view_count_score", summary="Retrieves the total number of comments associated with posts that have a view count below the provided threshold and a specific score. This operation is useful for analyzing engagement trends based on view count and score criteria.")
async def get_count_comments_view_count_score(view_count: int = Query(..., description="View count threshold"), score: int = Query(..., description="Score value")):
    cursor.execute("SELECT COUNT(T1.Id) FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T2.ViewCount < ? AND T2.Score = ?", (view_count, score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of comments from posts with a specific comment count and score
@app.get("/v1/codebase_community/count_comments_comment_count_score", operation_id="get_count_comments_comment_count_score", summary="Retrieves the total number of comments associated with posts that have a specified comment count and score. The comment count and score are provided as input parameters, allowing for a targeted query of the comments table.")
async def get_count_comments_comment_count_score(comment_count: int = Query(..., description="Comment count value"), score: int = Query(..., description="Score value")):
    cursor.execute("SELECT COUNT(T1.id) FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T2.CommentCount = ? AND T2.Score = ?", (comment_count, score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct comments based on user score and age
@app.get("/v1/codebase_community/count_distinct_comments_by_score_age", operation_id="get_count_distinct_comments", summary="Retrieves the total number of unique comments made by users of a certain age who have received a specific score for their comments. This operation allows you to analyze the commenting activity of users based on their age and the quality of their comments as determined by the given score.")
async def get_count_distinct_comments(score: int = Query(..., description="Score of the comment"), age: int = Query(..., description="Age of the user")):
    cursor.execute("SELECT COUNT(DISTINCT T1.id) FROM comments AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.Score = ? AND T2.Age = ?", (score, age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get post IDs and comment texts based on post title
@app.get("/v1/codebase_community/get_post_ids_comments_by_title", operation_id="get_post_ids_comments", summary="Retrieves the unique identifiers and comment texts for posts that match a specified title. The title parameter is used to filter the posts and return only those that have the provided title.")
async def get_post_ids_comments(title: str = Query(..., description="Title of the post")):
    cursor.execute("SELECT T2.Id, T1.Text FROM comments AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id WHERE T2.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"posts": []}
    return {"posts": result}

# Endpoint to get user upvotes based on comment text
@app.get("/v1/codebase_community/get_user_upvotes_by_comment_text", operation_id="get_user_upvotes", summary="Retrieves the total number of upvotes received by users who have made a comment containing the specified text. The input parameter is the exact text of the comment.")
async def get_user_upvotes(comment_text: str = Query(..., description="Text of the comment")):
    cursor.execute("SELECT T2.UpVotes FROM comments AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.Text = ?", (comment_text,))
    result = cursor.fetchall()
    if not result:
        return {"upvotes": []}
    return {"upvotes": result}

# Endpoint to get comment texts based on user display name
@app.get("/v1/codebase_community/get_comments_by_display_name", operation_id="get_comments_by_display_name", summary="Retrieves the texts of comments made by a user identified by a specific display name. The display name is used to locate the user and subsequently, their associated comments.")
async def get_comments_by_display_name(display_name: str = Query(..., description="Display name of the user")):
    cursor.execute("SELECT T1.Text FROM comments AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T2.DisplayName = ?", (display_name,))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": result}

# Endpoint to get user display names based on comment score range and downvotes
@app.get("/v1/codebase_community/get_display_names_by_score_range_downvotes", operation_id="get_display_names", summary="Retrieve the display names of users whose comments have scores within a specified range and a given number of downvotes. This operation allows you to filter users based on the scores of their comments and the number of downvotes they have received.")
async def get_display_names(min_score: int = Query(..., description="Minimum score of the comment"), max_score: int = Query(..., description="Maximum score of the comment"), downvotes: int = Query(..., description="Number of downvotes")):
    cursor.execute("SELECT T2.DisplayName FROM comments AS T1 INNER JOIN users AS T2 ON T1.UserId = T2.Id WHERE T1.Score BETWEEN ? AND ? AND T2.DownVotes = ?", (min_score, max_score, downvotes))
    result = cursor.fetchall()
    if not result:
        return {"display_names": []}
    return {"display_names": result}

# Endpoint to get the percentage of users with zero upvotes based on comment score range
@app.get("/v1/codebase_community/get_percentage_zero_upvotes_by_score_range", operation_id="get_percentage_zero_upvotes", summary="Retrieves the percentage of users who have not upvoted any content and whose comments fall within a specified score range. The score range is defined by the minimum and maximum score parameters.")
async def get_percentage_zero_upvotes(min_score: int = Query(..., description="Minimum score of the comment"), max_score: int = Query(..., description="Maximum score of the comment")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.UpVotes = 0, 1, 0)) AS REAL) * 100/ COUNT(T1.Id) AS per FROM users AS T1 INNER JOIN comments AS T2 ON T1.Id = T2.UserId WHERE T2.Score BETWEEN ? AND ?", (min_score, max_score))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/codebase_community/users/highest_reputation_display_names?display_name1=Harlan&display_name2=Jarrod%20Dixon",
    "/v1/codebase_community/users/display_names_by_creation_year?year=2011",
    "/v1/codebase_community/users/count_last_access_after_date?date=2014-09-01",
    "/v1/codebase_community/users/highest_views_display_names",
    "/v1/codebase_community/users/count_upvotes_downvotes?upvotes=100&downvotes=1",
    "/v1/codebase_community/users/count_creation_year_views?year=2013&views=10",
    "/v1/codebase_community/posts/count_posts_by_user?display_name=csgillespie",
    "/v1/codebase_community/posts/titles_by_user?display_name=csgillespie",
    "/v1/codebase_community/posts/display_names_by_title?title=Eliciting%20priors%20from%20experts",
    "/v1/codebase_community/posts/most_viewed_post_by_user?display_name=csgillespie",
    "/v1/codebase_community/top_favorite_post_user?limit=1",
    "/v1/codebase_community/total_comment_count_by_user?display_name=csgillespie",
    "/v1/codebase_community/max_answer_count_by_user?display_name=csgillespie",
    "/v1/codebase_community/last_editor_by_post_title?title=Examples%20for%20teaching:%20Correlation%20does%20not%20mean%20causation",
    "/v1/codebase_community/top_level_post_count_by_user?display_name=csgillespie",
    "/v1/codebase_community/users_with_closed_posts",
    "/v1/codebase_community/post_count_by_score_and_age?min_score=20&min_age=65",
    "/v1/codebase_community/user_location_by_post_title?title=Eliciting%20priors%20from%20experts",
    "/v1/codebase_community/post_body_by_tag?tag_name=bayesian",
    "/v1/codebase_community/top_tagged_post_body?limit=1",
    "/v1/codebase_community/badges/count_by_display_name?display_name=csgillespie",
    "/v1/codebase_community/badges/names_by_display_name?display_name=csgillespie",
    "/v1/codebase_community/badges/count_by_year_and_display_name?year=2011&display_name=csgillespie",
    "/v1/codebase_community/badges/top_user_by_badge_count",
    "/v1/codebase_community/posts/average_score_by_display_name?display_name=csgillespie",
    "/v1/codebase_community/badges/ratio_badges_to_users_by_views?min_views=200",
    "/v1/codebase_community/posts/percentage_by_age_and_score?min_age=65&min_score=5",
    "/v1/codebase_community/votes/count_by_user_and_date?user_id=58&creation_date=2010-07-19",
    "/v1/codebase_community/votes/most_votes_date",
    "/v1/codebase_community/badges/count_by_name?name=Revival",
    "/v1/codebase_community/top_scoring_comment_post_title",
    "/v1/codebase_community/post_count_by_view_count?view_count=1910",
    "/v1/codebase_community/favorite_count_by_comment_date_user?creation_date=2014-04-23%2020:29:39.0&user_id=3025",
    "/v1/codebase_community/comment_text_by_parent_id_comment_count?parent_id=107829&comment_count=1",
    "/v1/codebase_community/post_finish_status_by_user_comment_date?user_id=23853&creation_date=2013-07-12%2009:08:18.0",
    "/v1/codebase_community/user_reputation_by_post_id?post_id=65041",
    "/v1/codebase_community/user_count_by_display_name?display_name=Tiago%20Pasqualini",
    "/v1/codebase_community/user_display_name_by_vote_id?vote_id=6347",
    "/v1/codebase_community/post_count_by_title_phrase?title_phrase=data%20visualization",
    "/v1/codebase_community/user_badge_names_by_display_name?display_name=DatEpicCoderGuyWhoPrograms",
    "/v1/codebase_community/posts_to_votes_ratio?user_id=24",
    "/v1/codebase_community/post_view_count?title=Integration%20of%20Weka%20and/or%20RapidMiner%20into%20Informatica%20PowerCenter/Developer",
    "/v1/codebase_community/comment_text_by_score?score=17",
    "/v1/codebase_community/user_display_name_by_website?website_url=http://stackoverflow.com",
    "/v1/codebase_community/user_display_name_by_comment?comment_text=thank%20you%20user93!",
    "/v1/codebase_community/comments_by_user?display_name=A%20Lion",
    "/v1/codebase_community/user_info_by_post?post_title=Understanding%20what%20Dassault%20iSight%20is%20doing?",
    "/v1/codebase_community/comments_by_post?post_title=How%20does%20gentle%20boosting%20differ%20from%20AdaBoost?",
    "/v1/codebase_community/user_display_names_by_badge?badge_name=Necromancer",
    "/v1/codebase_community/sum_scores_website_urls_by_editor?display_name=Yevgeny",
    "/v1/codebase_community/comments_by_post_title?post_title=Why%20square%20the%20difference%20instead%20of%20taking%20the%20absolute%20value%20in%20standard%20deviation%3F",
    "/v1/codebase_community/sum_bounty_amounts_by_post_title_keyword?title_keyword=data",
    "/v1/codebase_community/display_names_titles_by_bounty_amount_title_keyword?bounty_amount=50&title_keyword=variance",
    "/v1/codebase_community/avg_view_count_titles_text_by_tag?tag=%3Chumor%3E",
    "/v1/codebase_community/comment_count_by_user?user_id=13",
    "/v1/codebase_community/user_ids_with_max_reputation",
    "/v1/codebase_community/user_ids_with_min_views",
    "/v1/codebase_community/badge_count_by_year_and_name?year=2011&name=Supporter",
    "/v1/codebase_community/user_count_by_badge_count?min_badges=5",
    "/v1/codebase_community/badge_count_by_location_and_names?name1=Supporter&name2=Teacher&location=New%20York",
    "/v1/codebase_community/user_details_by_post_id?post_id=1",
    "/v1/codebase_community/user_ids_by_view_count_and_history_types?min_view_count=1000&distinct_history_types=1",
    "/v1/codebase_community/most_common_badge_name?limit=1",
    "/v1/codebase_community/badge_count_by_location_and_name?location=India&name=Teacher",
    "/v1/codebase_community/badge_percentage_difference_by_years?year1=2010&year2=2011&name=Student",
    "/v1/codebase_community/post_history_type_and_user_count?post_id=3720",
    "/v1/codebase_community/post_score_link_type?post_id=395",
    "/v1/codebase_community/post_history_min_score?min_score=60",
    "/v1/codebase_community/sum_favorite_counts?user_id=686&year=2011",
    "/v1/codebase_community/avg_upvotes_age?min_posts=10",
    "/v1/codebase_community/badge_names_by_date?date=2010-07-19%2019:39:08.0",
    "/v1/codebase_community/comment_count_min_score?min_score=60",
    "/v1/codebase_community/comment_text_by_date?creation_date=2010-07-19%2019:16:14.0",
    "/v1/codebase_community/post_count_by_score?score=10",
    "/v1/codebase_community/top_reputation_user_badge",
    "/v1/codebase_community/user_reputation_by_badge_date?badge_date=2010-07-19%2019:39:08.0",
    "/v1/codebase_community/badge_dates_by_user_location?location=Rochester,%20NY",
    "/v1/codebase_community/percentage_users_with_badge?badge_name=Teacher",
    "/v1/codebase_community/percentage_users_with_badge_age_range?badge_name=Organizer&min_age=13&max_age=18",
    "/v1/codebase_community/comment_scores_by_creation_date?creation_date=2010-07-19%2019:19:56.0",
    "/v1/codebase_community/comment_texts_by_creation_date?creation_date=2010-07-19%2019:37:33.0",
    "/v1/codebase_community/user_ages_by_location?location=Vienna,%20Austria",
    "/v1/codebase_community/count_users_with_badge_age_range?badge_name=Supporter&min_age=19&max_age=65",
    "/v1/codebase_community/user_views_by_badge_date?badge_date=2010-07-19%2019:39:08.0",
    "/v1/codebase_community/badges_min_reputation",
    "/v1/codebase_community/user_count_age_badge?min_age=65&badge_name=Supporter",
    "/v1/codebase_community/user_display_name_by_id?user_id=30",
    "/v1/codebase_community/user_count_by_location?location=New%20York",
    "/v1/codebase_community/vote_count_by_year?year=2010",
    "/v1/codebase_community/user_count_by_age_range?min_age=19&max_age=65",
    "/v1/codebase_community/users_max_views",
    "/v1/codebase_community/vote_ratio_by_years?year1=2010&year2=2011",
    "/v1/codebase_community/post_tags_by_user?display_name=John%20Salvatier",
    "/v1/codebase_community/user_count_by_display_name_and_votes?display_name=Harlan",
    "/v1/codebase_community/top_post_by_answer_count?display_name=slashnick",
    "/v1/codebase_community/top_display_name_by_view_count?display_name1=Harvey%20Motulsky&display_name2=Noah%20Snyder",
    "/v1/codebase_community/user_count_by_display_name_and_vote_count?display_name=Matt%20Parker&min_votes=4",
    "/v1/codebase_community/comment_count_by_display_name_and_score?display_name=Neil%20McGuigan&max_score=60",
    "/v1/codebase_community/post_tags_by_display_name_and_comment_count?display_name=Mark%20Meckes&comment_count=0",
    "/v1/codebase_community/user_display_names_by_badge_name?badge_name=Organizer",
    "/v1/codebase_community/percentage_posts_by_tag_and_display_name?tag_name=r&display_name=Community",
    "/v1/codebase_community/view_count_difference_between_users?display_name1=Mornington&display_name2=Amos",
    "/v1/codebase_community/badges_count_by_name_and_year?name=Commentator&year=2014",
    "/v1/codebase_community/post_history_count_by_creation_date?creation_date=2010-07-21",
    "/v1/codebase_community/users_with_max_views",
    "/v1/codebase_community/posts_last_edit_by_title?title=Detecting%20a%20given%20face%20in%20a%20database%20of%20facial%20images",
    "/v1/codebase_community/comments_count_by_user_and_score?user_id=13&score=60",
    "/v1/codebase_community/posts_comments_by_score?score=60",
    "/v1/codebase_community/badges_by_year_and_location?year=2011&location=North%20Pole",
    "/v1/codebase_community/users_by_favorite_count?favorite_count=150",
    "/v1/codebase_community/post_history_by_title?title=What%20is%20the%20best%20introductory%20Bayesian%20statistics%20textbook%3F",
    "/v1/codebase_community/users_by_badge_name?badge_name=outliers",
    "/v1/codebase_community/related_post_titles?post_title=How%20to%20tell%20if%20something%20happened%20in%20a%20data%20set%20which%20monitors%20a%20value%20over%20time",
    "/v1/codebase_community/user_posts_badges_by_year?user_display_name=Samuel&year=2013",
    "/v1/codebase_community/top_viewed_post_user",
    "/v1/codebase_community/user_info_by_tag?tag_name=hypothesis-testing",
    "/v1/codebase_community/related_post_titles_link_types?post_title=What%20are%20principal%20component%20scores%3F",
    "/v1/codebase_community/top_scored_post_user",
    "/v1/codebase_community/top_bounty_user?vote_type_id=8",
    "/v1/codebase_community/top_viewed_posts",
    "/v1/codebase_community/tag_count_range?min_count=5000&max_count=7000",
    "/v1/codebase_community/max_favorite_post_user",
    "/v1/codebase_community/user_age_highest_reputation",
    "/v1/codebase_community/count_posts_bounty_amount_year?bounty_amount=50&year=2011",
    "/v1/codebase_community/user_id_youngest",
    "/v1/codebase_community/sum_scores_last_activity_date?last_activity_date=2010-07-19%",
    "/v1/codebase_community/avg_monthly_post_links_answer_count_year?answer_count=2&year=2010",
    "/v1/codebase_community/post_id_highest_favorite_count_user?user_id=1465",
    "/v1/codebase_community/earliest_post_title",
    "/v1/codebase_community/user_most_badges",
    "/v1/codebase_community/earliest_vote_creation_date_user?display_name=chl",
    "/v1/codebase_community/earliest_post_creation_date_youngest_user",
    "/v1/codebase_community/user_display_name_by_badge?badge_name=Autobiographer",
    "/v1/codebase_community/user_count_by_location_and_favorite_count?location=United%20Kingdom&min_favorite_count=4",
    "/v1/codebase_community/average_post_id_voted_by_oldest_user",
    "/v1/codebase_community/user_with_highest_reputation",
    "/v1/codebase_community/user_count_by_reputation_and_views?min_reputation=2000&min_views=1000",
    "/v1/codebase_community/user_display_names_by_age_range?min_age=19&max_age=65",
    "/v1/codebase_community/post_count_by_user_and_year?year=2010&display_name=Jay%20Stevens",
    "/v1/codebase_community/highest_scored_post",
    "/v1/codebase_community/average_post_score_by_user?display_name=Stephen%20Turner",
    "/v1/codebase_community/user_display_names_by_post_year_view_count?post_year=2011&min_view_count=20000",
    "/v1/codebase_community/top_user_by_favorite_count_year?user_creation_year=2010",
    "/v1/codebase_community/percentage_high_reputation_users_posted_year?post_year=2011&min_reputation=1000",
    "/v1/codebase_community/percentage_users_age_range?min_age=13&max_age=18",
    "/v1/codebase_community/view_count_last_editor_by_post_history_text?post_history_text=Computer%20Game%20Datasets",
    "/v1/codebase_community/post_ids_above_average_view_count",
    "/v1/codebase_community/top_post_by_score_sum",
    "/v1/codebase_community/post_count_view_count_no_comments?min_view_count=35000&comment_count=0",
    "/v1/codebase_community/user_info_last_editor_post?post_id=183",
    "/v1/codebase_community/recent_badge_by_user_display_name?display_name=Emmett",
    "/v1/codebase_community/user_count_age_upvotes?min_age=19&max_age=65&min_upvotes=5000",
    "/v1/codebase_community/badge_user_date_difference?display_name=Zolomon",
    "/v1/codebase_community/recent_user_post_comment_count",
    "/v1/codebase_community/comments_display_name_post_title?post_title=Analysing%20wind%20data%20with%20R",
    "/v1/codebase_community/tag_count_by_name?tag_name=careers",
    "/v1/codebase_community/user_reputation_views?display_name=Jarrod%20Dixon",
    "/v1/codebase_community/post_comment_answer_count?post_title=Clustering%201D%20data",
    "/v1/codebase_community/user_creation_date?display_name=IrishStat",
    "/v1/codebase_community/vote_count_bounty_amount?min_bounty_amount=30",
    "/v1/codebase_community/percentage_posts_score_max_reputation?score=50",
    "/v1/codebase_community/count_posts_score_less_than?score=20",
    "/v1/codebase_community/count_tags_count_id_less_than?count=20&id=15",
    "/v1/codebase_community/excerpt_wiki_post_id_by_tag_name?tag_name=sample",
    "/v1/codebase_community/reputation_upvotes_by_comment_text?text=fine,%20you%20win%20:)",
    "/v1/codebase_community/comment_text_by_post_title?title_phrase=linear%20regression",
    "/v1/codebase_community/top_comment_by_view_count_range?min_view_count=100&max_view_count=150",
    "/v1/codebase_community/user_creation_age_by_comment_text?text_phrase=http://",
    "/v1/codebase_community/count_comments_view_count_score?view_count=5&score=0",
    "/v1/codebase_community/count_comments_comment_count_score?comment_count=1&score=0",
    "/v1/codebase_community/count_distinct_comments_by_score_age?score=0&age=40",
    "/v1/codebase_community/get_post_ids_comments_by_title?title=Group%20differences%20on%20a%20five%20point%20Likert%20item",
    "/v1/codebase_community/get_user_upvotes_by_comment_text?comment_text=R%20is%20also%20lazy%20evaluated.",
    "/v1/codebase_community/get_comments_by_display_name?display_name=Harvey%20Motulsky",
    "/v1/codebase_community/get_display_names_by_score_range_downvotes?min_score=1&max_score=5&downvotes=0",
    "/v1/codebase_community/get_percentage_zero_upvotes_by_score_range?min_score=5&max_score=10"
]
