from fastapi import FastAPI


from server import california_schools, card_games, codebase_community, debit_card_specializing, european_football_2, student_club, financial, toxicology, formula_1, thrombosis_prediction, superhero
from server import bike_share_1
from server import donor, olympics, sales, shipping, software_company, video_games
from server import movielens, mondial_geo, legislator, regional_sales, world_development_indicators, food_inspection_2, retail_world, citeseer, computer_student, college_completion, talkingdata  
from server import synthea, book_publishing_company, trains, retails, soccer_2016, law_episode, food_inspection, european_football_1, mental_health_survey, hockey  
from server import public_review_platform, retail_complains, ice_hockey_draft, menu, cs_semester, beer_factory, cars, genes, shakespeare, image_and_language  
from server import disney, music_tracker, works_cycles, movie_platform, books, social_media, restaurant, superstore, address, chicago_crime  
from server import professional_basketball, coinmarketcap, movies_4, sales_in_weather, app_store, craftbeer, movie, world, movie_3
from server import airline, authors, cookbook, human_resources, language_corpus, music_platform_2, shooting, student_loan
from server import car_retails, codebase_comments, simpson_episodes, university



app = FastAPI()

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

app.include_router(california_schools.app)
app.include_router(card_games.app)
app.include_router(codebase_community.app)
app.include_router(debit_card_specializing.app)
app.include_router(european_football_2.app)
app.include_router(financial.app)
app.include_router(formula_1.app)
app.include_router(student_club.app)
app.include_router(superhero.app)
app.include_router(thrombosis_prediction.app)
app.include_router(toxicology.app)

app.include_router(bike_share_1.app)

app.include_router(donor.app)
app.include_router(olympics.app)
app.include_router(sales.app)
app.include_router(shipping.app)
app.include_router(software_company.app)
app.include_router(video_games.app)

app.include_router(movielens.app)
app.include_router(mondial_geo.app)
app.include_router(legislator.app)
app.include_router(regional_sales.app)
app.include_router(world_development_indicators.app)
app.include_router(food_inspection_2.app)
app.include_router(retail_world.app)
app.include_router(citeseer.app)
app.include_router(computer_student.app)
app.include_router(college_completion.app)
app.include_router(talkingdata.app)
app.include_router(synthea.app)
app.include_router(book_publishing_company.app)
app.include_router(trains.app)
app.include_router(retails.app)
app.include_router(soccer_2016.app)
app.include_router(law_episode.app)
app.include_router(food_inspection.app)
app.include_router(european_football_1.app)
app.include_router(mental_health_survey.app)
app.include_router(hockey.app)

app.include_router(public_review_platform.app)
app.include_router(retail_complains.app)
app.include_router(ice_hockey_draft.app)
app.include_router(menu.app)
app.include_router(cs_semester.app)
app.include_router(beer_factory.app)
app.include_router(cars.app)
app.include_router(genes.app)
app.include_router(shakespeare.app)
app.include_router(image_and_language.app)

app.include_router(disney.app)
app.include_router(music_tracker.app)
app.include_router(works_cycles.app)
app.include_router(movie_platform.app)
app.include_router(books.app)
app.include_router(social_media.app)
app.include_router(restaurant.app)
app.include_router(superstore.app)
app.include_router(address.app)
app.include_router(chicago_crime.app)
app.include_router(professional_basketball.app)
app.include_router(coinmarketcap.app)
app.include_router(movies_4.app)
app.include_router(sales_in_weather.app)
app.include_router(app_store.app)
app.include_router(craftbeer.app)
app.include_router(movie.app)
app.include_router(world.app)
app.include_router(movie_3.app)

app.include_router(airline.app)
app.include_router(authors.app)
app.include_router(cookbook.app)
app.include_router(human_resources.app)
app.include_router(language_corpus.app)
app.include_router(music_platform_2.app)
app.include_router(shooting.app)
app.include_router(student_loan.app)
app.include_router(car_retails.app)
app.include_router(codebase_comments.app)
app.include_router(simpson_episodes.app)
app.include_router(university.app)


