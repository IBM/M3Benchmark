"""Unit tests for initialize_active_data in sql_tools.py.

Test cases are drawn from real agent traces in data/tasks/task_1/output/ for
the disney, california_schools, and olympics domains.

Each fixture yields the real SQLite database if it exists under data/db/,
otherwise it falls back to a minimal mock database with the same schema.
"""

import sqlite3
from pathlib import Path

import pytest

from apis.m3.python_tools.tools.sql_tools import initialize_active_data
from apis.m3.python_tools.tools.dtype_utils import DTYPE_METADATA_KEY

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DB = _PROJECT_ROOT / "data" / "db"


def _real_db(domain: str) -> Path:
    return _DATA_DB / domain / f"{domain}.sqlite"


def _data_keys(result: dict) -> set:
    """Return all keys except the dtype metadata sentinel."""
    return {k for k in result if k != DTYPE_METADATA_KEY}


# ---------------------------------------------------------------------------
# Mock database builders
# ---------------------------------------------------------------------------

def _build_disney_mock(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE characters (
            movie_title TEXT, release_date TEXT, hero TEXT, villian TEXT, song TEXT
        );
        INSERT INTO characters VALUES
            ('Sleeping Beauty',      '1959-01-29', 'Aurora',     'Maleficent', 'Once Upon a Dream'),
            ('Beauty and the Beast', '1991-11-22', 'Belle',      'Gaston',     'Beauty and the Beast'),
            ('Pinocchio',            '1940-02-23', 'Pinocchio',  'Stromboli',  'When You Wish Upon a Star');

        CREATE TABLE director (name TEXT, director TEXT);
        INSERT INTO director VALUES
            ('Sleeping Beauty',      'Clyde Geronimi'),
            ('Beauty and the Beast', 'Gary Trousdale');
    """)
    conn.close()


def _build_california_schools_mock(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE frpm (
            CDSCode                TEXT,
            "Academic Year"        TEXT,
            "County Name"          TEXT,
            "Charter School (Y/N)" INTEGER
        );
        INSERT INTO frpm VALUES
            ('0161119', '2014-2015', 'Alameda', 1),
            ('0161120', '2014-2015', 'Alameda', 0),
            ('9999999', '2014-2015', 'Nowhere', 1);

        CREATE TABLE schools (CDSCode TEXT, County TEXT, School TEXT, Zip TEXT);
        INSERT INTO schools VALUES
            ('0161119', 'Alameda', 'Lincoln Elementary', '94501'),
            ('0161120', 'Alameda', 'Washington Middle',  '94502');
    """)
    conn.close()


def _build_olympics_mock(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE games (id INTEGER, games_year INTEGER, games_name TEXT, season TEXT);
        INSERT INTO games VALUES
            (1, 1992, '1992 Winter', 'Winter'),
            (2, 1994, '1994 Winter', 'Winter');

        CREATE TABLE games_competitor (id INTEGER, games_id INTEGER, person_id INTEGER, age INTEGER);
        INSERT INTO games_competitor VALUES
            (1, 1, 10, 30),
            (2, 2, 10, 32),
            (3, 1, 11, 25);

        CREATE TABLE person (id INTEGER, full_name TEXT, gender TEXT, height INTEGER, weight INTEGER);
        INSERT INTO person VALUES
            (10, 'John Aalberg', 'M', 175, 70),
            (11, 'Jane Doe',    'F', 165, 60);

        CREATE TABLE person_region (person_id INTEGER, region_id INTEGER);
        INSERT INTO person_region VALUES (10, 5), (11, 7);

        CREATE TABLE noc_region (id INTEGER, noc TEXT, region_name TEXT);
        INSERT INTO noc_region VALUES (5, 'NOR', 'Norway'), (7, 'GBR', 'Great Britain');

        CREATE TABLE competitor_event (event_id INTEGER, competitor_id INTEGER, medal_id INTEGER);
        INSERT INTO competitor_event VALUES (1, 1, 2), (2, 2, 1), (3, 3, 3);

        CREATE TABLE medal (id INTEGER, medal_name TEXT);
        INSERT INTO medal VALUES (1, 'Gold'), (2, 'Silver'), (3, 'Bronze');
    """)
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures — yield real DB path when available, mock otherwise
# ---------------------------------------------------------------------------

@pytest.fixture
def disney_db(tmp_path):
    real = _real_db("disney")
    if real.is_file():
        yield str(real)
    else:
        mock = tmp_path / "disney.sqlite"
        _build_disney_mock(mock)
        yield str(mock)


@pytest.fixture
def california_schools_db(tmp_path):
    real = _real_db("california_schools")
    if real.is_file():
        yield str(real)
    else:
        mock = tmp_path / "california_schools.sqlite"
        _build_california_schools_mock(mock)
        yield str(mock)


@pytest.fixture
def olympics_db(tmp_path):
    real = _real_db("olympics")
    if real.is_file():
        yield str(real)
    else:
        mock = tmp_path / "olympics.sqlite"
        _build_olympics_mock(mock)
        yield str(mock)


# ---------------------------------------------------------------------------
# Disney
# ---------------------------------------------------------------------------

class TestDisney:
    def test_no_join_single_table(self, disney_db):
        """Single table with empty alias returns columns without a prefix."""
        result = initialize_active_data(
            condition_sequence=[],
            alias_to_table_dict={"": {"original_table_name": "characters", "modified_table_name": "characters"}},
            database_path=disney_db,
        )

        assert isinstance(result, dict)
        assert DTYPE_METADATA_KEY in result
        keys = _data_keys(result)
        # Empty alias → no prefix applied
        assert "movie_title" in keys
        assert "hero" in keys
        assert "villian" in keys
        assert "song" in keys
        assert len(result["movie_title"]) > 0

    def test_single_join_characters_and_director(self, disney_db):
        """characters JOIN director on movie_title = name."""
        result = initialize_active_data(
            condition_sequence=[["T1.movie_title", "T2.name", "INNER"]],
            alias_to_table_dict={
                "T1": {"original_table_name": "characters", "modified_table_name": "characters"},
                "T2": {"original_table_name": "director",   "modified_table_name": "director"},
            },
            database_path=disney_db,
        )

        assert isinstance(result, dict)
        assert DTYPE_METADATA_KEY in result
        keys = _data_keys(result)
        assert "characters_movie_title" in keys
        assert "characters_hero" in keys
        assert "characters_villian" in keys
        assert "director_name" in keys
        assert "director_director" in keys
        assert len(result["characters_movie_title"]) > 0


# ---------------------------------------------------------------------------
# California schools
# ---------------------------------------------------------------------------

class TestCaliforniaSchools:
    def test_no_join_single_table_safe_column_names(self, california_schools_db):
        """Columns with spaces and special chars are safe-named (no prefix for empty alias)."""
        result = initialize_active_data(
            condition_sequence=[],
            alias_to_table_dict={"": {"original_table_name": "frpm", "modified_table_name": "frpm"}},
            database_path=california_schools_db,
        )

        assert isinstance(result, dict)
        assert DTYPE_METADATA_KEY in result
        keys = _data_keys(result)
        assert "CDSCode" in keys           # already safe
        assert "Academic_Year" in keys     # "Academic Year" → "Academic_Year"
        assert "County_Name" in keys       # "County Name" → "County_Name"
        # "Charter School (Y/N)" → "Charter_School_Y_N"
        assert "Charter_School_Y_N" in keys
        assert len(result["CDSCode"]) > 0

    def test_single_join_frpm_and_schools(self, california_schools_db):
        """frpm JOIN schools on CDSCode; both tables' columns appear with their prefix."""
        result = initialize_active_data(
            condition_sequence=[["T1.CDSCode", "T2.CDSCode", "INNER"]],
            alias_to_table_dict={
                "T1": {"original_table_name": "frpm",    "modified_table_name": "frpm"},
                "T2": {"original_table_name": "schools", "modified_table_name": "schools"},
            },
            database_path=california_schools_db,
        )

        assert isinstance(result, dict)
        assert DTYPE_METADATA_KEY in result
        keys = _data_keys(result)
        assert "frpm_CDSCode" in keys
        assert "frpm_Academic_Year" in keys
        assert "frpm_Charter_School_Y_N" in keys
        assert "schools_CDSCode" in keys
        # Row count: inner join excludes '9999999' (no matching school)
        assert len(result["frpm_CDSCode"]) > 0
        assert "9999999" not in result["frpm_CDSCode"]


# ---------------------------------------------------------------------------
# Olympics
# ---------------------------------------------------------------------------

class TestOlympics:
    def test_single_join_person_region_and_noc_region(self, olympics_db):
        """person_region JOIN noc_region on region_id = id."""
        result = initialize_active_data(
            condition_sequence=[["T1.region_id", "T2.id", "INNER"]],
            alias_to_table_dict={
                "T1": {"original_table_name": "person_region", "modified_table_name": "person_region"},
                "T2": {"original_table_name": "noc_region",    "modified_table_name": "noc_region"},
            },
            database_path=olympics_db,
        )

        assert isinstance(result, dict)
        assert DTYPE_METADATA_KEY in result
        keys = _data_keys(result)
        assert "person_region_person_id" in keys
        assert "person_region_region_id" in keys
        assert "noc_region_noc" in keys
        assert "noc_region_region_name" in keys
        assert len(result["noc_region_region_name"]) > 0

    def test_two_joins_games_competitor_person(self, olympics_db):
        """games JOIN games_competitor JOIN person (chained 2-hop join)."""
        result = initialize_active_data(
            condition_sequence=[
                ["T1.id",        "T2.games_id",  "INNER"],
                ["T2.person_id", "T3.id",         "INNER"],
            ],
            alias_to_table_dict={
                "T1": {"original_table_name": "games",            "modified_table_name": "games"},
                "T2": {"original_table_name": "games_competitor", "modified_table_name": "games_competitor"},
                "T3": {"original_table_name": "person",           "modified_table_name": "person"},
            },
            database_path=olympics_db,
        )

        assert isinstance(result, dict)
        assert DTYPE_METADATA_KEY in result
        keys = _data_keys(result)
        assert "games_id" in keys
        assert "games_games_name" in keys
        assert "games_competitor_games_id" in keys
        assert "games_competitor_person_id" in keys
        assert "person_id" in keys
        assert "person_full_name" in keys
        assert len(result["games_games_name"]) > 0

    def test_three_joins_person_through_medal(self, olympics_db):
        """person → games_competitor → competitor_event → medal (3-hop join)."""
        result = initialize_active_data(
            condition_sequence=[
                ["T1.id",         "T2.person_id",     "INNER"],
                ["T2.id",         "T3.competitor_id", "INNER"],
                ["T3.medal_id",   "T4.id",             "INNER"],
            ],
            alias_to_table_dict={
                "T1": {"original_table_name": "person",           "modified_table_name": "person"},
                "T2": {"original_table_name": "games_competitor", "modified_table_name": "games_competitor"},
                "T3": {"original_table_name": "competitor_event", "modified_table_name": "competitor_event"},
                "T4": {"original_table_name": "medal",            "modified_table_name": "medal"},
            },
            database_path=olympics_db,
        )

        assert isinstance(result, dict)
        assert DTYPE_METADATA_KEY in result
        keys = _data_keys(result)
        assert "person_full_name" in keys
        assert "games_competitor_games_id" in keys
        assert "competitor_event_medal_id" in keys
        assert "medal_id" in keys
        assert "medal_medal_name" in keys
        assert len(result["medal_medal_name"]) > 0
