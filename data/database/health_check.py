from sqlalchemy import text

from data.database.connection import engine


def check_database() -> None:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        value = result.scalar()

        if value == 1:
            print("PostgreSQL connection successful")


if __name__ == "__main__":
    check_database()
