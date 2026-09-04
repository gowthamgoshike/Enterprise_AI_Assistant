import os

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


load_dotenv()


DATABASE_HOST = os.getenv(
    "DATABASE_HOST",
    "localhost",
)

DATABASE_PORT = os.getenv(
    "DATABASE_PORT",
    "5432",
)

DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    "enterprise_ai",
)

DATABASE_USER = os.getenv(
    "DATABASE_USER",
    "enterprise_user",
)

DATABASE_PASSWORD = os.getenv(
    "DATABASE_PASSWORD",
)


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DATABASE_USER}:"
    f"{DATABASE_PASSWORD}@"
    f"{DATABASE_HOST}:"
    f"{DATABASE_PORT}/"
    f"{DATABASE_NAME}"
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)