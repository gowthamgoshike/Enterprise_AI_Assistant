from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config import settings


DATABASE_URL = settings.database_url


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)
