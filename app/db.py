# db.py
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load .env (optional but handy for local dev)
load_dotenv()

# Use env var if present; otherwise fall back to your default DSN
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:rootpass@127.0.0.1:3306/ecs_mapper?charset=utf8mb4",
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)

# ❗ remove the trailing comma here
Base = declarative_base()
