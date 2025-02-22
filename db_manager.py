#!/usr/bin/env python3
# =============================================================================
# FILE: db_manager.py
# =============================================================================
"""
db_manager.py

Manages the asynchronous database connection, engine, and session for PostgreSQL
using SQLAlchemy with async support. This module initializes the database schema,
provides session management for database operations, and ensures graceful shutdown
of the database engine.

Key Features:
- Asynchronous engine and session management for non-blocking database operations.
- Schema initialization using models defined in models.py.
- Environment-based configuration for database connection.
- Graceful shutdown of the database engine.
- Integrated logging for monitoring and debugging.

Usage:
- Initialize the DBManager in main.py or another entry point.
- Use get_session() to retrieve an AsyncSession for database operations.
- Call shutdown() when the application is stopping to release resources.

Dependencies:
- SQLAlchemy with asyncio support (install: `pip install sqlalchemy[asyncio] asyncpg`)
- python-dotenv for loading environment variables (install: `pip install python-dotenv`)
"""

import os
import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from models import Base  # Import Base from models.py

logger = logging.getLogger(__name__)
load_dotenv()

class DBManager:
    """
    Manages the asynchronous database connection and session for PostgreSQL.

    Attributes:
        engine: The asynchronous SQLAlchemy engine connected to PostgreSQL.
        async_session: A session factory for creating AsyncSession instances.
    """

    def __init__(self):
        """
        Initializes the DBManager by loading the database connection string
        from environment variables and setting up the async engine.
        """
        self.postgres_dsn = os.getenv("POSTGRES_DSN")
        if not self.postgres_dsn:
            raise ValueError("POSTGRES_DSN not set in .env file")

        # Create the async engine with connection pooling for performance
        self.engine = create_async_engine(
            self.postgres_dsn,
            echo=True,
            pool_size=20,
            max_overflow=10
        )

        # Session factory for AsyncSession
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init(self):
        """
        Initializes the database by creating all tables defined in models.py.
        This method should be called once when the application starts.
        """
        try:
            async with self.engine.begin() as conn:
                # Create all tables defined in Base.metadata
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized successfully.")
        except Exception as e:
            logger.exception(f"Error initializing database: {e}")
            raise

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provides an asynchronous session for database operations.
        Usage:
            async with db_manager.get_session() as session:
                # Perform database operations
        """
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                logger.exception(f"Session error: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()

    async def shutdown(self):
        """
        Gracefully shuts down the database engine, releasing all resources.
        This method should be called when the application is stopping.
        """
        try:
            await self.engine.dispose()
            logger.info("Database engine shut down successfully.")
        except Exception as e:
            logger.exception(f"Error shutting down database engine: {e}")
            raise



if __name__ == "__main__":
    # Example usage for testing
    import asyncio

    async def test_init():
        db_manager = DBManager()
        await db_manager.init()
        await db_manager.shutdown()

    asyncio.run(test_init())