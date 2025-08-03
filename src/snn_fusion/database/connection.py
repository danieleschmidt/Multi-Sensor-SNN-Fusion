"""
Database Connection Management

Provides unified interface for database operations with support for
SQLite, PostgreSQL, and in-memory storage for experimental tracking.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager
import threading
from datetime import datetime, timezone

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class DatabaseManager:
    """
    Unified database management for SNN-Fusion experimental data.
    
    Supports multiple backend types with automatic connection pooling
    and transaction management for neuromorphic research workflows.
    """
    
    def __init__(
        self,
        db_type: str = "sqlite",
        db_path: Optional[str] = None,
        connection_params: Optional[Dict[str, Any]] = None,
        auto_migrate: bool = True,
    ):
        """
        Initialize database manager.
        
        Args:
            db_type: Database type ('sqlite', 'postgresql', 'memory')
            db_path: Path to database file (for SQLite)
            connection_params: Connection parameters for external DBs
            auto_migrate: Automatically run migrations on startup
        """
        self.db_type = db_type.lower()
        self.db_path = db_path
        self.connection_params = connection_params or {}
        self.auto_migrate = auto_migrate
        
        # Thread-local storage for connections
        self._local = threading.local()
        self._connection_lock = threading.Lock()
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._initialize_database()
        
        if auto_migrate:
            self.run_migrations()
    
    def _initialize_database(self) -> None:
        """Initialize database based on type."""
        if self.db_type == "sqlite":
            if not self.db_path:
                self.db_path = "snn_fusion.db"
            
            # Create directory if needed
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
        elif self.db_type == "postgresql":
            if not POSTGRES_AVAILABLE:
                raise ImportError("PostgreSQL support requires psycopg2: pip install psycopg2-binary")
            
            # Validate connection parameters
            required_params = ['host', 'database', 'user', 'password']
            missing_params = [p for p in required_params if p not in self.connection_params]
            if missing_params:
                raise ValueError(f"Missing PostgreSQL parameters: {missing_params}")
        
        elif self.db_type == "memory":
            # In-memory database using SQLite
            self.db_path = ":memory:"
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        connection = None
        try:
            connection = self._get_connection()
            yield connection
        finally:
            if connection and self.db_type != "memory":
                self._return_connection(connection)
    
    def _get_connection(self):
        """Get connection from pool or create new one."""
        # Check if we have a connection in thread-local storage
        if hasattr(self._local, 'connection') and self._local.connection:
            return self._local.connection
        
        with self._connection_lock:
            if self.db_type in ["sqlite", "memory"]:
                connection = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0,
                )
                connection.row_factory = sqlite3.Row  # Enable dict-like access
                # Enable foreign keys
                connection.execute("PRAGMA foreign_keys = ON")
                
            elif self.db_type == "postgresql":
                connection = psycopg2.connect(
                    **self.connection_params,
                    cursor_factory=RealDictCursor,
                )
                connection.autocommit = False
            
            # Store in thread-local storage
            self._local.connection = connection
            return connection
    
    def _return_connection(self, connection) -> None:
        """Return connection to pool."""
        if hasattr(self._local, 'connection'):
            self._local.connection = None
        
        if connection:
            try:
                connection.close()
            except Exception as e:
                self.logger.warning(f"Error closing connection: {e}")
    
    def execute_query(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], List]] = None,
        fetch: str = "none",
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute SQL query with parameters.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Fetch mode ('none', 'one', 'all')
            
        Returns:
            Query results based on fetch mode
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Handle different fetch modes
                if fetch == "one":
                    result = cursor.fetchone()
                    return dict(result) if result else None
                elif fetch == "all":
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                else:
                    # For INSERT/UPDATE/DELETE operations
                    conn.commit()
                    return None
                    
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Database query failed: {e}")
                self.logger.error(f"Query: {query}")
                self.logger.error(f"Params: {params}")
                raise
            finally:
                cursor.close()
    
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """
        Execute multiple operations in a single transaction.
        
        Args:
            operations: List of operation dictionaries with 'query' and 'params'
            
        Returns:
            Success status
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                for operation in operations:
                    query = operation['query']
                    params = operation.get('params')
                    
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                
                conn.commit()
                return True
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction failed: {e}")
                return False
            finally:
                cursor.close()
    
    def run_migrations(self) -> None:
        """Run database migrations to create/update schema."""
        self.logger.info("Running database migrations...")
        
        from .migrations import get_migrations
        migrations = get_migrations(self.db_type)
        
        # Create migrations table if it doesn't exist
        migration_table_query = """
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY,
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.execute_query(migration_table_query)
        
        # Get applied migrations
        applied_migrations = self.execute_query(
            "SELECT migration_name FROM migrations",
            fetch="all"
        )
        applied_names = {m['migration_name'] for m in applied_migrations or []}
        
        # Apply new migrations
        for migration_name, migration_sql in migrations.items():
            if migration_name not in applied_names:
                self.logger.info(f"Applying migration: {migration_name}")
                
                try:
                    # Execute migration
                    self.execute_query(migration_sql)
                    
                    # Record migration
                    self.execute_query(
                        "INSERT INTO migrations (migration_name) VALUES (?)",
                        [migration_name]
                    )
                    
                except Exception as e:
                    self.logger.error(f"Migration {migration_name} failed: {e}")
                    raise
        
        self.logger.info("Database migrations completed")
    
    def insert_record(
        self,
        table: str,
        record: Dict[str, Any],
        return_id: bool = True,
    ) -> Optional[int]:
        """
        Insert a record into the specified table.
        
        Args:
            table: Table name
            record: Record data
            return_id: Whether to return the inserted ID
            
        Returns:
            Inserted record ID if return_id=True
        """
        # Prepare insert query
        columns = list(record.keys())
        placeholders = ['?' if self.db_type == 'sqlite' else '%s'] * len(columns)
        
        query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        values = list(record.values())
        
        if return_id:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(query, values)
                    
                    if self.db_type == "sqlite":
                        record_id = cursor.lastrowid
                    else:  # PostgreSQL
                        record_id = cursor.fetchone()[0]
                    
                    conn.commit()
                    return record_id
                    
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"Insert failed: {e}")
                    raise
                finally:
                    cursor.close()
        else:
            self.execute_query(query, values)
            return None
    
    def update_record(
        self,
        table: str,
        record_id: int,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update a record in the specified table.
        
        Args:
            table: Table name
            record_id: Record ID to update
            updates: Fields to update
            
        Returns:
            Success status
        """
        if not updates:
            return True
        
        # Prepare update query
        set_clauses = []
        values = []
        
        for column, value in updates.items():
            if self.db_type == "sqlite":
                set_clauses.append(f"{column} = ?")
            else:
                set_clauses.append(f"{column} = %s")
            values.append(value)
        
        query = f"""
        UPDATE {table}
        SET {', '.join(set_clauses)}
        WHERE id = {'?' if self.db_type == 'sqlite' else '%s'}
        """
        
        values.append(record_id)
        
        try:
            self.execute_query(query, values)
            return True
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            return False
    
    def get_record(
        self,
        table: str,
        record_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        query = f"SELECT * FROM {table} WHERE id = {'?' if self.db_type == 'sqlite' else '%s'}"
        return self.execute_query(query, [record_id], fetch="one")
    
    def search_records(
        self,
        table: str,
        conditions: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search records with conditions.
        
        Args:
            table: Table name
            conditions: Search conditions
            order_by: Ordering field
            limit: Maximum number of results
            
        Returns:
            List of matching records
        """
        query = f"SELECT * FROM {table}"
        params = []
        
        if conditions:
            where_clauses = []
            for column, value in conditions.items():
                if self.db_type == "sqlite":
                    where_clauses.append(f"{column} = ?")
                else:
                    where_clauses.append(f"{column} = %s")
                params.append(value)
            
            query += f" WHERE {' AND '.join(where_clauses)}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, params, fetch="all") or []
    
    def delete_record(self, table: str, record_id: int) -> bool:
        """Delete a record by ID."""
        query = f"DELETE FROM {table} WHERE id = {'?' if self.db_type == 'sqlite' else '%s'}"
        
        try:
            self.execute_query(query, [record_id])
            return True
        except Exception as e:
            self.logger.error(f"Delete failed: {e}")
            return False
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, 'connection') and self._local.connection:
            try:
                self._local.connection.close()
                self._local.connection = None
            except Exception as e:
                self.logger.warning(f"Error closing connection: {e}")


# Global database instance
_db_instance: Optional[DatabaseManager] = None
_db_lock = threading.Lock()


def get_database(
    db_type: str = "sqlite",
    db_path: Optional[str] = None,
    **kwargs
) -> DatabaseManager:
    """
    Get global database instance with lazy initialization.
    
    Args:
        db_type: Database type
        db_path: Database path
        **kwargs: Additional connection parameters
        
    Returns:
        Database manager instance
    """
    global _db_instance
    
    with _db_lock:
        if _db_instance is None:
            _db_instance = DatabaseManager(
                db_type=db_type,
                db_path=db_path,
                **kwargs
            )
        
        return _db_instance