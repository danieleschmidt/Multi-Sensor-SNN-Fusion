"""
Base Repository Pattern Implementation

Provides abstract base classes and common CRUD operations
for data access layers in SNN-Fusion experimental tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generic, TypeVar, Union
from datetime import datetime
import json
import logging

from ..database.connection import DatabaseManager, get_database


T = TypeVar('T')  # Type variable for model objects


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository providing common data access patterns.
    
    Implements standard CRUD operations with transaction support
    and error handling for neuromorphic research data management.
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        table_name: Optional[str] = None,
    ):
        """
        Initialize repository.
        
        Args:
            db_manager: Database manager instance
            table_name: Database table name
        """
        self.db = db_manager or get_database()
        self.table_name = table_name or self._get_table_name()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def _get_table_name(self) -> str:
        """Get the table name for this repository."""
        pass
    
    @abstractmethod
    def _to_dict(self, obj: T) -> Dict[str, Any]:
        """Convert model object to dictionary for database storage."""
        pass
    
    @abstractmethod
    def _from_dict(self, data: Dict[str, Any]) -> T:
        """Convert dictionary from database to model object."""
        pass
    
    def create(self, obj: T) -> int:
        """
        Create a new record in the database.
        
        Args:
            obj: Model object to create
            
        Returns:
            Created record ID
        """
        try:
            record_data = self._to_dict(obj)
            
            # Remove id if present (auto-generated)
            if 'id' in record_data:
                del record_data['id']
            
            # Add created_at timestamp if not present
            if 'created_at' not in record_data:
                record_data['created_at'] = datetime.utcnow().isoformat()
            
            record_id = self.db.insert_record(
                table=self.table_name,
                record=record_data,
                return_id=True,
            )
            
            self.logger.debug(f"Created {self.table_name} record with ID: {record_id}")
            return record_id
            
        except Exception as e:
            self.logger.error(f"Failed to create {self.table_name} record: {e}")
            raise
    
    def get_by_id(self, record_id: int) -> Optional[T]:
        """
        Get a record by its ID.
        
        Args:
            record_id: Record ID
            
        Returns:
            Model object or None if not found
        """
        try:
            record_data = self.db.get_record(self.table_name, record_id)
            
            if record_data:
                return self._from_dict(record_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get {self.table_name} record {record_id}: {e}")
            raise
    
    def update(self, record_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update a record with new data.
        
        Args:
            record_id: Record ID to update
            updates: Fields to update
            
        Returns:
            Success status
        """
        try:
            # Add updated_at timestamp
            updates_with_timestamp = updates.copy()
            updates_with_timestamp['updated_at'] = datetime.utcnow().isoformat()
            
            success = self.db.update_record(
                table=self.table_name,
                record_id=record_id,
                updates=updates_with_timestamp,
            )
            
            if success:
                self.logger.debug(f"Updated {self.table_name} record {record_id}")
            else:
                self.logger.warning(f"Failed to update {self.table_name} record {record_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update {self.table_name} record {record_id}: {e}")
            raise
    
    def delete(self, record_id: int) -> bool:
        """
        Delete a record by ID.
        
        Args:
            record_id: Record ID to delete
            
        Returns:
            Success status
        """
        try:
            success = self.db.delete_record(self.table_name, record_id)
            
            if success:
                self.logger.debug(f"Deleted {self.table_name} record {record_id}")
            else:
                self.logger.warning(f"Failed to delete {self.table_name} record {record_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete {self.table_name} record {record_id}: {e}")
            raise
    
    def find_all(
        self,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> List[T]:
        """
        Get all records from the table.
        
        Args:
            limit: Maximum number of records
            order_by: Field to order by
            
        Returns:
            List of model objects
        """
        try:
            records = self.db.search_records(
                table=self.table_name,
                conditions=None,
                order_by=order_by,
                limit=limit,
            )
            
            return [self._from_dict(record) for record in records]
            
        except Exception as e:
            self.logger.error(f"Failed to get all {self.table_name} records: {e}")
            raise
    
    def find_by(
        self,
        conditions: Dict[str, Any],
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> List[T]:
        """
        Find records matching conditions.
        
        Args:
            conditions: Search conditions
            limit: Maximum number of records
            order_by: Field to order by
            
        Returns:
            List of matching model objects
        """
        try:
            records = self.db.search_records(
                table=self.table_name,
                conditions=conditions,
                order_by=order_by,
                limit=limit,
            )
            
            return [self._from_dict(record) for record in records]
            
        except Exception as e:
            self.logger.error(f"Failed to search {self.table_name} records: {e}")
            raise
    
    def find_one_by(self, conditions: Dict[str, Any]) -> Optional[T]:
        """
        Find first record matching conditions.
        
        Args:
            conditions: Search conditions
            
        Returns:
            First matching model object or None
        """
        results = self.find_by(conditions, limit=1)
        return results[0] if results else None
    
    def count(self, conditions: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records matching conditions.
        
        Args:
            conditions: Optional search conditions
            
        Returns:
            Number of matching records
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {self.table_name}"
            params = []
            
            if conditions:
                where_clauses = []
                for column, value in conditions.items():
                    if self.db.db_type == "sqlite":
                        where_clauses.append(f"{column} = ?")
                    else:
                        where_clauses.append(f"{column} = %s")
                    params.append(value)
                
                query += f" WHERE {' AND '.join(where_clauses)}"
            
            result = self.db.execute_query(query, params, fetch="one")
            return result['count'] if result else 0
            
        except Exception as e:
            self.logger.error(f"Failed to count {self.table_name} records: {e}")
            raise
    
    def exists(self, record_id: int) -> bool:
        """
        Check if record exists by ID.
        
        Args:
            record_id: Record ID to check
            
        Returns:
            True if record exists
        """
        return self.get_by_id(record_id) is not None
    
    def bulk_create(self, objects: List[T]) -> List[int]:
        """
        Create multiple records in a single transaction.
        
        Args:
            objects: List of model objects to create
            
        Returns:
            List of created record IDs
        """
        if not objects:
            return []
        
        operations = []
        
        for obj in objects:
            record_data = self._to_dict(obj)
            
            # Remove id if present
            if 'id' in record_data:
                del record_data['id']
            
            # Add created_at timestamp
            if 'created_at' not in record_data:
                record_data['created_at'] = datetime.utcnow().isoformat()
            
            # Prepare insert operation
            columns = list(record_data.keys())
            placeholders = ['?' if self.db.db_type == 'sqlite' else '%s'] * len(columns)
            
            query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            
            operations.append({
                'query': query,
                'params': list(record_data.values())
            })
        
        try:
            success = self.db.execute_transaction(operations)
            
            if success:
                self.logger.debug(f"Bulk created {len(objects)} {self.table_name} records")
                # Note: Transaction doesn't return IDs, would need separate implementation
                return []
            else:
                raise Exception("Bulk create transaction failed")
                
        except Exception as e:
            self.logger.error(f"Failed to bulk create {self.table_name} records: {e}")
            raise
    
    def _serialize_json_field(self, data: Any) -> str:
        """Serialize data to JSON string for database storage."""
        if data is None:
            return None
        return json.dumps(data, default=str)
    
    def _deserialize_json_field(self, json_str: Optional[str]) -> Any:
        """Deserialize JSON string from database."""
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            self.logger.warning(f"Failed to deserialize JSON: {json_str}")
            return None