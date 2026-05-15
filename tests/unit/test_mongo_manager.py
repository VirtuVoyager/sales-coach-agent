"""Unit tests for app/database/mongo_manager.py — MongoManager operations."""
import pytest
from unittest.mock import MagicMock, patch, call

from app.database.mongo_manager import MongoManager


@pytest.fixture
def manager(mock_mongo_client):
    with patch("app.database.mongo_manager.MongoClient", return_value=mock_mongo_client), \
         patch("app.database.mongo_manager.certifi.where", return_value="/mock/cacerts"):
        mgr = MongoManager(uri="mongodb://localhost:27017/", db_name="test_db")
    mgr.client = mock_mongo_client
    mgr.db = mock_mongo_client["test_db"]
    return mgr


class TestGetCollection:
    def test_returns_correct_collection(self, manager, mock_mongo_client):
        _ = manager.get_collection("customer_memory")
        mock_mongo_client["test_db"].__getitem__.assert_called_with("customer_memory")


class TestGetCustomerMemory:
    def test_returns_memory_dict_for_known_customer(self, manager, mock_mongo_collection):
        manager.db.__getitem__ = MagicMock(return_value=mock_mongo_collection)

        result = manager.get_customer_memory("cust_001")

        mock_mongo_collection.find_one.assert_called_once_with(
            {"customer_id": "cust_001"}, {"_id": 0}
        )
        assert result["customer_id"] == "cust_001"

    def test_returns_none_for_unknown_customer(self, manager, mock_mongo_collection):
        mock_mongo_collection.find_one.return_value = None
        manager.db.__getitem__ = MagicMock(return_value=mock_mongo_collection)

        result = manager.get_customer_memory("unknown_cust")

        assert result is None

    def test_excludes_mongo_id_field(self, manager, mock_mongo_collection):
        manager.db.__getitem__ = MagicMock(return_value=mock_mongo_collection)
        manager.get_customer_memory("cust_001")

        _, projection = mock_mongo_collection.find_one.call_args[0]
        assert projection == {"_id": 0}


class TestUpdateCustomerMemory:
    def test_calls_update_one_with_upsert(self, manager, mock_mongo_collection):
        manager.db.__getitem__ = MagicMock(return_value=mock_mongo_collection)

        new_data = {"past_objections": ["price"], "industry": "FinTech"}
        manager.update_customer_memory("cust_002", new_data)

        mock_mongo_collection.update_one.assert_called_once_with(
            {"customer_id": "cust_002"},
            {"$set": new_data},
            upsert=True,
        )

    def test_upsert_creates_new_document_for_unknown_customer(
        self, manager, mock_mongo_collection
    ):
        mock_mongo_collection.update_one.return_value = MagicMock(
            upserted_id="new_id", modified_count=0
        )
        manager.db.__getitem__ = MagicMock(return_value=mock_mongo_collection)

        manager.update_customer_memory("brand_new_cust", {"industry": "Healthcare"})

        assert mock_mongo_collection.update_one.called

    def test_empty_data_dict_is_accepted(self, manager, mock_mongo_collection):
        manager.db.__getitem__ = MagicMock(return_value=mock_mongo_collection)
        # Should not raise
        manager.update_customer_memory("cust_001", {})
        mock_mongo_collection.update_one.assert_called_once()


class TestClose:
    def test_close_calls_client_close(self, manager, mock_mongo_client):
        manager.close()
        mock_mongo_client.close.assert_called_once()
