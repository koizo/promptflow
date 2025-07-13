"""
Unit tests for State Store (Redis-based flow state management).
"""
import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from core.state_store import StateStore
from core.schema import FlowState, FlowStatus, StepStatus
from core.executors.base_executor import ExecutionResult


class TestStateStore:
    """Test StateStore class."""
    
    @pytest.fixture
    def state_store(self):
        """Create a state store instance."""
        return StateStore()
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.set.return_value = True
        mock_client.get.return_value = None
        mock_client.delete.return_value = 1
        mock_client.exists.return_value = 1
        mock_client.expire.return_value = True
        return mock_client
    
    @pytest.fixture
    def sample_flow_state(self):
        """Create a sample flow state."""
        return FlowState(
            flow_id="test_flow_123",
            flow_name="test_flow",
            status=FlowStatus.RUNNING,
            inputs={"input1": "value1", "input2": "value2"},
            outputs={},
            step_results={},
            error=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_execution_result(self):
        """Create a sample execution result."""
        return ExecutionResult(
            success=True,
            outputs={"result": "test_output", "count": 42},
            metadata={"execution_time": 1.5, "model": "test_model"},
            execution_time=1.5
        )
    
    def test_state_store_creation(self, state_store):
        """Test creating a state store."""
        assert isinstance(state_store, StateStore)
        assert state_store.redis_client is None
        assert state_store.ttl > 0
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, state_store, mock_redis):
        """Test successful Redis initialization."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            await state_store.initialize()
            
            assert state_store.redis_client is not None
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, state_store):
        """Test Redis initialization failure."""
        with patch('redis.asyncio.from_url', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception) as exc_info:
                await state_store.initialize()
            
            assert "Connection failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_close(self, state_store, mock_redis):
        """Test closing Redis connection."""
        state_store.redis_client = mock_redis
        
        await state_store.close()
        
        mock_redis.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ping(self, state_store, mock_redis):
        """Test Redis ping."""
        state_store.redis_client = mock_redis
        
        result = await state_store.ping()
        
        assert result is True
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ping_not_initialized(self, state_store):
        """Test ping when Redis client not initialized."""
        with pytest.raises(RuntimeError) as exc_info:
            await state_store.ping()
        
        assert "Redis client not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_save_flow_state(self, state_store, mock_redis, sample_flow_state):
        """Test saving flow state."""
        state_store.redis_client = mock_redis
        
        await state_store.save_flow_state(sample_flow_state)
        
        # Verify Redis set was called with correct key and data
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        
        key = call_args[0][0]
        value = call_args[0][1]
        
        assert key == f"flow_state:{sample_flow_state.flow_id}"
        
        # Verify the serialized data
        saved_data = json.loads(value)
        assert saved_data["flow_id"] == sample_flow_state.flow_id
        assert saved_data["flow_name"] == sample_flow_state.flow_name
        assert saved_data["status"] == sample_flow_state.status.value
    
    @pytest.mark.asyncio
    async def test_get_flow_state_exists(self, state_store, mock_redis, sample_flow_state):
        """Test getting existing flow state."""
        state_store.redis_client = mock_redis
        
        # Mock Redis to return serialized flow state
        serialized_state = json.dumps({
            "flow_id": sample_flow_state.flow_id,
            "flow_name": sample_flow_state.flow_name,
            "status": sample_flow_state.status.value,
            "inputs": sample_flow_state.inputs,
            "outputs": sample_flow_state.outputs,
            "step_results": sample_flow_state.step_results,
            "error": sample_flow_state.error,
            "created_at": sample_flow_state.created_at.isoformat(),
            "updated_at": sample_flow_state.updated_at.isoformat()
        })
        mock_redis.get.return_value = serialized_state
        
        result = await state_store.get_flow_state(sample_flow_state.flow_id)
        
        assert isinstance(result, FlowState)
        assert result.flow_id == sample_flow_state.flow_id
        assert result.flow_name == sample_flow_state.flow_name
        assert result.status == sample_flow_state.status
        
        mock_redis.get.assert_called_once_with(f"flow_state:{sample_flow_state.flow_id}")
    
    @pytest.mark.asyncio
    async def test_get_flow_state_not_exists(self, state_store, mock_redis):
        """Test getting non-existent flow state."""
        state_store.redis_client = mock_redis
        mock_redis.get.return_value = None
        
        result = await state_store.get_flow_state("nonexistent_flow")
        
        assert result is None
        mock_redis.get.assert_called_once_with("flow_state:nonexistent_flow")
    
    @pytest.mark.asyncio
    async def test_delete_flow_state(self, state_store, mock_redis):
        """Test deleting flow state."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        
        result = await state_store.delete_flow_state(flow_id)
        
        assert result is True
        mock_redis.delete.assert_called_once_with(f"flow_state:{flow_id}")
    
    @pytest.mark.asyncio
    async def test_save_step_result(self, state_store, mock_redis, sample_execution_result):
        """Test saving step result."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        step_name = "test_step"
        
        await state_store.save_step_result(flow_id, step_name, sample_execution_result)
        
        # Verify Redis set was called
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        
        key = call_args[0][0]
        value = call_args[0][1]
        
        assert key == f"step_result:{flow_id}:{step_name}"
        
        # Verify the serialized data
        saved_data = json.loads(value)
        assert saved_data["success"] == sample_execution_result.success
        assert saved_data["outputs"] == sample_execution_result.outputs
        assert saved_data["metadata"] == sample_execution_result.metadata
    
    @pytest.mark.asyncio
    async def test_get_step_result_exists(self, state_store, mock_redis, sample_execution_result):
        """Test getting existing step result."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        step_name = "test_step"
        
        # Mock Redis to return serialized execution result
        serialized_result = json.dumps({
            "success": sample_execution_result.success,
            "outputs": sample_execution_result.outputs,
            "error": sample_execution_result.error,
            "metadata": sample_execution_result.metadata,
            "execution_time": sample_execution_result.execution_time
        })
        mock_redis.get.return_value = serialized_result
        
        result = await state_store.get_step_result(flow_id, step_name)
        
        assert isinstance(result, ExecutionResult)
        assert result.success == sample_execution_result.success
        assert result.outputs == sample_execution_result.outputs
        assert result.metadata == sample_execution_result.metadata
        
        mock_redis.get.assert_called_once_with(f"step_result:{flow_id}:{step_name}")
    
    @pytest.mark.asyncio
    async def test_get_step_result_not_exists(self, state_store, mock_redis):
        """Test getting non-existent step result."""
        state_store.redis_client = mock_redis
        mock_redis.get.return_value = None
        
        result = await state_store.get_step_result("flow_123", "step_456")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_step_result(self, state_store, mock_redis):
        """Test deleting step result."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        step_name = "test_step"
        
        result = await state_store.delete_step_result(flow_id, step_name)
        
        assert result is True
        mock_redis.delete.assert_called_once_with(f"step_result:{flow_id}:{step_name}")
    
    @pytest.mark.asyncio
    async def test_flow_exists(self, state_store, mock_redis):
        """Test checking if flow exists."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        
        # Test when flow exists
        mock_redis.exists.return_value = 1
        result = await state_store.flow_exists(flow_id)
        assert result is True
        
        # Test when flow doesn't exist
        mock_redis.exists.return_value = 0
        result = await state_store.flow_exists(flow_id)
        assert result is False
        
        # Verify correct key was checked
        mock_redis.exists.assert_called_with(f"flow_state:{flow_id}")
    
    @pytest.mark.asyncio
    async def test_step_exists(self, state_store, mock_redis):
        """Test checking if step result exists."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        step_name = "test_step"
        
        # Test when step exists
        mock_redis.exists.return_value = 1
        result = await state_store.step_exists(flow_id, step_name)
        assert result is True
        
        # Test when step doesn't exist
        mock_redis.exists.return_value = 0
        result = await state_store.step_exists(flow_id, step_name)
        assert result is False
        
        # Verify correct key was checked
        mock_redis.exists.assert_called_with(f"step_result:{flow_id}:{step_name}")
    
    @pytest.mark.asyncio
    async def test_set_flow_ttl(self, state_store, mock_redis):
        """Test setting TTL for flow state."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        ttl_seconds = 3600
        
        await state_store.set_flow_ttl(flow_id, ttl_seconds)
        
        mock_redis.expire.assert_called_once_with(f"flow_state:{flow_id}", ttl_seconds)
    
    @pytest.mark.asyncio
    async def test_set_step_ttl(self, state_store, mock_redis):
        """Test setting TTL for step result."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        step_name = "test_step"
        ttl_seconds = 1800
        
        await state_store.set_step_ttl(flow_id, step_name, ttl_seconds)
        
        mock_redis.expire.assert_called_once_with(f"step_result:{flow_id}:{step_name}", ttl_seconds)
    
    @pytest.mark.asyncio
    async def test_cleanup_flow_data(self, state_store, mock_redis):
        """Test cleaning up all flow-related data."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        
        # Mock scan to return step result keys
        mock_redis.scan_iter.return_value = [
            f"step_result:{flow_id}:step1",
            f"step_result:{flow_id}:step2",
            f"step_result:other_flow:step1"  # Should not be deleted
        ]
        
        await state_store.cleanup_flow_data(flow_id)
        
        # Should delete flow state and matching step results
        expected_deletes = [
            f"flow_state:{flow_id}",
            f"step_result:{flow_id}:step1",
            f"step_result:{flow_id}:step2"
        ]
        
        # Verify delete was called for each expected key
        assert mock_redis.delete.call_count == len(expected_deletes)
    
    @pytest.mark.asyncio
    async def test_get_flow_statistics(self, state_store, mock_redis):
        """Test getting flow statistics."""
        state_store.redis_client = mock_redis
        flow_id = "test_flow_123"
        
        # Mock scan to return keys
        mock_redis.scan_iter.return_value = [
            f"step_result:{flow_id}:step1",
            f"step_result:{flow_id}:step2",
            f"step_result:{flow_id}:step3"
        ]
        
        # Mock flow state
        flow_state_data = {
            "flow_id": flow_id,
            "status": "completed",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:05:00Z"
        }
        mock_redis.get.return_value = json.dumps(flow_state_data)
        
        stats = await state_store.get_flow_statistics(flow_id)
        
        assert stats["flow_id"] == flow_id
        assert stats["total_steps"] == 3
        assert stats["status"] == "completed"
        assert "created_at" in stats
        assert "updated_at" in stats
    
    @pytest.mark.asyncio
    async def test_error_handling_redis_failure(self, state_store, mock_redis):
        """Test error handling when Redis operations fail."""
        state_store.redis_client = mock_redis
        
        # Mock Redis to raise exception
        mock_redis.set.side_effect = Exception("Redis connection lost")
        
        sample_state = FlowState(
            flow_id="test_flow",
            flow_name="test",
            status=FlowStatus.RUNNING,
            inputs={},
            outputs={},
            step_results={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Should propagate the Redis exception
        with pytest.raises(Exception) as exc_info:
            await state_store.save_flow_state(sample_state)
        
        assert "Redis connection lost" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_serialization_deserialization_edge_cases(self, state_store, mock_redis):
        """Test serialization/deserialization edge cases."""
        state_store.redis_client = mock_redis
        
        # Test with complex nested data
        complex_result = ExecutionResult(
            success=True,
            outputs={
                "nested_dict": {"key": "value", "number": 42},
                "list_data": [1, 2, {"nested": True}],
                "unicode_text": "Hello 世界 🌍",
                "null_value": None,
                "boolean": True
            },
            metadata={
                "timestamps": ["2024-01-01T00:00:00Z"],
                "complex_object": {"nested": {"deeply": {"value": "test"}}}
            }
        )
        
        # Save and retrieve
        await state_store.save_step_result("flow_123", "complex_step", complex_result)
        
        # Mock the return value
        serialized = json.dumps(complex_result.to_dict())
        mock_redis.get.return_value = serialized
        
        retrieved = await state_store.get_step_result("flow_123", "complex_step")
        
        assert retrieved.success == complex_result.success
        assert retrieved.outputs == complex_result.outputs
        assert retrieved.metadata == complex_result.metadata
    
    @pytest.mark.asyncio
    async def test_concurrent_access_simulation(self, state_store, mock_redis):
        """Test simulated concurrent access to state store."""
        state_store.redis_client = mock_redis
        
        # Simulate multiple concurrent operations
        flow_id = "concurrent_flow"
        
        # Create multiple tasks that would run concurrently
        tasks = []
        for i in range(5):
            step_name = f"step_{i}"
            result = ExecutionResult(success=True, outputs={"step": i})
            tasks.append(state_store.save_step_result(flow_id, step_name, result))
        
        # Execute all tasks
        await asyncio.gather(*tasks)
        
        # Verify all saves were called
        assert mock_redis.set.call_count == 5


@pytest.mark.integration
class TestStateStoreIntegration:
    """Integration tests for state store with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_flow_lifecycle(self, mock_redis):
        """Test complete flow lifecycle with state persistence."""
        state_store = StateStore()
        state_store.redis_client = mock_redis
        
        flow_id = "lifecycle_flow_123"
        
        # 1. Create initial flow state
        initial_state = FlowState(
            flow_id=flow_id,
            flow_name="lifecycle_test",
            status=FlowStatus.PENDING,
            inputs={"input1": "value1"},
            outputs={},
            step_results={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        await state_store.save_flow_state(initial_state)
        
        # 2. Update to running status
        initial_state.status = FlowStatus.RUNNING
        initial_state.updated_at = datetime.now(timezone.utc)
        await state_store.save_flow_state(initial_state)
        
        # 3. Save step results as they complete
        step_results = []
        for i in range(3):
            step_name = f"step_{i+1}"
            result = ExecutionResult(
                success=True,
                outputs={"result": f"output_{i+1}", "step_number": i+1},
                metadata={"execution_order": i+1}
            )
            step_results.append((step_name, result))
            await state_store.save_step_result(flow_id, step_name, result)
        
        # 4. Update final flow state
        initial_state.status = FlowStatus.COMPLETED
        initial_state.outputs = {"final_result": "All steps completed"}
        initial_state.updated_at = datetime.now(timezone.utc)
        await state_store.save_flow_state(initial_state)
        
        # 5. Verify all operations were called correctly
        # Flow state should be saved 3 times (initial, running, completed)
        flow_state_saves = [call for call in mock_redis.set.call_args_list 
                           if f"flow_state:{flow_id}" in call[0][0]]
        assert len(flow_state_saves) == 3
        
        # Step results should be saved 3 times
        step_result_saves = [call for call in mock_redis.set.call_args_list 
                            if f"step_result:{flow_id}" in call[0][0]]
        assert len(step_result_saves) == 3
    
    @pytest.mark.asyncio
    async def test_flow_recovery_scenario(self, mock_redis):
        """Test flow recovery from partial execution."""
        state_store = StateStore()
        state_store.redis_client = mock_redis
        
        flow_id = "recovery_flow_123"
        
        # Simulate existing partial state
        existing_state = {
            "flow_id": flow_id,
            "flow_name": "recovery_test",
            "status": "running",
            "inputs": {"input1": "value1"},
            "outputs": {},
            "step_results": {},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:02:00Z"
        }
        
        # Mock existing step results
        step1_result = {
            "success": True,
            "outputs": {"result": "step1_output"},
            "error": None,
            "metadata": {},
            "execution_time": 1.5
        }
        
        step2_result = {
            "success": False,
            "outputs": {},
            "error": "Step 2 failed",
            "metadata": {},
            "execution_time": 0.8
        }
        
        # Mock Redis responses for recovery
        def mock_get(key):
            if key == f"flow_state:{flow_id}":
                return json.dumps(existing_state)
            elif key == f"step_result:{flow_id}:step1":
                return json.dumps(step1_result)
            elif key == f"step_result:{flow_id}:step2":
                return json.dumps(step2_result)
            return None
        
        mock_redis.get.side_effect = mock_get
        
        # Retrieve flow state for recovery
        recovered_state = await state_store.get_flow_state(flow_id)
        step1_recovered = await state_store.get_step_result(flow_id, "step1")
        step2_recovered = await state_store.get_step_result(flow_id, "step2")
        
        # Verify recovery data
        assert recovered_state.flow_id == flow_id
        assert recovered_state.status == FlowStatus.RUNNING
        
        assert step1_recovered.success is True
        assert step1_recovered.outputs["result"] == "step1_output"
        
        assert step2_recovered.success is False
        assert step2_recovered.error == "Step 2 failed"
    
    @pytest.mark.asyncio
    async def test_cleanup_and_maintenance(self, mock_redis):
        """Test cleanup and maintenance operations."""
        state_store = StateStore()
        state_store.redis_client = mock_redis
        
        flow_id = "cleanup_flow_123"
        
        # Mock scan to return various keys
        all_keys = [
            f"flow_state:{flow_id}",
            f"step_result:{flow_id}:step1",
            f"step_result:{flow_id}:step2",
            f"step_result:{flow_id}:step3",
            f"flow_state:other_flow",
            f"step_result:other_flow:step1"
        ]
        
        mock_redis.scan_iter.return_value = all_keys
        
        # Perform cleanup
        await state_store.cleanup_flow_data(flow_id)
        
        # Verify only the target flow's data was deleted
        deleted_keys = [call[0][0] for call in mock_redis.delete.call_args_list]
        
        expected_deletions = [
            f"flow_state:{flow_id}",
            f"step_result:{flow_id}:step1",
            f"step_result:{flow_id}:step2",
            f"step_result:{flow_id}:step3"
        ]
        
        for expected_key in expected_deletions:
            assert expected_key in deleted_keys
        
        # Other flow's data should not be deleted
        assert f"flow_state:other_flow" not in deleted_keys
        assert f"step_result:other_flow:step1" not in deleted_keys
