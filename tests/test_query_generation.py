import pytest
from langchain_core.messages import HumanMessage
from retrieval_graph.graph import generate_query, SearchQuery
from retrieval_graph.state import State

@pytest.mark.asyncio
async def test_generate_query_first_message():
    """Test query generation with first message - should use direct input."""
    # Setup
    state = State(
        messages=[HumanMessage(content="What is the weather like?")],
        queries=[],
        retrieved_docs=[],
        tool_messages=[]
    )
    config = {"configurable": {"user_id": "test_user"}}

    # Execute
    result = await generate_query(state, config=config)

    # Assert
    assert result["queries"] == ["What is the weather like?"]

@pytest.mark.asyncio
async def test_generate_query_with_tool_calls():
    """Test query generation with tool calling structure."""
    # Setup
    state = State(
        messages=[
            HumanMessage(content="What is the weather like?"),
            HumanMessage(content="Is it going to rain tomorrow?")
        ],
        queries=["weather forecast"],
        retrieved_docs=[],
        tool_messages=[]
    )
    config = {
        "configurable": {
            "user_id": "test_user",
            "query_model": "openai/gpt-3.5-turbo"  # Use your configured model
        }
    }

    # Execute
    result = await generate_query(state, config=config)

    # Assert
    assert "queries" in result
    assert isinstance(result["queries"], list)
    assert len(result["queries"]) == 1
    
    # Check tool messages structure
    assert "messages" in result
    assert len(result["messages"]) == 2
    
    tool_call = result["messages"][0]
    assert tool_call["role"] == "assistant"
    assert tool_call["content"] is None
    assert "tool_calls" in tool_call
    
    tool_response = result["messages"][1]
    assert tool_response["role"] == "tool"
    assert "tool_call_id" in tool_response
    assert tool_response["tool_call_id"] == "search_query_generator"

@pytest.mark.asyncio
async def test_generate_query_with_error_handling():
    """Test query generation with error - should fallback to direct input."""
    # Setup
    state = State(
        messages=[
            HumanMessage(content="First message"),
            HumanMessage(content="What happens on error?")
        ],
        queries=[],
        retrieved_docs=[],
        tool_messages=[]
    )
    # Invalid config to trigger error
    config = {"configurable": {}}

    # Execute
    result = await generate_query(state, config=config)

    # Assert
    assert result["queries"] == ["What happens on error?"] 