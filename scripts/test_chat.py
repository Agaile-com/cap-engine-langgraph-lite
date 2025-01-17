import asyncio
from langchain_core.messages import HumanMessage
from retrieval_graph.graph import generate_query
from retrieval_graph.state import State

async def main():
    state = State(
        messages=[
            HumanMessage(content="What is the weather like?"),
            HumanMessage(content="Will it rain tomorrow?")
        ],
        queries=[],
        retrieved_docs=[],
        tool_messages=[]
    )
    
    config = {
        "configurable": {
            "user_id": "test_user",
            "query_model": "openai/gpt-3.5-turbo"
        }
    }
    
    result = await generate_query(state, config=config)
    print("Generated Query:", result["queries"])
    print("\nTool Messages:")
    for msg in result["messages"]:
        print(f"\nRole: {msg['role']}")
        print(f"Content: {msg.get('content')}")
        if "tool_calls" in msg:
            print("Tool Calls:", msg["tool_calls"])

if __name__ == "__main__":
    asyncio.run(main()) 