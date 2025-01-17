"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast
import json

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model

# Define the function that calls the model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "weather forecast for tomorrow"
                }
            ]
        }
    }


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str] | list[dict]]:
    """Generate a search query based on the current state and configuration."""
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    
    try:
        configuration = Configuration.from_runnable_config(config)
        
        # Create a proper prompt for the model
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a search query based on the user's question. Previous queries: {queries}"),
            ("human", "{input}")
        ])
        
        # Format the input for the model with all required variables
        last_message = get_message_text(messages[-1])
        previous_queries = "\n- ".join(state.queries) if state.queries else "None"
        
        message_value = await prompt.ainvoke(
            {
                "input": last_message,
                "queries": previous_queries,
                "system_time": datetime.now(tz=timezone.utc).isoformat()
            },
            config,
        )
        
        # Generate the query using structured output with function calling
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery,
            method="function_calling"  # Explicitly use function calling
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        
        # Create tool messages
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "search_query_generator",
                "type": "function",
                "function": {
                    "name": "generate_search_query",
                    "arguments": {"query": generated.query}
                }
            }]
        }
        
        tool_response = {
            "role": "tool",
            "content": generated.query,
            "tool_call_id": "search_query_generator"
        }
        
        return {
            "queries": [generated.query],
            "messages": [tool_call_message, tool_response]
        }
        
    except Exception as e:
        # Fallback to using the last user message directly
        human_input = get_message_text(messages[-1])
        print(f"Query generation failed: {e}. Falling back to direct user input.")
        return {"queries": [human_input]}


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    configuration = Configuration.from_runnable_config(config)
    with retrieval.make_retriever(config, alternate_milvus_uri = configuration.alternate_milvus_uri) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    return {"messages": [response]}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(generate_query)
builder.add_node(retrieve)
builder.add_node(respond)
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "RetrievalGraph"
