import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Define system prompt
system_prompt = """You're an AI assistant with access to OpenSearch tools.
You can answer questions directly when appropriate, or use tools when needed.
For data-specific questions, use the appropriate tools.
For multi-step analysis, combine tools in sequence to solve complex problems.

When you need multiple tools to answer a question:
1. Use tools in a logical sequence
2. Use the results of one tool to inform the next tool call
3. Synthesize information from all tools in your final answer

Remember previous interactions with the user and maintain context throughout the conversation.
For conversation, just respond normally."""

# Create model instance
model = ChatOllama(
    model="qwen3:14b",
    base_url="http://192.168.18.201:11434",
    temperature=0.1,
    system=system_prompt
)

# Initialize variables to be accessible from outside
tools = []
agent = None
client = None
conversation_history = [SystemMessage(content=system_prompt)]
is_initialized = False  # Track if we've initialized before

async def initialize_agent():
    """Initialize the MCP client and agent"""
    global client, tools, agent, conversation_history, is_initialized
    
    # Only reset conversation history on the first initialization
    if not is_initialized:
        conversation_history = [SystemMessage(content=system_prompt)]
        is_initialized = True
    
    # Create MCP client
    client = MultiServerMCPClient({
        "opensearch": {
            "url": "http://localhost:9200/_plugins/_ml/mcp/sse?append_to_base_url=true",
            "transport": "sse",
            "headers": {
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            }
        }
    })

    # Get tools
    tools = await client.get_tools()
    
    # Create LangGraph agent with MCP tools
    agent = create_react_agent(model, tools)
    
    return {
        "tools": tools,
        "tool_names": [tool.name for tool in tools],
        "client": client,
        "agent": agent
    }

async def process_query(user_input):
    """Process a user query and return results"""
    global agent, conversation_history
    
    if agent is None:
        await initialize_agent()
    
    # Add user message to history
    conversation_history.append(HumanMessage(content=user_input))
    
    # Process the query with full conversation history
    result = await agent.ainvoke({"messages": conversation_history})

    # Extract information
    final_message = result["messages"][-1]
    
    # Add AI response to conversation history
    conversation_history.append(AIMessage(content=final_message.content))
    
    # Get tools used
    tool_messages = [msg for msg in result["messages"] if hasattr(msg, "tool_call_id")]
    tool_names = []
    if tool_messages:
        tool_names = list(set(msg.tool for msg in tool_messages if hasattr(msg, "tool")))
    
    # Ensure conversation history doesn't grow too large (keep last 10 messages)
    if len(conversation_history) > 12:  # system message + 10 exchanges
        conversation_history = [conversation_history[0]] + conversation_history[-11:]
    
    return {
        "response": final_message.content,
        "tools_used": tool_names,
        "raw_result": result
    }

# Default main function just for direct testing
async def main():
    await initialize_agent()
    
    while True:
        user_input = input("Query: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
            
        result = await process_query(user_input)
        print(f"Response: {result['response']}")
        if result['tools_used']:
            print(f"Tools used: {', '.join(result['tools_used'])}")

if __name__ == "__main__":
    asyncio.run(main())