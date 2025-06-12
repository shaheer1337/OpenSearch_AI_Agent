import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from typing import Dict, Any

# Load environment variables
load_dotenv()

system_prompt = """You're an AI assistant with access to OpenSearch tools.
You can answer questions directly when appropriate, or use tools when needed.
For data-specific questions, use the appropriate tools.
For multi-step analysis, combine tools in sequence to solve complex problems.

When you need multiple tools to answer a question:
1. Use tools in a logical sequence
2. Use the results of one tool to inform the next tool call
3. Synthesize information from all tools in your final answer

For conversation, just respond normally."""

model = ChatOllama(
    model="qwen3:14b",
    base_url="http://192.168.18.201:11434",
    temperature=0.1,
    system=system_prompt  # This is how we pass the system prompt to Ollama
)

async def main():
    print("Starting Pure MCP OpenSearch Agent...")

    try:
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

        print("Loading MCP tools...")
        tools = await client.get_tools()
        
        # No need to create wrapper functions - use the tools directly from the MCP client
        print(f"Found {len(tools)} MCP tools: {[tool.name for tool in tools]}")

        if not tools:
            print("No tools found! Check MCP server registration.")
            return

        # Create LangGraph agent with MCP tools - directly using the working approach from agentsingletool.py
        agent = create_react_agent(model, tools)

        print("\n" + "="*50)
        print("Pure MCP OpenSearch AI Assistant Ready")
        print(f"MCP Tools: {[tool.name for tool in tools]}")
        print("="*50)
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Examples:")
        print("- 'list all indices, then show me the mapping of the most interesting one'")
        print("- 'search firewall-logs-+000000 for src_ip and show me unique values'") 
        print("- 'find suspicious patterns in the firewall logs'")
        print("="*50)

        # Chat loop
        while True:
            print()
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAssistant: Goodbye!")
                break

            if not user_input:
                continue

            print("\nAssistant: Working with MCP tools...")

            try:
                # LangGraph invocation - directly copied from the working example
                messages = [HumanMessage(content=user_input)]
                result = await agent.ainvoke({"messages": messages})

                # Extract the final message
                final_message = result["messages"][-1]
                
                # Show which tools were used if any
                tool_messages = [msg for msg in result["messages"] if hasattr(msg, "tool_call_id")]
                if tool_messages:
                    tool_names = list(set(msg.tool for msg in tool_messages if hasattr(msg, "tool")))
                    if tool_names:
                        print(f"\nTools used: {', '.join(tool_names)}")
                
                print(f"\nAssistant: {final_message.content}")

            except Exception as e:
                print(f"\nAssistant: Error: {str(e)}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"Failed to connect to MCP server: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())