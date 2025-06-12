import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama  # Changed from OllamaLLM
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

system_prompt = """You're an AI assistant with access to OpenSearch tools.
You can answer questions directly when appropriate, or use tools when needed.
For data-specific questions, use the appropriate tools.
For conversation, just respond normally."""

model = ChatOllama(
    model="qwen3:14b",
    base_url="http://192.168.18.201:11434",
    temperature=0.1,
    system=system_prompt  # Add this line
)

async def main():
    print("Starting Pure MCP OpenSearch Agent with LangGraph...")

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
        print(f"Found {len(tools)} MCP tools: {[tool.name for tool in tools]}")

        if not tools:
            print("No tools found! Check MCP server registration.")
            return

        # Create LangGraph agent with MCP tools
        agent = create_react_agent(model, tools)

        print("\n" + "="*50)
        print("Pure MCP OpenSearch AI Assistant Ready")
        print(f"MCP Tools: {[tool.name for tool in tools]}")
        print("="*50)
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Examples:")
        print("- 'list all indices'")
        print("- 'get mapping of firewall-logs-+000000'")
        print("- 'search firewall-logs-+000000 for all documents'")
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
                # LangGraph invocation
                messages = [HumanMessage(content=user_input)]
                result = await agent.ainvoke({"messages": messages})

                # Extract the final message
                final_message = result["messages"][-1]
                print(f"\nAssistant: {final_message.content}")

            except Exception as e:
                print(f"\nAssistant: MCP Error: {str(e)}")
                print("This might be a parameter serialization issue.")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"Failed to connect to MCP server: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())