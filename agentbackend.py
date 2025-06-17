import asyncio
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
@@ -8,17 +9,43 @@
# Load environment variables
load_dotenv()

# Define system prompt
# Define system prompt with error handling guidance
system_prompt = """You're an AI assistant with access to OpenSearch tools.
You can answer questions directly when appropriate, or use tools when needed.
For data-specific questions, use the appropriate tools.
For multi-step analysis, combine tools in sequence to solve complex problems.

TOOL SELECTION GUIDANCE:
- Use ListIndexTool to discover available indices
- Use IndexMappingTool to understand the structure and fields in an index
- Use SearchIndexTool for simple document retrieval and basic filtering
- Use WebSearchTool when you need information not available in the indices


When you need multiple tools to answer a question:
1. Use tools in a logical sequence
2. Use the results of one tool to inform the next tool call
3. Synthesize information from all tools in your final answer

IMPORTANT: When a tool returns an error, explain the issue clearly to the user.
Common errors include:
- Non-existent indices
- Invalid query syntax
- Missing required parameters
- Permission issues

Remember previous interactions with the user and maintain context throughout the conversation.
For conversation, just respond normally."""

@@ -36,6 +63,7 @@
client = None
conversation_history = [SystemMessage(content=system_prompt)]
is_initialized = False  # Track if we've initialized before
debug_mode = True  # Enable error display for development

async def initialize_agent():
    """Initialize the MCP client and agent"""
@@ -73,7 +101,7 @@ async def initialize_agent():

async def process_query(user_input):
    """Process a user query and return results"""
    global agent, conversation_history
    global agent, conversation_history, debug_mode

    if agent is None:
        await initialize_agent()
@@ -96,13 +124,86 @@ async def process_query(user_input):
    if tool_messages:
        tool_names = list(set(msg.tool for msg in tool_messages if hasattr(msg, "tool")))

    # Extract tool errors from the result
    tool_errors = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_call_id") and hasattr(msg, "tool"):
            tool_name = msg.tool
            
            # Check if there's an error in the tool output
            if hasattr(msg, "tool_output"):
                output = msg.tool_output
                error_found = False
                error_msg = None
                
                # Handle string outputs that might be JSON
                if isinstance(output, str):
                    try:
                        # Try to parse as JSON
                        json_data = json.loads(output)
                        
                        # Check for OpenSearch error format
                        if "status" in json_data and "error" in json_data:
                            error_obj = json_data["error"]
                            error_type = error_obj.get("type", "Unknown Error")
                            error_reason = error_obj.get("reason", "")
                            error_details = error_obj.get("details", "")
                            
                            error_msg = f"Error Type: {error_type}\nReason: {error_reason}\nDetails: {error_details}"
                            error_found = True
                        
                        # Check for other error indicators
                        elif "error" in json_data:
                            error_msg = str(json_data["error"])
                            error_found = True
                    
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON or couldn't parse, check for error keywords
                        if "error" in output.lower() or "exception" in output.lower():
                            error_msg = output
                            error_found = True
                
                # Handle direct dict outputs
                elif isinstance(output, dict):
                    # Check for OpenSearch error format
                    if "status" in output and "error" in output:
                        error_obj = output["error"]
                        error_type = error_obj.get("type", "Unknown Error")
                        error_reason = error_obj.get("reason", "")
                        error_details = error_obj.get("details", "")
                        
                        error_msg = f"Error Type: {error_type}\nReason: {error_reason}\nDetails: {error_details}"
                        error_found = True
                    
                    # Check for other error indicators
                    elif any(key in output for key in ["error", "errorMessage", "exception"]):
                        error_msg = output.get("error") or output.get("errorMessage") or output.get("exception")
                        error_found = True
                
                # If we found an error, add it to the list
                if error_found and error_msg:
                    tool_errors.append({
                        "tool": tool_name,
                        "error": error_msg
                    })
    
    # Ensure conversation history doesn't grow too large (keep last 10 messages)
    if len(conversation_history) > 12:  # system message + 10 exchanges
        conversation_history = [conversation_history[0]] + conversation_history[-11:]

    # For development mode, add error details to the response
    response = final_message.content
    if debug_mode and tool_errors:
        error_details = "\n\n--- DEVELOPMENT INFO: TOOL ERRORS ---\n"
        for error in tool_errors:
            error_details += f"[ERROR] Tool: {error['tool']}\n"
            error_details += f"{error['error']}\n\n"
        response += error_details
    
    return {
        "response": final_message.content,
        "response": response,
        "tools_used": tool_names,
        "tool_errors": tool_errors if debug_mode else [],
        "raw_result": result
    }

@@ -119,6 +220,12 @@ async def main():
        print(f"Response: {result['response']}")
        if result['tools_used']:
            print(f"Tools used: {', '.join(result['tools_used'])}")
        
        # Display errors in the terminal
        if 'tool_errors' in result and result['tool_errors']:
            print("\nTool Errors:")
            for error in result['tool_errors']:
                print(f"- {error['tool']}: {error['error']}")

if __name__ == "__main__":
    asyncio.run(main())