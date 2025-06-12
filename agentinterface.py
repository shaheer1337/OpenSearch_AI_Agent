import asyncio
import gradio as gr
import sys
import os
import re

# Ensure we can import the agent backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing agent
import agentbackend

# Track if agent is initialized
agent_initialized = False

async def init_agent():
    """Initialize the agent from the backend module"""
    global agent_initialized
    
    try:
        # Call the initialize_agent function from your existing code
        result = await agentbackend.initialize_agent()
        tool_names = result["tool_names"]
        agent_initialized = True
        return f"Connected! Found {len(tool_names)} tools: {', '.join(tool_names)}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

async def chat(message, history):
    """Send message to agent and get response"""
    global agent_initialized
    
    # Check if agent is initialized
    if not agent_initialized:
        return "", history + [[message, "Please initialize the agent first by clicking the 'Connect to Agent' button."]]
    
    try:
        # Call the process_query function from your existing code
        result = await agentbackend.process_query(message)
        response = result["response"]
        
        # Clean response (remove thinking sections)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()
        
        # Add info about tools used
        if result["tools_used"]:
            response += f"\n\n*Tools used: {', '.join(result['tools_used'])}*"
        
        # Add to history and return
        return "", history + [[message, response]]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", history + [[message, f"Error: {str(e)}"]]

# Create a simple interface
with gr.Blocks(title="OpenSearch Agent") as demo:
    gr.Markdown("# OpenSearch Agent")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500)
        
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", value="Not connected")
            init_button = gr.Button("Connect to Agent")
            gr.Markdown("### Examples")
            gr.Markdown("- list all indices")
            gr.Markdown("- show mapping of firewall-logs")
            gr.Markdown("- search for source IPs")
            clear_button = gr.Button("Clear Chat")
    
    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Ask about your OpenSearch data...",
            show_label=False,
            lines=2
        )
        send_button = gr.Button("Send", variant="primary")
    
    # Connect the components
    init_button.click(
        fn=lambda: asyncio.run(init_agent()),
        outputs=status
    )
    
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    send_button.click(chat, [msg, chatbot], [msg, chatbot])
    clear_button.click(lambda: None, None, chatbot)

# Run the app
if __name__ == "__main__":
    print("Starting OpenSearch Agent Interface")
    print("Open your browser to http://localhost:7860")
    
    demo.launch(server_name="0.0.0.0", server_port=7860)