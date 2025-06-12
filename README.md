# OpenSearch AI Agent

An AI assistant that interacts with OpenSearch using LangGraph and MCP tools.

## Features

- Natural language interface to OpenSearch
- Tool-based architecture using MCP (Model, Connector, Plugin)
- Support for multiple query types including DSL and PPL
- Memory to maintain conversation context
- Terminal and web interface options

## Prerequisites

- Python 3.9+
- OpenSearch instance running
- Ollama for local LLM inference

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/OpenSearch-AI-Agent.git
   cd OpenSearch-AI-Agent
   ```

2. Create a venv:
   ```python
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```python
   pip install langchain langchain-community langchain-mcp-adapters python-dotenv gradio
   pip install langchain-ollama langgraph
   ```

4. Netty4:
   ```bash
   docker exec -it opensearch-node1 ./bin/opensearch-plugin install transport-reactor-netty4
   ```

5. Set initial admin password:
   ```bash
   export OPENSEARCH_INITIAL_ADMIN_PASSWORD=
   ```

