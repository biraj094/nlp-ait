# A7 - MCP with n8n

## Task 1: MCP Infrastructure & Server Setup

### Overview

Two workflows were created in n8n to implement an MCP (Model Context Protocol) server and an AI agent client that communicates with it over a public internet URL via ngrok.

---

### Workflow 1: MCP Server

The MCP Server workflow exposes three tools via an MCP Server Trigger node. The production URL is tunneled to the internet using ngrok, making it accessible from external clients.

![MCP Server Workflow](./workflow1.png)

**Tools implemented:**

---

#### Tool 1: Calculator

Uses n8n's built-in Calculator node. Performs arithmetic on two numbers. Invoked when the user asks for a math calculation.

![Calculator Tool](./tool1.png)

---

#### Tool 2: Date & Time

Uses n8n's built-in Date & Time node. Returns the current date and time. Invoked when the user asks for the current time or date.

![Date & Time Tool](./tool2.png)

---

#### Tool 3: Crypto Hash

Uses n8n's built-in Crypto node. Generates an MD5 hash of a given input text. Invoked when the user asks to hash a string.

![Crypto Hash Tool](./tool3.png)

---

### ngrok Tunnel

The MCP Server's production URL is exposed publicly via ngrok, allowing the AI Agent workflow to connect to it from anywhere over the internet.

![ngrok Tunnel](./ngrok.png)

---

### Workflow 2: AI Agent Client

The AI Agent workflow provides a chat interface that connects to the MCP Server using the MCP Client tool node.

![AI Agent Workflow](./workflow2.png)

**Components:**

- **Chat Trigger:** Receives user messages via the n8n chat interface
- **AI Agent:** Orchestrates tool calling and response generation
- **LLM:** Groq API using model `meta-llama/llama-4-scout-17b-16e-instruct`
- **Window Buffer Memory:** Maintains conversation context across turns (Simple Memory node)
- **MCP Client:** Connects to the MCP Server's production URL via SSE transport and exposes all three tools to the agent

---

### Verification

The AI agent successfully calls MCP tools in response to user prompts:

| User Prompt | Tool Called | Result |
|---|---|---|
| `49*49` | Calculator | `2401` |
| `what is the time right now?` | Date & Time | `3:43:50 AM on March 29, 2026` |
| `Please hash this text "NLP"` | Crypto | `9943ce409ce385195d913731919052f5f0404130...` |

---

### Setup Summary

| Component | Details |
|---|---|
| n8n version | 2.13.4 (Self Hosted, Docker) |
| Tunnel | ngrok v3.37.3 (Free tier, Asia Pacific) |
| MCP Transport | SSE |
| LLM Provider | Groq |
| Model | meta-llama/llama-4-scout-17b-16e-instruct |
| Memory | Window Buffer (Simple Memory) |
| Tools | Calculator, Date & Time, Crypto Hash |
