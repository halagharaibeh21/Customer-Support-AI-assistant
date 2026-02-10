# Customer Support Chatbot

## Problem Statement
Modern businesses face high volumes of customer queries. Manual support is slow, inconsistent, and expensive. Customers expect instant and accurate responses, which is difficult to provide at scale.

This project addresses the need for an automated, reliable, and fast customer support system.

---

## Solution Overview
The Customer Support Chatbot uses Azure OpenAI to generate intelligent responses to customer questions. It can handle multiple queries in a single session and provides clear, context-aware answers.

**Key points:**
- Uses Azure API for natural language processing
- Reads API key from environment variable to keep credentials secure
- Designed for reproducibility and easy setup

---

## How to Run
Set your Azure API key in the environment and run the chatbot in **one command**:

```bash
export AZURE_API_KEY="your_azure_key_here" && python chatbot.py
