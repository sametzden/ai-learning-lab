# 01 - First LangChain App

First section of Eden Marco's *LangChain & LangGraph AI Engineer* course.

A `PromptTemplate` is piped into `ChatOpenAI` (`prompt | llm`) to turn a short bio into a summary and two interesting facts. Runs are traced in LangSmith.

## Run

Create a `.env` file in this folder:

```
OPENAI_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

Then:

```bash
uv sync
uv run main.py
```
