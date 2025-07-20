# Compliance-Aware Content Moderator with Semantic Kernel

This project implements a **Compliance-Aware Content Moderator** as a Proof of Concept (PoC) using **Semantic Kernel (SK)** in a Google Colab notebook. The moderator evaluates user-generated content (e.g., forum posts) against a dynamic set of policies (static, scraped via Firecrawl, and fetched via News API) using Retrieval-Augmented Generation (RAG). It leverages **HuggingFace** for embeddings, **Groq** (primary) or **OpenAI** (fallback) for moderation, and anonymized logging for enterprise-grade compliance. The PoC is lightweight, stateless, and designed for enterprise use cases, showcasing SK’s strengths over alternatives like LangChain, LlamaIndex, or Haystack.

This README covers:
1. **Semantic Kernel Concepts**: Detailed explanations with examples and samples.
2. **Workflow and RAG in SK**: How the PoC works with a block diagram.
3. **Notebook PoC Explanation**: Overview of the Colab implementation.
4. **Code Workflow**: Step-by-step code explanation with a flowchart.
5. **Unique SK Features**: What sets SK apart from other frameworks.

---

## Semantic Kernel Concepts

Semantic Kernel (SK) is a lightweight framework for building AI-powered applications, designed for enterprise-grade integration, low-latency stateless tasks, and modularity. Below, we explain each SK concept with examples tailored for an 18-year-old with basic Python and LangChain knowledge, along with sample code snippets.

### 1. Kernel, Function, Plugin, Skill
**What It Is**: The **Kernel** is the core of SK, like the brain of your app, coordinating plugins, AI services, and memory. **Plugins** are modular tools (like apps on your phone), and **skills** are tasks within plugins, either **native** (Python code) or **semantic** (AI prompts). **Functions** are the specific actions skills perform.

**Example**: Imagine you’re building a chatbot. The Kernel is the chatbot’s brain, managing plugins like a “WeatherPlugin” (to check weather) and a “ChatPlugin” (to respond to users). The WeatherPlugin might have a native skill to fetch data from a weather API and a semantic skill to summarize the forecast.

**Sample**:
```python
import semantic_kernel as sk

kernel = sk.Kernel()  # Initialize Kernel (the brain)
class WeatherPlugin:
    @sk.function(name="get_weather")
    async def get_weather(self, city: str) -> str:
        return f"Weather in {city}: Sunny!"  # Native skill

kernel.import_plugin_from_object(WeatherPlugin(), "Weather")
result = asyncio.run(kernel.run_async(
    kernel.get_plugin("Weather").get_function("get_weather"),
    input_vars={"city": "Seattle"}
))
print(result)  # Output: Weather in Seattle: Sunny!
```

**In PoC**: The Kernel manages plugins like `ModerationPlugin` (semantic, evaluates content) and `ComplianceLoggerPlugin` (native, logs decisions).

### 2. Memory (SemanticTextMemory, Embedding support)
**What It Is**: **SemanticTextMemory** stores data as **embeddings** (numerical representations of text) in a vector store (like SK’s FAISS connector), enabling RAG to retrieve relevant information. It’s like a smart librarian finding books that match your question.

**Example**: If you store restaurant reviews, you can ask, “Find positive reviews,” and the memory retrieves relevant ones based on embeddings.

**Sample**:
```python
from semantic_kernel.connectors.memory.faiss import FaissMemoryStore
from pydantic import BaseModel

class Review(BaseModel):
    text: str
    embedding: list

memory_store = FaissMemoryStore()
collection = memory_store.get_collection("reviews", Review)
embedding = [0.1, 0.2, 0.3]  # Example embedding
asyncio.run(collection.upsert([("1", Review(text="Great food!", embedding=embedding))]))
results = asyncio.run(collection.query(query_embeddings=[embedding], k=1))
print(results[0][0].text)  # Output: Great food!
```

**In PoC**: Policies are stored in SK’s FAISS as `PolicyRecord` objects, retrieved for moderation using HuggingFace embeddings.

### 基本的### 3. Skills (Native & Semantic)
**What It Is**: **Skills** are reusable tools within plugins. **Native skills** are Python code (e.g., API calls), and **semantic skills** are AI prompts (e.g., summarization).

**Example**: A native skill might fetch data from an API, while a semantic skill might summarize text using an LLM.

**Sample**:
```python
class SummaryPlugin:
    @sk.function(name="summarize")
    async def summarize(self, text: str) -> str:
        prompt = f"Summarize: {{ $text }}"
        response = await kernel.run_async(
            kernel.create_semantic_function(prompt, max_tokens=50),
            input_vars={"text": text}
        )
        return response.result
```

**In PoC**: `ModerationPlugin` (semantic) evaluates content, and `ComplianceLoggerPlugin` (native) logs decisions.

### 4. Planner (Sequential & Stepwise)
**What It Is**: A **Planner** breaks complex tasks into steps. A **Sequential Planner** follows a fixed order, while a **Stepwise Planner** dynamically adjusts steps, like a smart to-do list.

**Example**: For “Plan a trip,” the planner might create steps: search flights, book hotel, plan activities.

**Sample**: A simple sequential planner might chain functions:
```python
async def plan_trip(destination):
    plan = [
        {"plugin": "Search", "function": "search_flights", "inputs": {"destination": destination}},
        {"plugin": "Booking", "function": "book_hotel", "inputs": {"destination": destination}}
    ]
    for step in plan:
        await kernel.run_async(kernel.get_plugin(step["plugin"]).get_function(step["function"]), input_vars=step["inputs"])
```

**In PoC**: A Stepwise Planner chains policy retrieval, moderation, and logging.

### 5. Prompt Templates
**What It Is**: **Prompt Templates** are reusable AI instructions with placeholders for dynamic data, like a form with blanks.

**Example**: A template like `Hello, {{ $name }}!` can be filled with different names.

**Sample**:
```python
prompt = "Summarize: {{ $text }}"
semantic_function = kernel.create_semantic_function(prompt, max_tokens=50)
result = asyncio.run(semantic_function(input_vars={"text": "Long story about a cat..."}))
print(result)  # Output: Short summary of the cat story.
```

**In PoC**: The `ModerationPlugin` uses a prompt template to evaluate content.

### 6. Function Calling (KernelFunction.InvokeAsync)
**What It Is**: **Function Calling** runs plugins asynchronously using `InvokeAsync`, ensuring fast execution.

**Example**: Calling a weather function to get data quickly.

**Sample**: See the `WeatherPlugin` sample above.

**In PoC**: Functions like `moderate` and `log` are called asynchronously.

### 7. Skill Chaining
**What It Is**: **Skill Chaining** links plugins into a pipeline, like a conveyor belt of tasks.

**Example**: Chaining a search plugin to a summary plugin to process search results.

**Sample**: See the `plan_trip` sample above.

**In PoC**: Chains policy retrieval, moderation, and logging.

### 8. Input/Output Binding (context.Variables["key"])
**What It Is**: **Input/Output Binding** passes data between steps using a dictionary (`context.Variables`), like passing notes.

**Example**: Passing a search query to a search plugin, then its results to a summary plugin.

**Sample**:
```python
context = {"query": "cats"}
result = await kernel.run_async(
    kernel.get_plugin("Search").get_function("search"),
    input_vars=context
)
context["results"] = result.result
```

**In PoC**: Passes policies to moderation and results to logging.

### 9. External Integration
**What It Is**: SK connects to external services like APIs or LLMs.

**Example**: Using OpenAI for text generation or a weather API for data.

**Sample**: See the custom LLM connector in the PoC code.

**In PoC**: Uses Groq, OpenAI, HuggingFace, Firecrawl, and News API.

### 10. Pluggability
**What It Is**: **Pluggability** allows adding/removing plugins without changing core logic, like adding apps to a phone.

**Example**: Adding a new plugin to translate text without rewriting the app.

**Sample**: See the plugin import in the PoC code.

**In PoC**: Plugins like `PolicyScraperPlugin` can be replaced or extended.

---

## Workflow and RAG in Semantic Kernel

### Workflow Overview
The Compliance-Aware Content Moderator evaluates user content (e.g., “This service is terrible!”) against policies:
1. **Input**: User submits content.
2. **Policy Retrieval**: Retrieves relevant policies (static, Firecrawl, News API) from SK’s FAISS using RAG.
3. **Moderation**: Uses Groq/OpenAI to evaluate content against policies.
4. **Logging**: Anonymously logs the decision (timestamp, content length, decision).
5. **Output**: Returns “Approved” or “Flagged” with a reason.

### How RAG Works in Semantic Kernel
**Retrieval-Augmented Generation (RAG)** combines information retrieval with text generation:
- **Retrieval**: Policies are stored as embeddings in SK’s FAISS connector. The app retrieves relevant policies based on the content’s embedding.
- **Augmentation**: Retrieved policies are added to the LLM prompt.
- **Generation**: The LLM (Groq/OpenAI) generates a moderation decision.

**Block Diagram** (Mermaid syntax):
```mermaid
graph TD
    A[User Content] --> B[Generate Embedding<br>(HuggingFace)]
    B --> C[Retrieve Policies<br>(SK FAISS)]
    C --> D[Augment Prompt<br>(Policies + Content)]
    D --> E[Generate Decision<br>(Groq/OpenAI)]
    E --> F[Log Decision<br>(ComplianceLoggerPlugin)]
    E --> G[Output Decision]
```

---

## Notebook PoC Explanation

### Overview
The Google Colab notebook implements the PoC with the following steps:
1. **Install Libraries**: Installs `semantic-kernel[faiss]`, `sentence-transformers`, `groq`, `openai`, `firecrawl-sdk`, `newsapi-python`, `streamlit`, `pyngrok`.
2. **Import Libraries and API Keys**: Sets up APIs for Groq, OpenAI, Firecrawl, and News API.
3. **Initialize Kernel and LLM Connector**: Configures the Kernel with a custom Groq/OpenAI connector.
4. **Embed Policies**: Stores static, scraped (Firecrawl), and news (News API) policies in SK’s FAISS using HuggingFace embeddings.
5. **Define Plugins**: Creates `PolicyScraperPlugin`, `NewsFetcherPlugin`, `ModerationPlugin`, and `ComplianceLoggerPlugin`.
6. **Implement Planner**: Uses a Stepwise Planner to chain retrieval, moderation, and logging.
7. **Test Moderator**: Tests content moderation and logs results.
8. **Streamlit App (Optional)**: Creates a web interface for user interaction.

### Why Semantic Kernel?
SK is chosen for its:
- **Lightweight Design**: Ideal for low-latency, stateless RAG tasks.
- **Enterprise Focus**: Compliance features (anonymized logging, stateless processing).
- **Modularity**: Reusable plugins and tight control via the Kernel.
- **Integration**: Seamless use with Microsoft ecosystems and custom APIs.

---

## Code Workflow

### Step-by-Step Workflow
1. **Setup**: Installs libraries and loads API keys.
2. **Kernel Initialization**: Configures the Kernel with a Groq/OpenAI connector.
3. **Policy Embedding**: Embeds policies (static, Firecrawl, News API) into SK’s FAISS.
4. **Plugin Execution**:
   - Retrieves policies using FAISS.
   - Moderates content using the `ModerationPlugin`.
   - Logs decisions using the `ComplianceLoggerPlugin`.
5. **Output**: Displays moderation result and logs.

### Flowchart (Mermaid syntax)
```mermaid
graph TD
    A[Start: Input Content] --> B[Initialize Kernel<br>(Groq/OpenAI)]
    B --> C[Embed Policies<br>(HuggingFace, SK FAISS)]
    C --> D[Retrieve Policies<br>(SemanticTextMemory)]
    D --> E[Moderate Content<br>(ModerationPlugin)]
    E --> F[Log Decision<br>(ComplianceLoggerPlugin)]
    F --> G[Output Result]
    G --> H[Optional: Streamlit App]
```

---

## Additional Components and Unique SK Features

### Additional Components
- **Firecrawl Integration**: Scrapes policy documents from websites, enhancing the dynamic policy set.
- **News API Integration**: Fetches real-time regulatory updates, ensuring up-to-date compliance.
- **Streamlit App**: Provides a user-friendly web interface (optional).
- **Custom LLM Connector**: Combines Groq (fast) and OpenAI (reliable) for robust moderation.

### Unique Features of Semantic Kernel
Compared to **LangChain**, **LlamaIndex**, and **Haystack**, SK offers:
- **Lightweight and Stateless**: Optimized for low-latency RAG tasks, unlike LangChain’s heavier agent frameworks.
- **Enterprise-Grade Compliance**: Stateless processing and anonymized logging (e.g., `ComplianceLoggerPlugin`) are ideal for secure enterprise applications, surpassing LangChain’s LangSmith observability.
- **Tight Control**: The Kernel and Planners provide precise orchestration, unlike LlamaIndex’s RAG-focused pipelines or Haystack’s flexible but less controlled pipelines.
- **Plugin Modularity**: Plugins like `ModerationPlugin` are highly reusable across projects, offering more modularity than LangChain’s tools.
- **Microsoft Ecosystem Integration**: Seamless embedding in .NET/Azure apps, a strength not matched by Python-centric frameworks.
- **Pluggability**: Easy addition of plugins (e.g., `PolicyScraperPlugin`) without core changes, more streamlined than Haystack’s pipeline modifications.

---

### Running the Notebook
1. **Copy to Colab**: Paste each code cell into a new Google Colab notebook.
2. **Replace API Keys**:
   - Groq: https://console.groq.com
   - OpenAI: https://platform.openai.com
   - Firecrawl: https://www.firecrawl.dev
   - News API: https://newsapi.org
3. **Run Cells**: Execute cells in order.
4. **Streamlit (Optional)**: Use `ngrok` for Colab or run locally with `streamlit run app.py`.
5. **Experiment**:
   - Add new plugins (e.g., `TrendAnalysisPlugin`).
   - Modify the moderation prompt.
   - Use SK’s `StepwisePlanner` for dynamic planning.

### Extending the PoC
- **FastAPI Endpoint**:
  ```python
  from fastapi import FastAPI
  app = FastAPI()
  @app.post("/moderate")
  async def moderate_content(content: str):
      return {"result": await plan_and_execute(content)}
  ```
- **Production Deployment**: Deploy FastAPI on a cloud server (e.g., AWS, Azure).

---

### Conclusion
This PoC demonstrates Semantic Kernel’s power for enterprise-grade, lightweight, stateless RAG applications. It covers all SK concepts, leverages multiple APIs, and provides a foundation for production-ready apps. For support, refer to the [Semantic Kernel documentation](https://learn.microsoft.com/en-us/semantic-kernel/) or contact the developer.