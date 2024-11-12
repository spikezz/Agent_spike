# Agent_spike

A chatbot which integrate a bunch of toolings

# How to build own knowledgebase

## Example Paper "Attention Is All You Need"

https://arxiv.org/pdf/1706.03762

## 1. Tool to converts PDF to markdown: Marker

https://github.com/VikParuchuri/marker

### install

> pip install marker-pdf

### run cmd

> marker_single /path/to/pdf/file.pdf /path/to/output/folder --batch_multiplier 4 --max_pages 60 --langs English

play with flags like batch_multiplier to meet your own need.

## 2. Graphrag

https://github.com/microsoft/graphrag

https://microsoft.github.io/graphrag/

### install

> pip install graphrag

### bootstrap

start with project folder structure:
```
.
├── doc
│   ├── input
│   │   └── paper.md
```
### run cmd

>python -m graphrag.index  --init  --root /path/to/doc/

### settings.yaml

insert your own api key according to needs.

embeddings model please use openai text-embedding-3-small with:
```
api_base: https://api.openai.com/v1
```
The tokens_per_minute and requests_per_minute of the first version of graphrag does not work, so we need hacking for TPM and RPM:
site-packages/graphrag/index/llm/load_llm.py
```
def _get_base_config(config: dict[str, Any]) -> dict[str, Any]:
    api_key = config.get("api_key")

    return {
        # Pass in all parameterized values
        **config,
        # Set default values
        "api_key": api_key,
        "api_base": config.get("api_base"),
        "api_version": config.get("api_version"),
        "organization": config.get("organization"),
        "proxy": config.get("proxy"),
        "max_retries": config.get("max_retries", 10),
        "request_timeout": config.get("request_timeout", 60.0),
        "model_supports_json": config.get("model_supports_json"),
        "concurrent_requests": config.get("concurrent_requests", 4),
        "encoding_model": config.get("encoding_model", "cl100k_base"),
        "cognitive_services_endpoint": config.get("cognitive_services_endpoint"),
        "requests_per_minute":28, # add this line
        "tokens_per_minute":6000, # add this line
    }
```
show dialog:
site-packages/graphrag/llm/openai/openai_chat_llm.py
```

class OpenAIChatLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A Chat-based LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        history = kwargs.get("history") or []
        messages = [
            *history,
            {"role": "user", "content": input},
        ]
        completion = await self.client.chat.completions.create(
            messages=messages, **args
        )
        print("Prompt:")
        print(input)
        print("Response:")
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content
```

here is the several combination for the rest settings:

#### groq

  ```settings.yaml
  api_base: https://api.groq.com/openai/v1
  model: llama-3.1-405b-reasoning
  ```
  
  limits for hack: 
  https://console.groq.com/settings/limits

#### ollama

  ```settings.yaml
  api_base: http://localhost:11434/v1/
  model: llama3.1:70b-instruct-q6_K
  request_timeout: 500.0
  max_retries: 30
  max_retry_wait: 20.0
  concurrent_requests: 1
  ```

#### deepseek

  ```settings.yaml
  api_base: https://api.deepseek.com
  model: deepseek-coder
  ```
  ```site-packages/graphrag/index/llm/load_llm.py
  "requests_per_minute":28, # add this line
  "tokens_per_minute":1000, # add this line 17*60
  ```
#### openai

  ```settings.yaml
  api_base: https://api.openai.com/v1
  model: gpt-4o-mini
  ```
  ```site-packages/graphrag/index/llm/load_llm.py
  "requests_per_minute":400, # add this line
  "tokens_per_minute":10000, # add this line
  ```

### run cmd

>python -m graphrag.index  --root /path/to/doc/

### run script to convert parquet into csv

>python agent/parquet_to_csv.py --parquet_dir /path/to/parquet/artifacts --csv_dir /path/to/output/import

### xml graph visu

> python xml_graph.py -f x.xml

### install neo4j on linux

https://neo4j.com/download/

or

https://neo4j.com/docs/operations-manual/current/installation/linux/debian/#debian-installation

### modify neo4j conf
```
server.directories.import=/path/to/output/import
dbms.security.allow_csv_import_from_file_urls=true
```
### run neo4j cmd

> ./neo4j-desktop-1.6.0-x86_64.AppImage

or

> neo4j start

### run query in neo4j browser

copy paste content from neo4j_query.txt

