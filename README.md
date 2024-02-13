## Overview

| Developed by | Guardrails AI |
| --- | --- |
| Date of development | Feb 15, 2024 |
| Validator type | RAG |
| Blog | https://www.guardrailsai.com/blog/reduce-ai-hallucinations-provenance-guardrails |
| License | Apache 2 |
| Input/Output | Output |

## Description

This validator uses embedding similarity to check if an LLM generated text can be supported by the sources. In order for this validator to work, a list of sources must be provided to which the LLM generated text can be  attributed. In order to use the validator, you must provide either of the following in the metadata:
- a `query_function`: The `query_function` function should take a string as input (the LLM-generated text) and return a list of relevant chunks. The list should be sorted in ascending order by the distance between the chunk and the LLM-generated text.
2. `sources` and `embed_function`: The `embed_function` should take a string or a list of strings as input and return a np array of floats. The vector should be normalized to unit length.

Below is a step-wise breakdown of how the validator works:
1. The list of sources is chunked based on user's parameters. 
2. Each source chunk is embedded and stored in a vector database or an in-memory embedding store.
3. The LLM generated output is chunked based on user-specified parameters.
4. Each LLM output chunk is embedded by the same model used for embedding source chunks.
5. The cosine distance is computed between the LLM output chunk and the `k` nearest source chunks. If the cosine distance is less than a certain threshold, then the LLM-generated text is considered not hallucinated.

### Intended use

The primary intended uses are for RAG applications -- to check if a text is hallucinated by establishing a source (i.e. provenance) for any LLM generated text. Out of scope use cases are general question answering without RAG or text grounding.

### Resources required

- Dependencies: Embedding library
- Foundation model access keys: Yes (depending on which model is used for embeddings)

## Installation

```bash
$ gudardrails hub install hub://guardrails/provenance_embeddings
```

## Example Usage Guide

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails.hub import ProvenanceEmbeddings
from guardrails import Guard

# Import embedding model
from sentence_transformers import SentenceTransformer

# Initialize Validator
val = ProvenanceEmbeddings(
    threshold=0.8,
    validation_method="sentence",
    embed_function=Callable
)

# Setup Guard
guard = Guard.from_string(validators=[val])

# Setup text sources
sources = [
    "The sun is a star.",
    "The sun rises in the east and sets in the west."
]

# Load model for embedding function
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create embed function
def embed_function(sources: list[str]) -> np.array:
		return model.encode(sources)

guard.parse(
    llm_output="The sun rises in the east.",
    metadata={
        "sources": sources,
        "embed_function": embed_function
    }
)
```

### Validating JSON output via Python

In this example, we apply the validator to a string field of a JSON output generated by an LLM.

```python
# Import Guard and Validator
from pydantic import BaseModel
from guardrails.hub import ProvenanceEmbeddings
from guardrails import Guard

# Import embedding model
from sentence_transformers import SentenceTransformer

# Initialize Validator
val = ProvenanceEmbeddings(
    threshold=0.8,
    validation_method="sentence",
    embed_function=Callable
)

# Create Pydantic BaseModel
class LLMOutput(BaseModel):
    output: str = Field(
        description="Output generated by LLM", validators=[val]
    )

# Create a Guard to check for valid Pydantic output
guard = Guard.from_pydantic(output_class=LLMOutput)

# Setup text sources
sources = [
    "The sun is a star.",
    "The sun rises in the east and sets in the west."
]

# Load model for embedding function
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create embed function
def embed_function(sources: list[str]) -> np.array:
		return model.encode(sources)

# Run LLM output generating JSON through guard
guard.parse("""
{
    "output": "The sun rises in the east."
},
""",
    metadata={
        "sources": sources,
        "embed_function": embed_function
    }
)
```

## API Reference

`__init__`
- `threshold` - The minimum cosine similarity between the generated text and the source text. Defaults to 0.8.
- `validation_method` - Whether to validate at the sentence level or over the full text. Must be one of `sentence` or `full`. Defaults to `sentence`
    
`__call__`    
- `query_function` *Callable, optional* - A callable that takes a string and returns a list of (chunk, score) tuples. In order to use this validator, you must provide either a `query_function` or `sources` with an `embed_function` in the metadata. The query_function should take a string as input and return a list of (chunk, score) tuples. The chunk is a string and the score is a float representing the cosine distance between the chunk and the input string. The list should be sorted in ascending order by score.
- `sources` *List[str], optional* - The source text. In order to use this validator, you must provide either a `query_function` or `sources` with an `embed_function` in the metadata. 
- `embed_function` *Callable, optional* - A callable that creates embeddings for the sources. Must accept a list of strings and return an np.array of floats.
