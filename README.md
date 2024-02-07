## Details

| Developed by | Guardrails AI |
| --- | --- |
| Date of development | Feb 15, 2024 |
| Validator type | RAG |
| Blog | https://www.guardrailsai.com/blog/reduce-ai-hallucinations-provenance-guardrails |
| License | Apache 2 |
| Input/Output | Output |

## Description

This validator uses embedding similarity to check if an LLM generated text can be supported by the sources.

## Example Usage Guide

### Installation

```bash
$ gudardrails hub install provenance-v0
```

### Initialization

```python
guard = Guard.from_string(validators=[
    ProvenanceV0(
			threshold=0.8,
			validation_method="sentence",
			embed_function=Callable
		)
])
```

### Invocation

```python
def embed_function(text: Union[str, List[str]]) -> np.ndarray:
    return np.array([[0.1, 0.2, 0.3]])

guard.parse(
    llm_output=...,
    metadata={"query_function": query_function}
)
```

## API Ref

- `threshold` - The minimum cosine similarity between the generated text and the source text. Defaults to 0.8.
- `validation_method` - Whether to validate at the sentence level or over the full text. Must be one of `sentence` or `full`. Defaults to `sentence`
    
    Other parameters: Metadata
    
- `query_function` *Callable, optional* - A callable that takes a string and returns a list of (chunk, score) tuples.
- `sources` *List[str], optional* - The source text.
- `embed_function` *Callable, optional* - A callable that creates embeddings for the sources. Must accept a list of strings and return an np.array of floats.
    
    In order to use this validator, you must provide either a `query_function` or `sources` with an `embed_function` in the metadata.
    
    If providing query_function, it should take a string as input and return a list of (chunk, score) tuples. The chunk is a string and the score is a float representing the cosine distance between the chunk and the input string. The list should be sorted in ascending order by score.
    

## Intended use

- Primary intended uses: For RAG applications, checking if a text is hallucinated by establishing a source (i.e. provenance) for any LLM generated text.
- Out-of-scope use cases: General question answering without RAG or text grounding.

## Expected deployment metrics

|  | CPU | GPU |
| --- | --- | --- |
| Latency | 1.5 seconds | - |
| Memory | n/a | - |
| Cost | token cost for LLM invocation | - |
| Expected quality | 80% | - |

## Resources required

- Dependencies: Embedding library
- Foundation model access keys: Yes (depending on which model is used for embeddings)
- Compute: N/A

## Validator Performance

### Evaluation Dataset

https://huggingface.co/datasets/miracl/hagrid

### Model Performance Measures

| Accuracy | 80% |
| --- | --- |
| F1 Score | 0.9 |

### Decision thresholds

0.5
