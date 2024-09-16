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
1. a `query_function`: The `query_function` function should take a string as input (the LLM-generated text) and return a list of relevant chunks. The list should be sorted in ascending order by the distance between the chunk and the LLM-generated text.
2. `sources` and `embed_function`: The `embed_function` should take a string or a list of strings as input and return a np array of floats. The vector should be normalized to unit length.

Below is a step-wise breakdown of how the validator works:
1. The list of sources is chunked based on user's parameters. 
2. Each source chunk is embedded and stored in a vector database or an in-memory embedding store.
3. The LLM generated output is chunked based on user-specified parameters.
4. Each LLM output chunk is embedded by the same model used for embedding source chunks.
5. The cosine distance is computed between the LLM output chunk and the `k` nearest source chunks. If the cosine distance is less than a certain threshold, then the LLM-generated text is considered attributable to the source text.

### Intended use

The primary intended uses are for RAG applications -- to check if a text is hallucinated by establishing a source (i.e. provenance) for any LLM generated text. Out of scope use cases are general question answering without RAG or text grounding. You can use a combination of RAG and this validator on the output to implement Retrieval-Augmented Validated Generation (RAVG).

### Requirements

* Dependencies:
    - `numpy`
    - `nltk`
    - guardrails-ai>=0.4.0


* To use in an example:
    - `sentence-transformers`
    - `chromadb`

* Foundation model access keys: 
    - Yes (depending on which model is used for embeddings)

## Installation

```bash
$ guardrails hub install hub://guardrails/provenance_embeddings
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails.hub import ProvenanceEmbeddings
from guardrails import Guard
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "This example requires the `sentence-transformers` package. "
        "Install it with `pip install sentence-transformers`, and try again."
    )

# Setup text sources
SOURCES = [
    "The sun is a star.",
    "The sun rises in the east and sets in the west.",
    "Sun is the largest object in the solar system, and all planets revolve around it.",
]
# Load model for embedding function
MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")
# Create embed function
def embed_function(sources: list[str]) -> np.array:
    return MODEL.encode(sources)

# Use the Guard with the validator
guard = Guard().use(
    ProvenanceEmbeddings,
    threshold=0.2,  # Lower the threshold to make the validator stricter
    validation_method="sentence",
    on_fail="exception",
)


# Test passing response
guard.validate(
    """
    The sun is a star that rises in the east and sets in the west.
    """,
    metadata={"sources": SOURCES, "embed_function": embed_function},
)

try:
    # Test failing response
    guard.validate(
        """
        Pluto is the farthest planet from the sun.
        """,  # This sentence is not "false", but is still NOT supported by the sources
        metadata={"sources": SOURCES, "embed_function": embed_function},
    )
except Exception as e:
    print(e)
```
Output:
```console
Validation failed for field with errors: None of the following sentences in your response are supported by provided context:
- Pluto is the farthest planet from the sun.
```

# API Reference


**`__init__(self, threshold=0.8, validation_method="sentence", on_fail="noop")`**
<ul>

Initializes a new instance of the Validator class.

**Parameters:**

- **`threshold`** _(float):_ The minimum cosine distance between the generated text and the source text. Defaults to 0.8.
- **`validation_method`** _(str):_ Whether to validate at the sentence level or over the full text. Must be one of `sentence` or `full`. Defaults to `sentence`
- **`on_fail`** *(str, Callable):* The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.

</ul>

<br>

**`__call__(self, value, metadata={}) -> ValidationResult`**

<ul>

Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters:**

- **`value`** *(Any):* The input value to validate.
- **`metadata`** *(dict):* A dictionary containing metadata required for validation. Keys and values must match the expectations of this validator.
    
    
    | Key | Type | Description | Default |
    | --- | --- | --- | --- |
    | `query_function` | _Optional[Callable]_ | A callable that takes a string and returns a list of (chunk, score) tuples. In order to use this validator, you must provide either a `query_function` or `sources` with an `embed_function` in the metadata. The query_function should take a string as input and return a list of (chunk, score) tuples. The chunk is a string and the score is a float representing the cosine distance between the chunk and the input string. The list should be sorted in ascending order by score. | None |
    | `sources` | *Optional[List[str]]* | The source text. In order to use this validator, you must provide either a `query_function` or `sources` with an `embed_function` in the metadata. | None |
    | `embed_function` | *Optional[Callable]* | A callable that creates embeddings for the sources. Must accept a list of strings and return an np.array of floats. | sentence-transformer's `paraphrase-MiniLM-L6-v2` |

</ul>
