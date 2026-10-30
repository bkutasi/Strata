# Model tokenizer

`Load` accepts a pack root or its `tokenizer` directory. `SpecialIDs` returns
the pack's original names and IDs, without translating between Qwen's GGUF
names and DeepSeek's short names. The returned map is a copy.

Call `EncodeWithBudget` for input from untrusted API callers. Its limit counts
UTF-8 input bytes, including special tokens. It is not a memory limit and does
not bound allocations made by the pre-tokenizer. `Encode` has no input limit.

Missing or malformed tokenizer metadata and Python-falsy `pre` values (null,
false, zero, empty string, array, or object) use `qwen35`, like the Python
server. Unlike the Python tokenizer's fallback for unknown values, this loader
rejects unknown nonempty families and truthy non-string values.
Special IDs must be integers within the vocabulary.
