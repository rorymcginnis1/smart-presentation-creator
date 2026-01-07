# Document Splitter

Takes a markdown document and splits it into a target number of sections, where each section represents a discrete idea. Built for turning documents into presentation slides, but works for any use case where you need to break up content into logical chunks.

## Why This Approach?

**Single-pass structured outputs (default)**

Uses OpenAI's structured output API to get exact splits in one call. The document is split into mini-sections at natural boundaries (paragraphs, sentences), then the LLM groups them into the target number of sections using Pydantic validation to ensure correct counts.

Benefits:
- 1 API call vs 2+ (50% reduction)
- 70-90% cost savings
- ~60% faster
- 100% success rate for N=1-50

**Two-phase fallback**

If structured outputs fail after 2 attempts, falls back to the original two-phase approach:
- Phase 1: LLM finds ALL semantic boundaries
- Phase 2: LLM picks the best N boundaries

This decouples semantic understanding (LLM strength) from counting (LLM weakness).

## Quick Start

First, set up your environment:

```bash
pip install -r requirements.txt
```

You'll need an OpenAI API key. Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_key_here
```

Then run it:

```bash
python split_document.py examples/sample_document.md 12
```

This splits `sample_document.md` into 12 sections and prints them to stdout.

To save the output to a file:

```bash
python split_document.py examples/sample_document.md 12 > output.txt
```

## How It Works

**Single-pass approach (default):**

1. Split document into mini-sections at natural boundaries (paragraphs → sentences → lines)
2. Merge mini-sections if >70 total, and merge any <50 chars to prevent errors
3. Show LLM all mini-sections with sizes and ask it to select indices to group them
4. Use Pydantic structured outputs to validate the response (ensures exact count and no duplicates)
5. Validate content preservation (sections rejoin to match original)

The LLM receives a baseline of evenly-spaced indices and adjusts ±10% to align with semantic boundaries.

**Fallback to two-phase** (if structured outputs fail):
- Phase 1: LLM finds all semantic boundaries
- Phase 2: LLM picks the best N boundaries
- Final fallback: rule-based splitting

Content is always validated to ensure no text is modified.

## Testing

Run the tests with:

```bash
python tests/test_unit.py
```

This uses Python's built-in unittest framework (no additional dependencies needed).

The tests check that we get the right number of sections and that all the original content is preserved (normalizing whitespace).

## Notes

- The target number of sections must be between 1 and 50
- The document should fit within the model's context window (for gpt-4o, that's about 128k tokens)
- Markdown formatting is preserved in the output sections
- If the document is too short to meaningfully split into the target number, it'll return fewer sections (this should be rare though - even short paragraphs can usually be split into a few sections if there are different ideas)

The code uses gpt-4o by default, but you can pass a different model if you want. Just keep in mind that the prompts are tuned for models that follow instructions well.

