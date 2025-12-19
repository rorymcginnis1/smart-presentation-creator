# Document Splitter

Takes a markdown document and splits it into a target number of sections, where each section represents a discrete idea. Built for turning documents into presentation slides, but works for any use case where you need to break up content into logical chunks.

## Why This Approach?

After trying a bunch of different methods, I settled on this two-phase approach with fallbacks. Here's the key insight: **LLMs are amazing at understanding where ideas start and stop, but they suck at counting.**

Initial attempts that didn't work well:
- **Single-shot prompting**: "Split this into exactly N sections" → Got wildly wrong counts (asked for 12, got 33)
- **Pure rule-based**: Split on paragraphs/sentences → Ignores semantic boundaries, splits ideas in half
- **Just iterative adjustment**: Get rough split, then combine/split until exact → Works but expensive (2-3x API calls)

**The solution: Two-phase approach with fallbacks**

**Phase 1 - Let the LLM do what it's good at:**
Ask it to find ALL semantic boundaries without worrying about hitting an exact count. The prompt is simple: "Mark every point where one idea ends and another begins." This removes the counting pressure and lets the LLM focus purely on understanding semantics. Typically finds 50-60 boundaries.

**Phase 2 - LLM intelligently picks the best N boundaries:**
Now we give the LLM a simpler task: "Here are 57 boundaries, pick the best 11 for creating 12 sections." This is way more reliable than asking it to find and count simultaneously. The LLM evaluates which boundaries separate the most important topic changes and returns exactly the right number.

**Why this works better:**
- Decouples semantic understanding from counting (plays to LLM strengths)
- Phase 2 is just selection from a list (much easier than generation + counting)
- Usually gets exact count on first try
- Better quality splits (LLM picks most important boundaries, not random ones)
- Only 2 API calls in the happy path

**Fallback layers** (if two-phase doesn't nail it):
- If we got fewer boundaries than needed in Phase 1, we use iterative splitting (parallel API calls to split large sections)
- Last resort: rule-based splitting on paragraph boundaries
- All paths preserve content through validation

This approach hits the exact count. The iterative adjustment logic is still there as a safety net for edge cases.

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

This splits `sample_document.md` into 12 sections. The sections get printed to stdout.

## How It Works

The flow is actually pretty simple once you understand the two phases:

1. **Phase 1 - Find ALL boundaries**: Ask the LLM to identify every semantic boundary in the document (where one complete idea ends and another begins). Insert `<<SPLIT>>` markers at each boundary. No pressure to hit an exact count - just find all the natural split points. Usually finds 50-60 boundaries in a typical document.

2. **Phase 2 - Pick the best N**: Give the LLM all the boundaries from Phase 1 (as previews of each section) and ask it to select exactly N-1 of them. The LLM evaluates which boundaries create the best semantic separations and returns just the boundary indices to keep. We rebuild the document with only those selected markers.

3. **Content validation**: After each phase, we check that the LLM didn't accidentally modify the text (sometimes they add/remove spaces). We normalize whitespace when comparing - we care about content, not exact spacing.

4. **Fallbacks** (if needed):
   - If Phase 2 gives us fewer boundaries than we need, iterative splitting kicks in (parallel LLM calls to split the largest sections)
   - If the LLM modifies content, we try to salvage the split points by matching context
   - Last resort: rule-based splitting on paragraphs/sentences

The two-phase approach usually nails the exact count on the first try. The fallback logic is insurance for edge cases (tiny documents, API issues, etc.).

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

