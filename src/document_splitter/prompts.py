import re
import asyncio
import logging
import openai
from .config import (
    INITIAL_API_TIMEOUT_SECONDS,
    SPLIT_API_TIMEOUT_SECONDS,
    MIN_SECTION_SIZE_CHARS
)

logger = logging.getLogger(__name__)


def get_all_semantic_boundaries(doc, client, model, attempt=0):
    """
    Phase 1: Ask the LLM to identify ALL natural semantic boundaries.
    This removes the counting constraint and lets the LLM focus on understanding semantics.

    The trick here is we're NOT asking for an exact count - just "find every reasonable
    place to split." This is way easier for the LLM and much more reliable.
    """
    prompt = f"""Identify ALL natural split points in this document where one complete discrete idea ends and another begins. Insert the marker <<SPLIT>> at each split point.

Use the marker: <<SPLIT>>

Each section should contain one complete discrete idea. Don't split an idea across multiple sections - each idea must be kept whole and intact.

Only split between complete discrete ideas. Place markers at clear endpoints where ideas naturally separate:
- After paragraphs (ideal for major topic changes)
- After bullet point lists (once the list completes) - if bullets refer to the same idea, keep them together
- After individual sentences (when the sentence completes a discrete idea)

The key is: it must be both a clear endpoint (sentence/paragraph/list end) AND a separate idea. Splits can be at any granularity as long as both conditions are met.

Never place a marker in the middle of a sentence. Never place a marker in the middle of a bullet list - place it after the last bullet only if the following content is a different idea. If multiple bullets discuss the same idea, they must stay together in one section. Never split markdown formatting.

Find ALL reasonable split points - we'll select the best ones later. Think of this as: if someone asked "where could we possibly split this into slides?", mark every single reasonable option.

CRITICAL: Return the complete original document with markers inserted.
- Copy the text EXACTLY character-by-character
- Preserve ALL newlines, blank lines, and spacing EXACTLY as they appear
- Do NOT modify, paraphrase, or reformat any text
- Do NOT change line breaks or remove blank lines
- The ONLY addition is the <<SPLIT>> marker

Document:

{doc}"""

    sys_msg = (
        'You insert <<SPLIT>> markers into documents at ALL natural boundaries where discrete ideas end. '
        'Split at clear endpoints: after paragraphs, after bullet lists, or after sentences (if the sentence completes a discrete idea). '
        'Return the complete document with markers at EVERY reasonable split point. Never split mid-sentence or mid-bullet-list. '
        'CRITICAL: Copy the document EXACTLY character-for-character. '
        'Preserve ALL formatting: newlines, blank lines, spacing, indentation, bullet points. '
        'Do NOT change, add, or remove any whitespace, line breaks, or formatting. '
        'The ONLY thing you add is the <<SPLIT>> marker - nothing else changes.'
    )

    if attempt > 0:
        sys_msg += f" Retry {attempt + 1}: Return the exact original text with only <<SPLIT>> markers added."

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt}
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            timeout=INITIAL_API_TIMEOUT_SECONDS
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f'API call failed (attempt {attempt + 1}): {e}')
        return None


def select_n_boundaries(marked_doc, original_doc, target_slides, client, model):
    """
    Phase 2: Use LLM to intelligently select exactly N-1 boundaries from all candidates.

    The key insight: asking LLM to both find boundaries AND count to exactly N is hard.
    But asking it to pick N items from a list? Much easier and more reliable.
    """
    # First validate the LLM didn't modify content in Phase 1
    no_markers = marked_doc.replace('<<SPLIT>>', '')
    orig_normalized = ' '.join(original_doc.split())
    returned_normalized = ' '.join(no_markers.split())

    if orig_normalized != returned_normalized:
        logger.warning("Content was modified in Phase 1")
        return None

    # Split into sections based on ALL the boundaries we found
    # Don't strip - we need to preserve exact spacing
    sections = re.split(r'<<SPLIT>>', marked_doc)

    num_boundaries = len(sections) - 1  # N sections = N-1 boundaries

    if num_boundaries == 0:
        logger.warning("No boundaries found")
        return None

    logger.info(f"Phase 1 found {num_boundaries} total boundaries, need {target_slides - 1}")

    # If we got exactly what we need, we're done
    if num_boundaries == target_slides - 1:
        return marked_doc

    # If we got fewer boundaries than needed, just use what we have
    # (caller will handle further splitting if needed)
    if num_boundaries < target_slides - 1:
        return marked_doc

    # Now the LLM needs to pick which boundaries to keep
    # Create a preview of each section for the LLM to evaluate
    needed = target_slides - 1

    section_previews = []
    for i, sec in enumerate(sections):
        # Show first ~150 chars of each section (strip for preview only)
        preview = sec.strip()[:150].replace('\n', ' ')
        if len(sec.strip()) > 150:
            preview += "..."
        section_previews.append(f"Section {i}: {preview}")

    sections_text = "\n".join(section_previews)

    prompt = f"""You previously identified {num_boundaries} potential split points in a document.
Now you need to select exactly {needed} of these boundaries to create {target_slides} final sections.

Here are the {len(sections)} sections that would be created if we kept ALL boundaries:

{sections_text}

Your task: Pick exactly {needed} boundaries to keep. These should be the boundaries that:
1. Create the most semantically coherent sections
2. Separate the most important topic changes
3. Result in roughly balanced section sizes (but prioritize semantic coherence)

A boundary exists between each pair of adjacent sections. For example:
- Boundary 0 is between Section 0 and Section 1
- Boundary 1 is between Section 1 and Section 2
- etc.

Return ONLY a comma-separated list of exactly {needed} boundary numbers (0 to {num_boundaries - 1}).
For example: 0, 5, 12, 18, 25, 31, 38

Your response (exactly {needed} numbers):"""

    messages = [
        {
            "role": "system",
            "content": f"You select exactly {needed} boundaries from a list to create the best document sections. Return only comma-separated numbers."
        },
        {"role": "user", "content": prompt}
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            timeout=INITIAL_API_TIMEOUT_SECONDS
        )

        response_text = resp.choices[0].message.content.strip()

        # Parse the boundary numbers from the LLM response
        # Expected format: "0, 5, 12, 18, 25, 31, 38"
        selected = []
        for num_str in response_text.replace(' ', '').split(','):
            try:
                num = int(num_str)
                # Only keep valid boundary indices
                if 0 <= num < num_boundaries:
                    selected.append(num)
            except ValueError:
                # Skip any non-numeric junk the LLM might have added
                continue

        # Dedupe and sort, then take exactly as many as we need
        selected = sorted(set(selected))[:needed]

        # If the LLM didn't give us exactly the right count, something went wrong
        if len(selected) != needed:
            logger.warning(f"LLM returned {len(selected)} boundaries, expected {needed}")
            return None

        logger.info(f"Phase 2: LLM selected {len(selected)} boundaries: {selected}")

        # Rebuild document with only the selected boundaries
        # Important: preserve exact spacing by only adding <<SPLIT>> back where selected
        result_parts = []
        for i, section in enumerate(sections):
            result_parts.append(section)
            # Add boundary marker if this boundary was selected
            if i < len(sections) - 1 and i in selected:
                result_parts.append('<<SPLIT>>')  # No extra whitespace

        result = ''.join(result_parts)

        # Verify final count
        final_count = result.count('<<SPLIT>>')
        logger.info(f"Phase 2 complete: {final_count} boundaries in final output")

        return result

    except Exception as e:
        logger.warning(f"Phase 2 LLM call failed: {e}")
        return None


def select_sections_to_combine(secs, target_count, client, model):
    """
    Ask the LLM which adjacent sections should be combined to reach target count.
    Returns a list of indices representing the start of each pair to combine.
    """
    n_combines = len(secs) - target_count
    if n_combines <= 0:
        return []

    formatted = []
    for i, sec in enumerate(secs):
        preview = sec[:200] + "..." if len(sec) > 200 else sec
        formatted.append(f"Section {i}: {preview}")

    sections_text = "\n\n".join(formatted)

    prompt = f"""You have {len(secs)} sections and need to combine {n_combines} pairs of adjacent sections to get down to {target_count} sections.

Analyze which adjacent sections contain related ideas that should be combined together.

You can only combine sections that are next to each other (adjacent). Section i can only combine with section i+1. Choose pairs that would create the most coherent combined sections. Return exactly {n_combines} pairs.

Return your answer as a comma-separated list of pairs in this format:
0-1, 3-4, 7-8

Sections:

{sections_text}

Return only the pairs, nothing else:"""

    msgs = [
        {
            'role': 'system',
            'content': 'You identify which adjacent document sections should be combined based on semantic coherence.'
        },
        {'role': 'user', 'content': prompt}
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0,
            timeout=INITIAL_API_TIMEOUT_SECONDS
        )

        text = resp.choices[0].message.content.strip()

        pairs = []
        for p in text.split(','):
            p = p.strip()
            if '-' in p:
                try:
                    left, right = p.split('-')
                    l = int(left.strip())
                    r = int(right.strip())

                    if r == l + 1 and 0 <= l < len(secs) - 1:
                        pairs.append(l)
                except (ValueError, IndexError):
                    continue

        pairs = sorted(set(pairs), reverse=True)
        return pairs[:n_combines]

    except Exception as e:
        logger.warning(f'Combining failed: {e}')
        return None


async def split_section_async(sec, client, model):
    """
    Ask the LLM to split one section into two parts.
    Returns a list of 2 parts if successful, None otherwise.
    """
    if len(sec.strip()) < MIN_SECTION_SIZE_CHARS:
        return None

    prompt = f"""Split this section into exactly 2 parts by inserting one <<SPLIT>> marker.

Find the natural break point between two ideas and insert <<SPLIT>> there.

Never split in the middle of a sentence. Never split in the middle of a bullet list - if bullets refer to the same idea, keep them together. Only split after a bullet list if the following content is a different idea. If you cannot find a good split point, return the section unchanged (no marker). Return the complete section with one <<SPLIT>> marker or unchanged.

Section:

{sec}"""

    msgs = [
        {
            'role': 'system',
            'content': 'You split sections at natural boundaries. Insert one <<SPLIT>> marker or return unchanged. Critical: Copy the text EXACTLY character-for-character - do not add spaces, remove spaces, or change any text. The only thing you add is the <<SPLIT>> marker.'
        },
        {'role': 'user', 'content': prompt}
    ]

    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=msgs,
            temperature=0,
            timeout=SPLIT_API_TIMEOUT_SECONDS
        )

        txt = resp.choices[0].message.content.strip()

        no_markers = txt.replace('<<SPLIT>>', '')
        orig = ' '.join(sec.split())
        returned = ' '.join(no_markers.split())

        if orig != returned:
            return None

        if '<<SPLIT>>' in txt:
            parts = re.split(r'<<SPLIT>>', txt)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) == 2:
                return parts

        return None

    except Exception as e:
        logger.warning(f'Split failed: {e}')
        return None


async def split_batch_parallel(sections_to_split, client, model):
    """
    Split multiple sections in parallel using async.
    Returns a list of results (each is either a 2-part list or None).
    """
    tasks = [
        split_section_async(sec, client, model)
        for _, sec in sections_to_split
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Convert exceptions to None
    results = [
        result if not isinstance(result, Exception) else None
        for result in results
    ]
    return results
