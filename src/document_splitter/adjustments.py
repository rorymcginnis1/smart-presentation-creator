import asyncio
from .config import MAX_PARALLEL_SPLITS, MAX_SPLIT_ROUNDS
from .prompts import select_sections_to_combine, split_batch_parallel
from .fallbacks import fallback_split


def combine_sections(secs, target_count):
    """
    Simple fallback combining - repeatedly merges the two smallest adjacent sections.
    Used when LLM-based combining fails.
    """
    curr = list(secs)
    while len(curr) > target_count:
        # Find the pair of adjacent sections with the smallest combined size
        best_idx = 0
        best_size = float('inf')
        for i in range(len(curr) - 1):
            size = len(curr[i]) + len(curr[i + 1])
            if size < best_size:
                best_size = size
                best_idx = i

        # Merge them
        combined = curr[best_idx] + "\n\n" + curr[best_idx + 1]
        curr = curr[:best_idx] + [combined] + curr[best_idx + 2:]

    return curr


def combine_sections_llm(secs, target_count, client, model):
    """
    Use the LLM to select which adjacent sections should be combined.
    Falls back to simple combining if the LLM fails.
    """
    curr = list(secs)

    if len(curr) <= target_count:
        return curr

    # Ask LLM which pairs of adjacent sections are most related
    pairs = select_sections_to_combine(curr, target_count, client, model)
    if pairs is None:
        return combine_sections(secs, target_count)

    # Merge the selected pairs
    for idx in pairs:
        if idx < len(curr) - 1:
            combined = curr[idx] + "\n\n" + curr[idx + 1]
            curr = curr[:idx] + [combined] + curr[idx + 2:]

    return curr


def split_sections_iteratively(sections, target_count, client, model, max_parallel=MAX_PARALLEL_SPLITS):
    """
    Iteratively split the largest sections until we reach the target count.
    Processes multiple sections in parallel for speed.
    """
    curr = list(sections)
    cant_split = set()  # Track sections that couldn't be split
    retried = False

    for _ in range(MAX_SPLIT_ROUNDS):
        if len(curr) >= target_count:
            return curr[:target_count]

        needed = target_count - len(curr)
        # Find sections we haven't tried to split yet
        candidates = [
            (i, section) for i, section in enumerate(curr)
            if i not in cant_split
        ]

        if not candidates:
            break

        # Split the largest sections first (they're most likely to have clear split points)
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        batch = min(needed, max_parallel, len(candidates))
        to_split = candidates[:batch]

        # Split multiple sections in parallel
        results = asyncio.run(split_batch_parallel(to_split, client, model))

        # Process results in reverse order to handle index shifts
        idx_map = {idx: result for (idx, _), result in zip(to_split, results)}
        reverse = sorted(to_split, key=lambda x: x[0], reverse=True)
        count = 0

        for idx, _ in reverse:
            r = idx_map[idx]

            if r and len(r) == 2:
                # Successfully split into 2 parts
                count += 1
                curr = curr[:idx] + r + curr[idx + 1:]
                # Adjust cant_split indices for the new section
                cant_split = {i + (1 if i > idx else 0) for i in cant_split}
            else:
                # Couldn't split this section
                cant_split.add(idx)

        # If nothing split this round, try one more time with all sections
        if count == 0:
            if not retried:
                retried = True
                cant_split.clear()
                continue
            break

    # Still short of target - use fallback
    if len(curr) < target_count:
        return fallback_split('\n\n'.join(curr), target_count, combine_sections)

    return curr
