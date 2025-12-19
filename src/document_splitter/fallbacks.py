import re
from .config import CONTEXT_WINDOW_CHARS, CONTEXT_WORDS_BEFORE, CONTEXT_WORDS_AFTER


def extract_valid_splits_from_failed_output(marked, orig):
    """
    Recovery mechanism when the LLM modifies content.
    Finds where the split markers are in the modified text, then locates those same
    positions in the original document using context matching.
    """
    positions = [m.start() for m in re.finditer(r'<<SPLIT>>', marked)]
    if not positions:
        return None

    pts = []
    marker_len = len('<<SPLIT>>')

    for pos in positions:
        # Get context around each marker
        before = marked[max(0, pos - CONTEXT_WINDOW_CHARS):pos]
        after = marked[pos + marker_len:pos + marker_len + CONTEXT_WINDOW_CHARS]

        w_before = before.split()[-CONTEXT_WORDS_BEFORE:] if before.split() else []
        w_after = after.split()[:CONTEXT_WORDS_AFTER] if after.split() else []

        if not w_before or not w_after:
            continue

        # Try to find these context words in the original document
        pattern1 = ' '.join(w_before)
        pattern2 = ' '.join(w_after)

        p1 = orig.find(pattern1)
        p2 = orig.find(pattern2, p1 if p1 != -1 else 0)

        # If we found both patterns, the split point is between them
        if p1 != -1 and p2 != -1 and p2 > p1:
            pt = p1 + len(pattern1)
            pts.append(pt)

    if not pts:
        return None

    pts = sorted(set(pts))

    # Split the original document at the recovered positions
    sections = []
    last = 0

    for pt in pts:
        section = orig[last:pt]
        if section:  # Only skip completely empty sections
            sections.append(section)
        last = pt

    final = orig[last:]
    if final:
        sections.append(final)

    if len(sections) < 2:
        return None

    return sections


def fallback_split(text, num_sections, combine_fn):
    """
    Non-LLM fallback that splits on paragraph boundaries.
    If that gives wrong count, combines or splits sections mechanically.
    """
    # Split on paragraph breaks
    parts = re.split(r'(\n\n+)', text)

    secs = []
    curr = ''

    for p in parts:
        if p.strip():
            if curr:
                secs.append(curr)
            curr = p
        else:
            curr += p

    if curr:
        secs.append(curr)

    secs = [s.strip() for s in secs if s.strip()]

    if not secs:
        return [text]

    # Adjust to target count
    if len(secs) == num_sections:
        return secs
    elif len(secs) > num_sections:
        return combine_fn(secs, num_sections)
    else:
        # Need more sections - split the longest ones
        while len(secs) < num_sections:
            longest_idx = max(range(len(secs)), key=lambda i: len(secs[i]))
            longest = secs[longest_idx]

            # Try splitting at sentence boundary first
            sent = r'([.!?]+)\s+'
            matches = list(re.finditer(sent, longest))

            if len(matches) > 0:
                mid = len(matches) // 2
                pos = matches[mid].end()
                p1 = longest[:pos].strip()
                p2 = longest[pos:].strip()
            else:
                # No sentences, try line breaks
                lines = longest.split('\n')
                if len(lines) > 1:
                    mid = len(lines) // 2
                    p1 = '\n'.join(lines[:mid]).strip()
                    p2 = '\n'.join(lines[mid:]).strip()
                else:
                    # Last resort: split on nearest space to midpoint
                    mid = len(longest) // 2
                    sp = longest.rfind(' ', 0, mid)
                    if sp == -1:
                        sp = longest.find(' ', mid)
                    if sp == -1:
                        sp = mid

                    p1 = longest[:sp].strip()
                    p2 = longest[sp:].strip()

            if p1 and p2:
                secs = secs[:longest_idx] + [p1, p2] + secs[longest_idx + 1:]
            else:
                break

        return secs[:num_sections]
