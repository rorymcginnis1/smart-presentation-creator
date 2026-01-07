import os
import re
import logging
from typing import List, Optional
import openai
from dotenv import load_dotenv
from .config import MAX_RETRIES
from . import config
from .prompts import get_all_semantic_boundaries, select_n_boundaries, single_pass_structured
from .fallbacks import extract_valid_splits_from_failed_output, fallback_split
from .adjustments import (
    combine_sections,
    combine_sections_llm,
    split_sections_iteratively
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_document_into_sections(
    document: str,
    target_slides: int,
    api_key: Optional[str] = None,
    model: str = "gpt-4o"
) -> List[str]:
    """
    Split a document into the target number of sections using LLM-based semantic splitting.
    Falls back to simpler methods if the LLM fails or modifies content.
    """
    if target_slides < 1 or target_slides > 50:
        raise ValueError('Need between 1-50 sections')
    if not document or not document.strip():
        raise ValueError('Document is empty')

    if target_slides == 1:
        return [document]

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError('Missing OpenAI API key')

    client = openai.OpenAI(api_key=api_key)

    # Get initial split from LLM
    secs = _get_initial_sections(document, target_slides, client, model)

    if len(secs) == target_slides:
        logger.info(f"Got exactly {target_slides} sections on first try")
        return secs
    elif len(secs) > target_slides:
        # Too many sections - use LLM to intelligently combine related ones
        logger.info(f'Got {len(secs)} sections, need to combine down to {target_slides}')
        temp = secs
        for _ in range(MAX_RETRIES):
            result = combine_sections_llm(temp, target_slides, client, model)

            if len(result) == target_slides:
                return result
            elif len(result) >= len(temp):
                break

            temp = result

        # If LLM combining didn't work, fall back to simple size-based combining
        if len(temp) > target_slides:
            return combine_sections(temp, target_slides)

        # Somehow ended up with too few, need to split more
        if len(temp) < target_slides:
            return split_sections_iteratively(temp, target_slides, client, model)

        return temp
    else:
        # Too few sections - split the largest ones until we hit the target
        logger.info(f"Got {len(secs)} sections, need to split up to {target_slides}")
        return split_sections_iteratively(secs, target_slides, client, model)


def _get_initial_sections(doc, target_slides, client, model):
    """
    Two-phase approach:
    Phase 1: Ask the LLM to identify ALL semantic boundaries (no counting constraint)
    Phase 2: Deterministically select exactly N-1 boundaries

    This decouples semantic understanding (LLM strength) from exact counting (LLM weakness).
    """
    if config.USE_STRUCTURED_OUTPUTS:
        for attempt in range(2):
            result = single_pass_structured(doc, target_slides, client, model)
            if result and len(result) == target_slides:
                logger.info(f"Structured output succeeded: {len(result)} sections (attempt {attempt+1})")
                return result
            if attempt == 0:
                logger.warning(f"Structured output failed on attempt {attempt+1}, retrying...")

        logger.error("Structured output failed after 2 attempts")
        raise ValueError(f"Structured output failed after 2 attempts for N={target_slides}")

    for attempt in range(MAX_RETRIES):
        # Phase 1: Get all semantic boundaries
        logger.info(f"Phase 1: Identifying all semantic boundaries (attempt {attempt + 1})")
        marked = get_all_semantic_boundaries(doc, client, model, attempt)

        if marked is None:
            if attempt == MAX_RETRIES - 1:
                logger.info('LLM not working, using fallback')
                return fallback_split(doc, target_slides, combine_sections)
            continue

        # Phase 2: Select exactly target_slides - 1 boundaries using LLM
        logger.info(f"Phase 2: Asking LLM to select exactly {target_slides - 1} boundaries")
        result = select_n_boundaries(marked, doc, target_slides, client, model)

        if result is None:
            # Phase 2 failed - but we have good boundaries from Phase 1, so use those
            # and let the caller's fallback logic adjust the count
            logger.warning("Phase 2 failed, using all boundaries from Phase 1")
            salvaged = extract_valid_splits_from_failed_output(marked, doc)
            if salvaged:
                logger.info(f'Using {len(salvaged)} sections from Phase 1, caller will adjust to {target_slides}')
                return salvaged

            # If we can't salvage anything, only retry Phase 1 if this isn't the last attempt
            if attempt == MAX_RETRIES - 1:
                return fallback_split(doc, target_slides, combine_sections)
            continue

        # Split on the selected boundaries
        secs = re.split(r'<<SPLIT>>', result)
        secs = [s for s in secs if s]  # Only filter out completely empty strings

        logger.info(f'Two-phase split successful - got exactly {len(secs)} sections (target: {target_slides})')
        return secs

    return fallback_split(doc, target_slides, combine_sections)
