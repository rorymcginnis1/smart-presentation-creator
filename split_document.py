"""
Simple CLI for splitting documents into sections.
Usage: python split_document.py <file> <num_sections>
"""
import sys
from src.document_splitter import split_document_into_sections


def main():
    if len(sys.argv) != 3:
        print("Usage: python split_document.py <file> <num_sections>")
        sys.exit(1)

    file_path = sys.argv[1]
    num_sections = int(sys.argv[2])

    with open(file_path) as f:
        document = f.read()

    print(f"Splitting {file_path} into {num_sections} sections...")
    sections = split_document_into_sections(document, num_sections)

    print("\n" + "="*60)
    for i, section in enumerate(sections, 1):
        print(f"\n--- SECTION {i}/{len(sections)} ---")
        print(section)

    print("\n" + "="*60)
    print(f"Created {len(sections)} sections")


if __name__ == "__main__":
    main()
