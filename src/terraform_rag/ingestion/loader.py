from dataclasses import dataclass, field
from pathlib import Path
import frontmatter

DOCS_DIR = Path(__file__).parents[3] / "data" / "docs"
SECTION_DIRS = {"cli", "internals", "intro", "language"}


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


def _derive_section(path: Path, docs_root: Path) -> str:
    relative = path.relative_to(docs_root)
    for part in relative.parts:
        if part in SECTION_DIRS:
            return part
    return "other"


def load_document(path: Path, docs_root: Path) -> Document:
    post = frontmatter.load(path)

    metadata = {
        "page_title":    post.metadata.get("page_title", ""),
        "description":   post.metadata.get("description", ""),
        "last_modified": str(post.metadata.get("last_modified", "")),
        "section":       _derive_section(path, docs_root),
        "source_path":   str(path.relative_to(docs_root)),
        "chunk_index":   0,
    }

    return Document(content=post.content, metadata=metadata)


def load_all(docs_root: Path = DOCS_DIR) -> list[Document]:
    paths = sorted(docs_root.rglob("*.mdx"))
    return [load_document(p, docs_root) for p in paths]
