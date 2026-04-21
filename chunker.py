"""
chunker.py — Smart code chunker for RAG pipeline
Handles: nested classes, nested functions, decorators, 2000-line files,
         .py .js .jsx .ts .tsx .go .html .css .scss .java .rs .rb .cpp .c
         .ipynb .md .rst .yaml .yml .json .toml .txt .log

Install:
    pip install tree-sitter tree-sitter-languages nbformat
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_OK = True
except ImportError:
    TREE_SITTER_OK = False
    print("[chunker] tree-sitter-languages not installed → sliding window fallback")

try:
    import nbformat
    NBFORMAT_OK = True
except ImportError:
    NBFORMAT_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Language maps
# ─────────────────────────────────────────────────────────────────────────────

EXT_TO_LANG = {
    # core
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".go":   "go",
    ".html": "html",
    ".css":  "css",
    ".scss": "css",
    # extended
    ".java": "java",
    ".rs":   "rust",
    ".rb":   "ruby",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".c":    "c",
    ".cs":   "c_sharp",
    ".php":  "php",
    ".kt":   "kotlin",
    ".swift":"swift",
    ".lua":  "lua",
    ".r":    "r",
    ".sh":   "bash",
    ".bash": "bash",
}

# ── Node types PER language that represent a meaningful semantic unit ─────────
# "top-level" AND "nested" types are listed — the recursion strategy decides
# which level to split at based on size.
CHUNK_NODE_TYPES: Dict[str, set] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",   # handles @decorator\ndef foo / @decorator\nclass Foo
        "async_function_definition",
    },
    "javascript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "export_statement",
        "generator_function_declaration",
        "async_function_declaration",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "export_statement",
        "async_function_declaration",
    },
    "tsx": {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
        "jsx_element",
        "jsx_self_closing_element",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "type_spec",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "constructor_declaration",
        "annotation_type_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "type_item",
        "macro_definition",
    },
    "ruby": {
        "method",
        "singleton_method",
        "class",
        "module",
        "do_block",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
        "template_declaration",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "declaration",
    },
    "c_sharp": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
        "property_declaration",
        "namespace_declaration",
        "enum_declaration",
    },
    "php": {
        "function_definition",
        "class_declaration",
        "method_declaration",
        "interface_declaration",
        "trait_declaration",
    },
    "kotlin": {
        "function_declaration",
        "class_declaration",
        "object_declaration",
        "companion_object",
        "secondary_constructor",
    },
    "swift": {
        "function_declaration",
        "class_declaration",
        "struct_declaration",
        "protocol_declaration",
        "extension_declaration",
        "computed_property",
    },
    "lua": {
        "function_declaration",
        "local_function",
        "function_definition",
    },
    "r": {
        "function_definition",
    },
    "bash": {
        "function_definition",
    },
    "html": {
        "script_element",
        "style_element",
        "element",
    },
    "css": {
        "rule_set",
        "media_statement",
        "keyframes_statement",
        "supports_statement",
    },
}

MAX_CHUNK_TOKENS  = 400    # ~1 600 chars — safe for all embedding models
SLIDE_OVERLAP     = 50     # lines overlap in sliding window
NESTED_SIZE_LIMIT = MAX_CHUNK_TOKENS * 2   # if a node > this, recurse into children


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _make_chunk(
    text: str,
    file_path: str,
    chunk_index: int,
    doc_type: str,
    language: str = "",
    name: str = "",
    start_line: int = 0,
    parent_name: str = "",          # NEW: e.g. "OuterClass" for nested methods
) -> Dict[str, Any]:
    full_name = f"{parent_name}.{name}" if parent_name and name else (name or f"chunk_{chunk_index}")
    return {
        "chunk_index": chunk_index,
        "file_path":   file_path,
        "doc_type":    doc_type,
        "language":    language,
        "name":        full_name,
        "start_line":  start_line,
        "parent":      parent_name,
        "text":        text.strip(),
    }


def _extract_name(node, src_bytes: bytes) -> str:
    """Best-effort name extraction from an AST node."""
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier",
                          "property_identifier", "field_identifier"):
            return src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sliding-window chunker (fallback / text / config)
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_chunks(
    text: str,
    file_path: str,
    doc_type: str = "text",
    language: str = "",
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = SLIDE_OVERLAP,
    idx_start: int = 0,
) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    step  = max(1, max_tokens - overlap)
    chunks, idx = [], idx_start
    i = 0
    while i < len(lines):
        window  = lines[i : i + max_tokens]
        content = "\n".join(window)
        if content.strip():
            chunks.append(_make_chunk(content, file_path, idx, doc_type, language, start_line=i))
            idx += 1
        i += step
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Tree-sitter recursive chunker  ← KEY: handles nesting properly
# ─────────────────────────────────────────────────────────────────────────────

def _walk_node(
    node,
    src_bytes: bytes,
    file_path: str,
    doc_type: str,
    language: str,
    target_types: set,
    idx: int,
    parent_name: str = "",
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """
    Recursively walk AST.

    Strategy:
      • If a node IS a target type:
          – Small enough  → emit as one chunk (with parent context prepended)
          – Too large     → recurse INTO its children to find sub-chunks,
                            then emit any "leftover" (top of function before
                            nested defs) as an extra chunk.
      • If a node is NOT a target type → recurse into its children.
    """
    chunks = []

    for child in node.children:

        if child.type not in target_types:
            # Not a target — recurse deeper
            sub = _walk_node(child, src_bytes, file_path, doc_type, language,
                             target_types, idx, parent_name, depth + 1)
            chunks.extend(sub)
            idx += len(sub)
            continue

        # ── This child IS a target node ───────────────────────────────────────
        node_text = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        node_name = _extract_name(child, src_bytes)
        full_name = f"{parent_name}.{node_name}" if parent_name and node_name else node_name

        if _approx_tokens(node_text) <= NESTED_SIZE_LIMIT:
            # ── Small enough: emit as single chunk ────────────────────────────
            # If we're nested, prepend a breadcrumb so the chunk is self-contained
            if parent_name:
                header    = f"# context: inside {parent_name}\n"
                emit_text = header + node_text
            else:
                emit_text = node_text

            chunks.append(_make_chunk(
                emit_text, file_path, idx, doc_type, language,
                name=node_name, start_line=child.start_point[0],
                parent_name=parent_name,
            ))
            idx += 1

        else:
            # ── Too large: recurse into children to find nested defs ──────────
            # 1. Collect nested sub-chunks
            sub = _walk_node(child, src_bytes, file_path, doc_type, language,
                             target_types, idx, full_name, depth + 1)

            # 2. Build a "header" chunk = everything BEFORE the first nested def
            #    so callers still see the class/function signature + docstring
            if sub:
                first_nested_start = sub[0]["start_line"]
                header_lines = node_text.splitlines()
                # lines before the first nested child (relative to node start)
                rel_start = child.start_point[0]
                cutoff    = first_nested_start - rel_start
                header_text = "\n".join(header_lines[:max(cutoff, 8)])  # at least 8 lines
                if header_text.strip():
                    chunks.append(_make_chunk(
                        header_text, file_path, idx, doc_type, language,
                        name=f"{node_name}[header]",
                        start_line=child.start_point[0],
                        parent_name=parent_name,
                    ))
                    idx += 1

            chunks.extend(sub)
            idx += len(sub)

    return chunks


def tree_sitter_chunks(
    text: str,
    file_path: str,
    lang_name: str,
    doc_type: str,
) -> List[Dict[str, Any]]:
    if not TREE_SITTER_OK:
        return sliding_window_chunks(text, file_path, doc_type, lang_name, max_tokens=50, overlap=10)

    try:
        get_language(lang_name)
        parser = get_parser(lang_name)
    except Exception as e:
        print(f"[chunker DEBUG] language={lang_name} parser_failed={e}")
        return sliding_window_chunks(text, file_path, doc_type, lang_name, max_tokens=50, overlap=10)

    src_bytes    = text.encode("utf-8")
    tree         = parser.parse(src_bytes)
    target_types = CHUNK_NODE_TYPES.get(lang_name, set())

    chunks = _walk_node(tree.root_node, src_bytes, file_path, doc_type,
                        lang_name, target_types, idx=0)

    # Fallback: file has no recognised nodes (e.g. pure script, no functions)
    if not chunks:
        return sliding_window_chunks(text, file_path, doc_type, lang_name, max_tokens=50, overlap=10)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Notebook chunker
# ─────────────────────────────────────────────────────────────────────────────

def notebook_chunks(path: str) -> List[Dict[str, Any]]:
    if not NBFORMAT_OK:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        # split by chars since notebook is 1 long JSON line
        chunks = []
        chunk_size = 8000  # chars
        for i in range(0, len(text), chunk_size):
            piece = text[i:i+chunk_size]
            if piece.strip():
                chunks.append(_make_chunk(piece, path, i//chunk_size, "notebook"))
        return chunks

    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    chunks, md_buf, idx = [], [], 0

    for cell in nb.cells:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if not src.strip():
            continue

        if cell["cell_type"] == "markdown":
            md_buf.append(src)

        elif cell["cell_type"] == "code":
            if md_buf:
                md_text = "\n\n".join(md_buf)
                chunks.append(_make_chunk(md_text, path, idx, "notebook_markdown",
                                          name=f"md_{idx}"))
                md_buf = []
                idx   += 1
            chunks.append(_make_chunk(src, path, idx, "notebook",
                                      language="python", name=f"cell_{idx}"))
            idx += 1

    # flush trailing markdown
    if md_buf:
        chunks.append(_make_chunk("\n\n".join(md_buf), path, idx,
                                  "notebook_markdown", name=f"md_{idx}"))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Config chunker  (yaml / json / toml)
# ─────────────────────────────────────────────────────────────────────────────

def config_chunks(text: str, file_path: str) -> List[Dict[str, Any]]:
    if _approx_tokens(text) <= 300:
        return [_make_chunk(text, file_path, 0, "config")]

    lines  = text.splitlines()
    chunks, buf, idx, start = [], [], 0, 0

    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith(("#", "//")):
            if buf:
                chunks.append(_make_chunk("\n".join(buf), file_path, idx,
                                          "config", start_line=start))
                idx += 1
            buf, start = [line], i
        else:
            buf.append(line)

    if buf:
        chunks.append(_make_chunk("\n".join(buf), file_path, idx,
                                  "config", start_line=start))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Markdown / RST chunker
# ─────────────────────────────────────────────────────────────────────────────

def markdown_chunks(text: str, file_path: str) -> List[Dict[str, Any]]:
    # split on any heading level 1-3
    parts = re.split(r"(?m)^(#{1,3} )", text)
    chunks, buf, idx = [], [], 0

    for part in parts:
        if re.match(r"^#{1,3} $", part):
            buf = [part]          # start new section with heading marker
        else:
            buf.append(part)
            combined = "".join(buf).strip()
            if combined:
                chunks.append(_make_chunk(combined, file_path, idx, "markdown"))
                idx += 1
            buf = []

    if buf:
        combined = "".join(buf).strip()
        if combined:
            chunks.append(_make_chunk(combined, file_path, idx, "markdown"))

    return chunks or [_make_chunk(text, file_path, 0, "markdown")]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             "dist", "build", ".mypy_cache", ".pytest_cache"}
SKIP_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
             ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff",
             ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".wav",
             ".zip", ".tar", ".gz", ".lock", ".sum"}


def chunk_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Dispatch a single file to the right chunker.
    Returns list of chunk dicts: {chunk_index, file_path, doc_type,
                                   language, name, parent, start_line, text}
    """
    path = Path(file_path)
    ext  = path.suffix.lower()
    stem = path.name.lower()

    # ── Notebooks ─────────────────────────────────────────────────────────────
    if ext == ".ipynb":
        return notebook_chunks(file_path)

    # ── Read text ─────────────────────────────────────────────────────────────
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[chunker] Cannot read {file_path}: {e}")
        return []

    if not text.strip():
        return []

    # ── Config ────────────────────────────────────────────────────────────────
    if ext in {".yaml", ".yml", ".json", ".toml"}:
        return config_chunks(text, file_path)

    # ── Markdown / RST ────────────────────────────────────────────────────────
    if ext in {".md", ".rst", ".mdx"}:
        return markdown_chunks(text, file_path)

    # ── Plain text ────────────────────────────────────────────────────────────
    if ext in {".txt", ".log", ".csv", ""}:
        return sliding_window_chunks(text, file_path, "text")

    # ── Known code / markup / style ───────────────────────────────────────────
    if ext in EXT_TO_LANG:
        lang_name = EXT_TO_LANG[ext]

        if ext in {".css", ".scss"}:
            doc_type = "style"
        elif ext == ".html":
            doc_type = "markup"
        elif stem.startswith("test_") or ".spec." in stem or "_test." in stem:
            doc_type = "test"
        else:
            doc_type = "code"

        return tree_sitter_chunks(text, file_path, lang_name, doc_type)

    # ── Unknown extension: sliding window ─────────────────────────────────────
    return sliding_window_chunks(text, file_path, "unknown")


def chunk_repo(
    repo_root: str,
    changed_files: List[str] = None,
) -> List[Dict[str, Any]]:
    """Walk a repo and chunk every relevant file."""
    all_chunks = []
    root = Path(repo_root)

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_EXTS:
            continue

        rel = path.relative_to(root).as_posix()

        if changed_files is not None:
            if rel not in changed_files and path.suffix.lower() != ".ipynb":
                continue

        chunks = chunk_file(str(path))
        for c in chunks:
            c["file_path"] = rel
        all_chunks.extend(chunks)

    print(f"[chunker] Total chunks: {len(all_chunks)}")
    return all_chunks
