"""
chunker.py — Smart code chunker for RAG pipeline
Handles: .py .js .jsx .ts .tsx .go .html .css .scss .java .rs .rb .cpp .c
         .ipynb .md .rst .yaml .yml .json .toml .txt .log

Install:
    pip install tree-sitter==0.21.3 tree-sitter-languages nbformat

Every chunk dict has exactly these fields:

    id            str   — "file_path::symbol_name::start_line"  (unique, readable)
    file_path     str   — relative path to source file  (always set)
    doc_type      str   — "code" | "code_context" | "style" | "markup" | "test" |
                          "test_context" | "markdown" | "config" | "notebook" | …
    name          str   — fully-qualified symbol e.g. "Dog.fetch", "[file_context]"
    parent        str   — enclosing class/scope name, "" for top-level
    start_line    int   — 0-indexed line where this chunk begins
    language      str   — e.g. "python", "javascript"
    calls         list  — chunk ids this chunk invokes (resolved GLOBALLY across all files)
    called_by     list  — chunk ids that call this chunk (back-filled)
    text          str   — the actual code/content  ← this gets embedded

Context chunks (doc_type ends in "_context") also carry:
    symbol_names  list  — every function/class name defined in this file/class scope

Two kinds of chunks per code file
──────────────────────────────────
1. CONTEXT chunk — one per file, one per class:
   imports + globals + class/function signatures (no full bodies).
   Grows until MAX_CHUNK_TOKENS then slides.
   Carries symbol_names so one retrieval hit tells you everything defined here.

2. FUNCTION/METHOD chunks — one per symbol.
   Classes are never emitted as full chunks when methods are chunked (no duplication).
   Each method carries "# context: inside ClassName" breadcrumb for self-contained retrieval.

calls / called_by are resolved GLOBALLY across ALL files in chunk_repo.
For incremental updates, pass existing_name_map from Qdrant so cross-file
deps to unchanged files still resolve correctly.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_OK = True
except ImportError:
    TREE_SITTER_OK = False
    print("[chunker] tree-sitter-languages not installed → sliding-window fallback")

try:
    import nbformat
    NBFORMAT_OK = True
except ImportError:
    NBFORMAT_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Language maps
# ─────────────────────────────────────────────────────────────────────────────

EXT_TO_LANG = {
    ".py":   "python",
    ".js":   "javascript",  ".jsx": "javascript",
    ".ts":   "typescript",  ".tsx": "tsx",
    ".go":   "go",
    ".html": "html",
    ".css":  "css",         ".scss": "css",
    ".java": "java",
    ".rs":   "rust",
    ".rb":   "ruby",
    ".cpp":  "cpp",  ".cc": "cpp",  ".cxx": "cpp",
    ".c":    "c",
    ".cs":   "c_sharp",
    ".php":  "php",
    ".kt":   "kotlin",
    ".swift":"swift",
    ".lua":  "lua",
    ".r":    "r",
    ".sh":   "bash",  ".bash": "bash",
}

CHUNK_NODE_TYPES: Dict[str, set] = {
    "python":     {"function_definition", "async_function_definition", "decorated_definition", "class_definition"},
    "javascript": {"function_declaration", "function_expression", "arrow_function", "class_declaration",
                   "class_expression", "method_definition", "export_statement",
                   "generator_function_declaration", "async_function_declaration"},
    "typescript": {"function_declaration", "function_expression", "arrow_function", "class_declaration",
                   "method_definition", "interface_declaration", "type_alias_declaration",
                   "enum_declaration", "export_statement", "async_function_declaration"},
    "tsx":        {"function_declaration", "function_expression", "arrow_function", "class_declaration",
                   "method_definition", "interface_declaration", "type_alias_declaration",
                   "export_statement", "jsx_element", "jsx_self_closing_element"},
    "go":         {"function_declaration", "method_declaration", "type_declaration", "type_spec"},
    "java":       {"method_declaration", "class_declaration", "interface_declaration",
                   "enum_declaration", "constructor_declaration", "annotation_type_declaration"},
    "rust":       {"function_item", "impl_item", "struct_item", "enum_item",
                   "trait_item", "mod_item", "type_item", "macro_definition"},
    "ruby":       {"method", "singleton_method", "class", "module", "do_block"},
    "cpp":        {"function_definition", "class_specifier", "struct_specifier",
                   "namespace_definition", "template_declaration"},
    "c":          {"function_definition", "struct_specifier", "declaration"},
    "c_sharp":    {"method_declaration", "class_declaration", "interface_declaration",
                   "constructor_declaration", "property_declaration",
                   "namespace_declaration", "enum_declaration"},
    "php":        {"function_definition", "class_declaration", "method_declaration",
                   "interface_declaration", "trait_declaration"},
    "kotlin":     {"function_declaration", "class_declaration", "object_declaration",
                   "companion_object", "secondary_constructor"},
    "swift":      {"function_declaration", "class_declaration", "struct_declaration",
                   "protocol_declaration", "extension_declaration", "computed_property"},
    "lua":        {"function_declaration", "local_function", "function_definition"},
    "r":          {"function_definition"},
    "bash":       {"function_definition"},
    "html":       {"script_element", "style_element", "element"},
    "css":        {"rule_set", "media_statement", "keyframes_statement", "supports_statement"},
}

CLASS_NODE_TYPES: Dict[str, set] = {
    "python":     {"class_definition"},
    "javascript": {"class_declaration", "class_expression"},
    "typescript": {"class_declaration"},
    "tsx":        {"class_declaration"},
    "java":       {"class_declaration", "interface_declaration", "enum_declaration"},
    "rust":       {"impl_item", "struct_item", "enum_item", "trait_item", "mod_item"},
    "ruby":       {"class", "module"},
    "cpp":        {"class_specifier", "struct_specifier", "namespace_definition"},
    "c_sharp":    {"class_declaration", "interface_declaration", "namespace_declaration", "enum_declaration"},
    "php":        {"class_declaration", "interface_declaration", "trait_declaration"},
    "kotlin":     {"class_declaration", "object_declaration"},
    "swift":      {"class_declaration", "struct_declaration", "protocol_declaration", "extension_declaration"},
    "go":         {"type_declaration"},
}

IMPORT_NODE_TYPES: Dict[str, set] = {
    "python":     {"import_statement", "import_from_statement"},
    "javascript": {"import_statement", "import_declaration"},
    "typescript": {"import_statement", "import_declaration"},
    "tsx":        {"import_statement", "import_declaration"},
    "go":         {"import_declaration"},
    "java":       {"import_declaration"},
    "rust":       {"use_declaration"},
    "ruby":       {"require", "require_relative"},
    "cpp":        {"preproc_include"},
    "c":          {"preproc_include"},
    "c_sharp":    {"using_directive"},
    "kotlin":     {"import_header"},
    "swift":      {"import_declaration"},
    "php":        {"include_expression", "require_expression", "use_declaration"},
    "lua":        {"require_call"},
    "bash":       {"source_command"},
}

VAR_NODE_TYPES: Dict[str, set] = {
    "python":     {"expression_statement", "assignment"},
    "javascript": {"lexical_declaration", "variable_declaration", "expression_statement"},
    "typescript": {"lexical_declaration", "variable_declaration", "expression_statement"},
    "tsx":        {"lexical_declaration", "variable_declaration", "expression_statement"},
    "go":         {"var_declaration", "const_declaration", "short_var_declaration"},
    "java":       {"field_declaration", "local_variable_declaration"},
    "rust":       {"static_item", "const_item", "let_declaration"},
    "ruby":       {"assignment"},
    "cpp":        {"declaration"},
    "c":          {"declaration"},
    "c_sharp":    {"field_declaration", "local_declaration"},
    "kotlin":     {"property_declaration"},
    "swift":      {"property_declaration"},
}

CALL_NODE_TYPES: Dict[str, set] = {
    "python":     {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "tsx":        {"call_expression", "new_expression"},
    "go":         {"call_expression"},
    "java":       {"method_invocation", "object_creation_expression"},
    "rust":       {"call_expression", "macro_invocation"},
    "ruby":       {"call", "method_call"},
    "cpp":        {"call_expression"},
    "c":          {"call_expression"},
    "c_sharp":    {"invocation_expression", "object_creation_expression"},
    "kotlin":     {"call_expression"},
    "swift":      {"call_expression"},
    "lua":        {"function_call"},
    "r":          {"call"},
    "bash":       {"command"},
    "html":       {"call_expression"},
    "css":        set(),
}

MAX_CHUNK_TOKENS  = 400
SLIDE_OVERLAP     = 50
NESTED_SIZE_LIMIT = MAX_CHUNK_TOKENS * 2


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _chunk_id(file_path: str, name: str, start_line: int) -> str:
    return f"{file_path}::{name}::{start_line}"


def _make_chunk(
    text: str,
    file_path: str,
    doc_type: str,
    name: str = "",
    start_line: int = 0,
    parent_name: str = "",
    language: str = "",
    raw_calls: Optional[List[str]] = None,
    symbol_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    full_name = (
        f"{parent_name}.{name}" if parent_name and name
        else (name or f"unnamed::{start_line}")
    )
    chunk: Dict[str, Any] = {
        "id":         _chunk_id(file_path, full_name, start_line),
        "file_path":  file_path,
        "doc_type":   doc_type,
        "name":       full_name,
        "parent":     parent_name,
        "start_line": start_line,
        "language":   language,
        "calls":      [],
        "called_by":  [],
        "_raw":       raw_calls or [],
        "text":       text.strip(),
    }
    if symbol_names is not None:
        chunk["symbol_names"] = symbol_names
    return chunk


def _extract_name(node, src_bytes: bytes) -> str:
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier",
                          "property_identifier", "field_identifier", "simple_identifier"):
            return src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# AST call collector
# ─────────────────────────────────────────────────────────────────────────────

def _collect_calls(node, src_bytes: bytes, lang: str, results: Set[str], depth: int = 0):
    if depth > 40:
        return
    call_types   = CALL_NODE_TYPES.get(lang, set())
    import_types = IMPORT_NODE_TYPES.get(lang, set())

    if node.type in call_types and node.children:
        callee = node.children[0]
        if callee.type in ("identifier", "name", "simple_identifier"):
            results.add(src_bytes[callee.start_byte:callee.end_byte].decode("utf-8", errors="replace"))
        elif callee.type in ("member_expression", "attribute", "field_expression",
                              "qualified_name", "selector_expression", "scoped_identifier"):
            for c in reversed(callee.children):
                if c.type in ("identifier", "property_identifier", "field_identifier",
                               "name", "simple_identifier"):
                    results.add(src_bytes[c.start_byte:c.end_byte].decode("utf-8", errors="replace"))
                    break

    if node.type in import_types:
        for c in node.children:
            if c.type in ("identifier", "name", "dotted_name", "type_identifier", "simple_identifier"):
                raw = src_bytes[c.start_byte:c.end_byte].decode("utf-8", errors="replace")
                results.add(raw.split(".")[-1])

    for child in node.children:
        _collect_calls(child, src_bytes, lang, results, depth + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency resolver — now accepts existing_name_map for incremental updates
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_deps(
    chunks: List[Dict[str, Any]],
    existing_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Resolves _raw call names → chunk ids globally.

    existing_name_map: {short_name: chunk_id} pre-populated from Qdrant
    for unchanged files. Pass this on incremental runs so cross-file deps
    to unchanged files still resolve correctly.
    """
    # Start with existing map (unchanged files from Qdrant) if provided
    name_to_id: Dict[str, str] = dict(existing_name_map) if existing_name_map else {}

    # Add new chunks — new chunks override existing on conflict (they are fresher)
    for ch in chunks:
        for key in (ch["name"], ch["name"].split(".")[-1]):
            if key:
                name_to_id[key] = ch["id"]

    id_map = {ch["id"]: ch for ch in chunks}

    for ch in chunks:
        resolved, seen = [], set()
        for raw in ch.pop("_raw", []):
            cid = name_to_id.get(raw)
            if cid and cid != ch["id"] and cid not in seen:
                resolved.append(cid)
                seen.add(cid)
        ch["calls"] = resolved

    # Back-fill called_by only for NEW chunks (can't mutate Qdrant chunks here)
    for ch in chunks:
        for callee_id in ch["calls"]:
            callee = id_map.get(callee_id)
            if callee and ch["id"] not in callee["called_by"]:
                callee["called_by"].append(ch["id"])

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Context-line extractor
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_lines(
    node, src_bytes: bytes, lang: str,
    target_types: set, class_types: set, import_types: set, var_types: set,
    scope_name: str = "",
) -> Tuple[List[str], List[str]]:
    full_src  = src_bytes.decode("utf-8", errors="replace").splitlines()
    ctx: List[str] = []
    syms: List[str] = []

    def node_lines(n) -> List[str]:
        return full_src[n.start_point[0]: n.end_point[0] + 1]

    for child in node.children:
        t = child.type
        if t in import_types and not scope_name:
            ctx.extend(node_lines(child)); ctx.append("")
        elif t in class_types:
            name = _extract_name(child, src_bytes)
            if name:
                ctx.append(f"# class: {name}")
                ctx.extend(node_lines(child)[:6]); ctx.append("")
                syms.append(name)
        elif t in target_types and t not in class_types:
            name = _extract_name(child, src_bytes)
            if name:
                ctx.extend(node_lines(child)[:3]); ctx.append("")
                syms.append(f"{scope_name}.{name}" if scope_name else name)
        elif t in var_types:
            lines = node_lines(child)
            if len(lines) <= 8:
                ctx.extend(lines); ctx.append("")

    return ctx, syms


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window fallback
# ─────────────────────────────────────────────────────────────────────────────

def sliding_window_chunks(
    text: str, file_path: str, doc_type: str = "text", language: str = "",
    max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = SLIDE_OVERLAP,
) -> List[Dict[str, Any]]:
    lines  = text.splitlines()
    step   = max(1, max_tokens - overlap)
    chunks = []
    i = 0
    while i < len(lines):
        content = "\n".join(lines[i: i + max_tokens])
        if content.strip():
            chunks.append(_make_chunk(content, file_path, doc_type,
                                      name=f"chunk::{i}", start_line=i, language=language))
        i += step
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# AST walk — function/method-level chunks
# ─────────────────────────────────────────────────────────────────────────────

def _walk_node(
    node, src_bytes: bytes, file_path: str, doc_type: str, language: str,
    target_types: set, class_types: set, import_types: set, var_types: set,
    parent_name: str = "", depth: int = 0,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    for child in node.children:

        if child.type in class_types:
            class_name = _extract_name(child, src_bytes)
            full_class = f"{parent_name}.{class_name}" if parent_name and class_name else class_name

            if class_name:
                ctx_lines, method_syms = _build_context_lines(
                    child, src_bytes, language, target_types, class_types,
                    import_types, var_types, scope_name=class_name,
                )
                ctx_text = "\n".join(ctx_lines).strip()
                if ctx_text:
                    raw: Set[str] = set()
                    _collect_calls(child, src_bytes, language, raw)
                    chunks.append(_make_chunk(
                        ctx_text, file_path, doc_type + "_context",
                        name=f"{class_name}[context]",
                        start_line=child.start_point[0],
                        parent_name=parent_name, language=language,
                        raw_calls=list(raw), symbol_names=method_syms,
                    ))

            sub = _walk_node(child, src_bytes, file_path, doc_type, language,
                             target_types, class_types, import_types, var_types,
                             parent_name=full_class, depth=depth + 1)
            chunks.extend(sub)
            continue

        if child.type not in target_types:
            sub = _walk_node(child, src_bytes, file_path, doc_type, language,
                             target_types, class_types, import_types, var_types,
                             parent_name=parent_name, depth=depth + 1)
            chunks.extend(sub)
            continue

        node_text = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        node_name = _extract_name(child, src_bytes)
        raw: Set[str] = set()
        _collect_calls(child, src_bytes, language, raw)

        if _approx_tokens(node_text) <= NESTED_SIZE_LIMIT:
            emit = (f"# context: inside {parent_name}\n" + node_text) if parent_name else node_text
            chunks.append(_make_chunk(
                emit, file_path, doc_type,
                name=node_name, start_line=child.start_point[0],
                parent_name=parent_name, language=language, raw_calls=list(raw),
            ))
        else:
            nested_parent = f"{parent_name}.{node_name}" if parent_name else node_name
            sub = _walk_node(child, src_bytes, file_path, doc_type, language,
                             target_types, class_types, import_types, var_types,
                             parent_name=nested_parent, depth=depth + 1)
            if sub:
                cutoff = sub[0]["start_line"] - child.start_point[0]
                header = "\n".join(node_text.splitlines()[:max(cutoff, 8)])
                if header.strip():
                    chunks.append(_make_chunk(
                        header, file_path, doc_type,
                        name=f"{node_name}[header]",
                        start_line=child.start_point[0],
                        parent_name=parent_name, language=language, raw_calls=list(raw),
                    ))
            chunks.extend(sub)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# File-level context chunk
# ─────────────────────────────────────────────────────────────────────────────

def _build_file_context_chunks(
    root_node, src_bytes: bytes, file_path: str, doc_type: str, language: str,
    target_types: set, class_types: set, import_types: set, var_types: set,
    all_symbol_names: List[str],
) -> List[Dict[str, Any]]:
    ctx_lines, local_syms = _build_context_lines(
        root_node, src_bytes, language, target_types, class_types, import_types, var_types,
    )
    all_syms = list(dict.fromkeys(local_syms + all_symbol_names))
    ctx_text = "\n".join(ctx_lines).strip()
    if not ctx_text:
        return []

    raw: Set[str] = set()
    _collect_calls(root_node, src_bytes, language, raw)

    if _approx_tokens(ctx_text) <= MAX_CHUNK_TOKENS:
        return [_make_chunk(
            ctx_text, file_path, doc_type + "_context",
            name="[file_context]", start_line=0, language=language,
            raw_calls=list(raw), symbol_names=all_syms,
        )]

    lines, step, chunks, i = ctx_text.splitlines(), max(1, MAX_CHUNK_TOKENS - SLIDE_OVERLAP), [], 0
    while i < len(lines):
        piece = "\n".join(lines[i: i + MAX_CHUNK_TOKENS]).strip()
        if piece:
            chunks.append(_make_chunk(
                piece, file_path, doc_type + "_context",
                name=f"[file_context::{i}]", start_line=i, language=language,
                raw_calls=list(raw), symbol_names=all_syms if i == 0 else [],
            ))
        i += step
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# tree_sitter_chunks — returns RAW chunks, NO dep resolution here
# ─────────────────────────────────────────────────────────────────────────────

def tree_sitter_chunks(text: str, file_path: str, lang_name: str, doc_type: str) -> List[Dict[str, Any]]:
    """
    Returns raw chunks with _raw call names intact.
    _resolve_deps is NOT called here — it runs once globally in chunk_repo.
    """
    if not TREE_SITTER_OK:
        return sliding_window_chunks(text, file_path, doc_type, lang_name)
    try:
        get_language(lang_name)
        parser = get_parser(lang_name)
    except Exception as e:
        print(f"[chunker] language={lang_name} parser failed: {e}")
        return sliding_window_chunks(text, file_path, doc_type, lang_name)

    src_bytes    = text.encode("utf-8")
    tree         = parser.parse(src_bytes)
    target_types = CHUNK_NODE_TYPES.get(lang_name, set())
    class_types  = CLASS_NODE_TYPES.get(lang_name, set())
    import_types = IMPORT_NODE_TYPES.get(lang_name, set())
    var_types    = VAR_NODE_TYPES.get(lang_name, set())

    fn_chunks  = _walk_node(tree.root_node, src_bytes, file_path, doc_type, lang_name,
                            target_types, class_types, import_types, var_types)
    ctx_chunks = _build_file_context_chunks(
        tree.root_node, src_bytes, file_path, doc_type, lang_name,
        target_types, class_types, import_types, var_types,
        all_symbol_names=[c["name"] for c in fn_chunks],
    )

    all_chunks = ctx_chunks + fn_chunks
    if not all_chunks:
        return sliding_window_chunks(text, file_path, doc_type, lang_name)

    # ← _resolve_deps removed from here, now runs globally in chunk_repo
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Notebook chunker
# ─────────────────────────────────────────────────────────────────────────────

def notebook_chunks(path: str) -> List[Dict[str, Any]]:
    if not NBFORMAT_OK:
        text   = Path(path).read_text(encoding="utf-8", errors="replace")
        chunks = []
        for i in range(0, len(text), 8000):
            piece = text[i: i + 8000]
            if piece.strip():
                chunks.append(_make_chunk(piece, path, "notebook", name=f"chunk::{i}"))
        return chunks

    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    chunks, md_buf, line = [], [], 0
    for cell in nb.cells:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if not src.strip():
            continue
        if cell["cell_type"] == "markdown":
            md_buf.append(src)
        elif cell["cell_type"] == "code":
            if md_buf:
                chunks.append(_make_chunk("\n\n".join(md_buf), path, "notebook_markdown",
                                          name=f"md::{line}", start_line=line))
                md_buf = []
            chunks.append(_make_chunk(src, path, "notebook",
                                      name=f"cell::{line}", start_line=line, language="python"))
            line += 1
    if md_buf:
        chunks.append(_make_chunk("\n\n".join(md_buf), path, "notebook_markdown",
                                  name=f"md::{line}", start_line=line))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Config / Markdown chunkers
# ─────────────────────────────────────────────────────────────────────────────

def config_chunks(text: str, file_path: str) -> List[Dict[str, Any]]:
    if _approx_tokens(text) <= 300:
        return [_make_chunk(text, file_path, "config", name="[config]")]
    lines, chunks, buf, start = text.splitlines(), [], [], 0
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith(("#", "//")):
            if buf:
                chunks.append(_make_chunk("\n".join(buf), file_path, "config",
                                          name=f"config::{start}", start_line=start))
            buf, start = [line], i
        else:
            buf.append(line)
    if buf:
        chunks.append(_make_chunk("\n".join(buf), file_path, "config",
                                  name=f"config::{start}", start_line=start))
    return chunks


def markdown_chunks(text: str, file_path: str) -> List[Dict[str, Any]]:
    parts = re.split(r"(?m)^(#{1,3} )", text)
    chunks, buf, i = [], [], 0
    for part in parts:
        if re.match(r"^#{1,3} $", part):
            buf = [part]
        else:
            buf.append(part)
            combined = "".join(buf).strip()
            if combined:
                chunks.append(_make_chunk(combined, file_path, "markdown",
                                          name=f"section::{i}"))
                i += 1
            buf = []
    if buf:
        combined = "".join(buf).strip()
        if combined:
            chunks.append(_make_chunk(combined, file_path, "markdown", name=f"section::{i}"))
    return chunks or [_make_chunk(text, file_path, "markdown", name="section::0")]


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             "dist", "build", ".mypy_cache", ".pytest_cache"}
SKIP_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
             ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
             ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".wav",
             ".zip", ".tar", ".gz", ".lock", ".sum"}


def chunk_file(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    ext  = path.suffix.lower()
    stem = path.name.lower()

    if ext == ".ipynb":
        return notebook_chunks(file_path)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[chunker] Cannot read {file_path}: {e}")
        return []

    if not text.strip():
        return []

    if ext in {".yaml", ".yml", ".json", ".toml"}:
        return config_chunks(text, file_path)

    if ext in {".md", ".rst", ".mdx"}:
        return markdown_chunks(text, file_path)

    if ext in {".txt", ".log", ".csv", ""}:
        return sliding_window_chunks(text, file_path, "text")

    if ext in EXT_TO_LANG:
        lang_name = EXT_TO_LANG[ext]
        if ext in {".css", ".scss"}:      doc_type = "style"
        elif ext == ".html":              doc_type = "markup"
        elif (stem.startswith("test_") or
              ".spec." in stem or "_test." in stem): doc_type = "test"
        else:                             doc_type = "code"
        return tree_sitter_chunks(text, file_path, lang_name, doc_type)

    return sliding_window_chunks(text, file_path, "unknown")


def chunk_repo(
    repo_root: str,
    changed_files: List[str] = None,
    existing_name_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Walk a repo and chunk every relevant file.
    Resolves cross-file deps GLOBALLY after all files are chunked.

    Args:
        repo_root:         path to local repo
        changed_files:     if set, only chunk these files (incremental)
        existing_name_map: {short_name: chunk_id} from Qdrant for unchanged files.
                           Pass this on incremental runs so cross-file deps to
                           unchanged files still resolve. Built by ingestor from
                           Qdrant scroll. If None, only new chunks resolve against
                           each other (fine for full re-index).
    """
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

    print(f"[chunker] Total chunks before dep resolution: {len(all_chunks)}")

    # Global dep resolution — cross-file calls now resolve correctly
    # existing_name_map seeds unchanged files' symbols for incremental runs
    all_chunks = _resolve_deps(all_chunks, existing_name_map=existing_name_map)

    print(f"[chunker] Total chunks: {len(all_chunks)}")
    return all_chunks