"""
================================================================================
  semantic.py  –  Phase 5 (Symbol Table + Scope) & Phase 6 (Semantic Analysis)
================================================================================

  PHASE 5  –  Symbol Table & Scope Handling
  ──────────────────────────────────────────
  • SymbolEntry  : name | type | scope | offset | line
  • SymbolTable  : insert(), lookup(), push_scope(), pop_scope()
                   (innermost-to-outermost search, same as C version)

  PHASE 6  –  Semantic Analysis  (walks the ParseNode tree)
  ──────────────────────────────────────────────────────────
  Detects:
    1. Use of undeclared variables
    2. Multiple declarations in the same scope
    3. Type mismatches in assignments / declarations
    4. Invalid boolean conditions (missing relational operator)

================================================================================
"""

import sys
from typing import List, Optional

from lexer import Lexer
from parser import Parser, ParseNode


# ==============================================================================
#  PHASE 5 – SYMBOL TABLE & SCOPE HANDLING
# ==============================================================================

class SymbolEntry:
    """
    Mirrors the C struct Symbol 
        char name[MAX_LABEL]
        char type[MAX_VALUE]    ("int" or "float")
        int  scope              (depth at which it was declared)
        int  offset             (simulated memory offset in bytes)
        int  line               (source line of declaration)
    """
    def __init__(self, name: str, var_type: str, scope: int,
                 offset: int, line: int):
        self.name   = name
        self.type   = var_type
        self.scope  = scope
        self.offset = offset
        self.line   = line


class SymbolTable:
    """
    Mirrors the symbol-table machinery 

    State
    ─────
    sym_table      : flat list of SymbolEntry  (never shrinks – same as C array)
    current_scope  : int   depth counter (0 = global)
    scope_offset   : dict  running byte offset per scope level
                     int=4 bytes, float=4 bytes  
    """

    _SIZE = {"int": 4, "float": 4} 

    def __init__(self):
        self.sym_table:     List[SymbolEntry] = []
        self.current_scope: int               = 0
        self._scope_offset: dict              = {0: 0}
        # kick off exactly like phase5.c main():
        print("  [SCOPE] >>> Entering scope level 0  (global)")

    # push_scope  (phase5.c push_scope)
    def push_scope(self):
        self.current_scope += 1
        self._scope_offset[self.current_scope] = 0
        print(f"  [SCOPE] >>> Entering scope level {self.current_scope}")

    # pop_scope  (phase5.c pop_scope)
    def pop_scope(self):
        print(f"  [SCOPE] <<< Leaving scope level {self.current_scope}"
              f"  (symbols declared here:)")
        found = False
        for s in self.sym_table:
            if s.scope == self.current_scope:
                print(f"            {s.name:<12}  type={s.type:<6}  offset={s.offset}")
                found = True
        if not found:
            print("            (none)")
        self.current_scope -= 1

    # sym_insert  (phase5.c sym_insert + phase6.c sym_insert)
    def insert(self, name: str, var_type: str, line: int) -> bool:
        """Returns True on success; False if name already declared in THIS scope."""
        for s in self.sym_table:
            if s.scope == self.current_scope and s.name == name:
                return False   # duplicate

        sz     = self._SIZE.get(var_type, 4)
        offset = self._scope_offset.get(self.current_scope, 0)
        self._scope_offset[self.current_scope] = offset + sz

        self.sym_table.append(
            SymbolEntry(name, var_type, self.current_scope, offset, line)
        )
        return True

    def lookup(self, name: str) -> Optional[SymbolEntry]:
        best: Optional[SymbolEntry] = None
        for s in self.sym_table:
            if s.name == name and s.scope <= self.current_scope:
                if best is None or s.scope > best.scope:
                    best = s
        return best

    # print_symbol_table  (phase5.c print_symbol_table)
    def print_table(self):
        n  = len(self.sym_table)
        pl = "y" if n == 1 else "ies"
        W  = 80
        print()
        print("=" * W)
        print(f"  SYMBOL TABLE  ({n} entr{pl})")
        print("=" * W)
        print(f"  {'Name':<16}  {'Type':<8}  {'Scope':<7}  {'Offset':<10}  Line")
        print(f"  {'----------------':<16}  {'--------':<8}  {'-------':<7}  {'----------':<10}  ----")
        for s in self.sym_table:
            print(f"  {s.name:<16}  {s.type:<8}  {s.scope:<7}  {s.offset:<10}  {s.line}")
        print("=" * W)
        print()


# ==============================================================================
#  PHASE 6 – SEMANTIC ANALYSIS
# ==============================================================================

class SemanticAnalyser:
    """
    Mirrors sem_walk(), infer_type(), has_relop(), sem_error() 
    Uses the same SymbolTable (phase 5 machinery).
    """

    def __init__(self):
        self.sym        = SymbolTable()
        self.sem_errors = 0

        # For the consolidated error report printed at the end:
        #   _error_log   : ordered list of (line, category, detail) for every
        #                  unique error (duplicates suppressed)
        #   _seen_errors : set of (line, msg) keys to suppress repeats
        self._error_log:   List[tuple] = []
        self._seen_errors: set         = set()

    # ── error categories (used in the consolidated report) ───────────────────
    _CAT = {
        "undeclared"  : "UNDECLARED VARIABLE",
        "duplicate"   : "DUPLICATE DECLARATION",
        "type_mismatch": "TYPE MISMATCH",
        "invalid_bool": "INVALID BOOLEAN CONDITION",
    }

    def _categorise(self, msg: str) -> tuple:
        """Return (category_key, detail) from a raw error message."""
        if msg.startswith("use of undeclared variable"):
            return ("undeclared", msg)
        if msg.startswith("multiple declaration"):
            return ("duplicate", msg)
        if msg.startswith("type mismatch"):
            return ("type_mismatch", msg)
        if "invalid boolean condition" in msg:
            return ("invalid_bool", msg)
        return ("other", msg)

    # sem_error   – now with deduplication
    def _sem_error(self, msg: str, line: int):
        key = (line, msg)
        if key in self._seen_errors:
            return                          # suppress exact duplicate
        self._seen_errors.add(key)

        cat, detail = self._categorise(msg)
        self._error_log.append((line, cat, detail))
        self.sem_errors += 1

        print(f"  [SEMANTIC ERROR] Line {line}: {msg}")

    # infer_type  
    def _infer_type(self, node: ParseNode) -> str:
        """Walk expr/term/factor subtree; return "int", "float", or "unknown"."""
        if node is None:
            return "unknown"

        label = node.label

        if label == "INTEGER":
            return "int"

        if label in ("FLOAT_LIT", "FLOAT"):
            return "float"

        if label == "ID":
            s = self.sym.lookup(node.token.value)
            if s is None:
                self._sem_error(
                    f"use of undeclared variable '{node.token.value}'",
                    node.token.line
                )
                return "unknown"
            return s.type

        # composite node: propagate float upward  (same as C version)
        result = "int"
        for child in node.children:
            ct = self._infer_type(child)
            if ct == "float":
                result = "float"
            elif ct == "unknown" and result != "float":
                result = "unknown"
        return result

    # has_relop  (phase6.c has_relop) – fixed to check child count
    def _has_relop(self, node: ParseNode) -> bool:
        """
        Return True only when the subtree contains a VALID rel_expr,
        i.e. one with 3 children (expr  rel_op  expr).
        A rel_expr with only 1 child is a bare expression – no operator present.
        """
        if node is None:
            return False
        if node.label == "rel_op":
            return True
        if node.label == "rel_expr":
            # 3 children → expr  rel_op  expr  ✓
            # 1 child    → bare expression only ✗
            return len(node.children) == 3
        for child in node.children:
            if self._has_relop(child):
                return True
        return False

    # ── helpers for detailed invalid-bool reporting ───────────────────────────

    def _collect_bare_rel_exprs(self, node: ParseNode, results: list):
        """Find every rel_expr node that has only 1 child (no relational op)."""
        if node is None:
            return
        if node.label == "rel_expr":
            if len(node.children) != 3:        # bare – missing rel_op
                results.append(node)
            return                              # don't recurse inside rel_expr
        for child in node.children:
            self._collect_bare_rel_exprs(child, results)

    def _expr_repr(self, node: ParseNode) -> str:
        """Best-effort readable string for a subtree (used in error messages)."""
        if node is None:
            return "?"
        if node.is_leaf() and node.token:
            return node.token.value
        return " ".join(self._expr_repr(c) for c in node.children)

    def _check_bool_condition(self, cond_node: ParseNode, stmt_line: int):
        """
        Walk the condition subtree.
        Report every rel_expr that is missing a relational operator,
        i.e. a bare variable / expression used directly as a boolean.
        """
        bare_nodes = []
        self._collect_bare_rel_exprs(cond_node, bare_nodes)
        for re_node in bare_nodes:
            bare_name = self._expr_repr(re_node).strip()
            line      = self._node_line(re_node) or stmt_line
            self._sem_error(
                f"invalid boolean condition - '{bare_name}' is used as a "
                f"condition but has no relational operator "
                f"(e.g. write '{bare_name} > 0' instead)",
                line
            )

    # sem_walk  (phase6.c sem_walk)
    def _sem_walk(self, node: ParseNode):
        if node is None:
            return

        label = node.label

        # ── BLOCK: push / pop scope ───────────────────────────────────────────
        if label == "block":
            self.sym.push_scope()
            for child in node.children:
                self._sem_walk(child)
            self.sym.pop_scope()
            return

        # ── DECLARATION ───────────────────────────────────────────────────────
        if label == "decl_stmt":
            decl_type = ""
            var_name  = ""
            init_expr = None

            for i, c in enumerate(node.children):
                if c.label == "TYPE":
                    # TYPE node wraps a leaf token
                    decl_type = (c.children[0].token.value
                                 if c.children else "")
                elif c.label == "ID":
                    var_name = c.token.value
                elif c.label == "ASSIGN" and i + 1 < len(node.children):
                    init_expr = node.children[i + 1]
                elif c.label == "expr" and init_expr is None:
                    init_expr = c

            line = self._node_line(node)

            # ② duplicate declaration
            if not self.sym.insert(var_name, decl_type, line):
                self._sem_error(
                    f"multiple declaration of '{var_name}' "
                    f"in scope {self.sym.current_scope}",
                    line
                )
            else:
                s = self.sym.lookup(var_name)
                print(f"  [DECL]   {var_name:<12}  type={decl_type:<6}  "
                      f"scope={s.scope}  (line {s.line})")

            # ③ type mismatch in initialiser
            if init_expr:
                rhs = self._infer_type(init_expr)
                if rhs != "unknown" and decl_type == "int" and rhs == "float":
                    self._sem_error(
                        f"type mismatch: cannot assign float expression "
                        f"to int variable '{var_name}'",
                        line
                    )
            return

        # ── ASSIGNMENT ────────────────────────────────────────────────────────
        if label == "assign_stmt":
            var_name = ""
            rhs_expr = None

            for i, c in enumerate(node.children):
                if c.label == "ID":
                    var_name = c.token.value
                elif c.label == "ASSIGN" and i + 1 < len(node.children):
                    rhs_expr = node.children[i + 1]
                elif c.label == "expr" and rhs_expr is None:
                    rhs_expr = c

            line = self._node_line(node)
            s    = self.sym.lookup(var_name)

            # ① undeclared variable
            if s is None:
                self._sem_error(
                    f"use of undeclared variable '{var_name}'", line
                )

            if rhs_expr:
                rhs = self._infer_type(rhs_expr)
                # ③ type mismatch
                if (s is not None and rhs != "unknown"
                        and s.type == "int" and rhs == "float"):
                    self._sem_error(
                        f"type mismatch: cannot assign float expression "
                        f"to int variable '{var_name}'",
                        line
                    )
                if s is not None:
                    print(f"  [ASSIGN] {var_name:<12} = {rhs:<6} expression  "
                          f"(line {line})")
            return

        # ── IF STATEMENT ──────────────────────────────────────────────────────
        if label == "if_stmt":
            stmt_line = self._node_line(node)
            print(f"  [IF]     checking condition (line {stmt_line})")
            for c in node.children:
                if c.label == "bool_expr":
                    # ④ check every rel_expr inside the condition for missing rel_op
                    self._check_bool_condition(c, stmt_line)
                self._sem_walk(c)
            return

        # ── WHILE STATEMENT ───────────────────────────────────────────────────
        if label == "while_stmt":
            stmt_line = self._node_line(node)
            print(f"  [WHILE]  checking condition (line {stmt_line})")
            for c in node.children:
                if c.label == "bool_expr":
                    # ④ check every rel_expr inside the condition for missing rel_op
                    self._check_bool_condition(c, stmt_line)
                self._sem_walk(c)
            return

        # ── PRINT STATEMENT ───────────────────────────────────────────────────
        if label == "print_stmt":
            print(f"  [PRINT]  (line {self._node_line(node)})")
            for c in node.children:
                self._sem_walk(c)
            return

        # ── standalone ID (inside bool / print expressions) ───────────────────
        if label == "ID":
            if self.sym.lookup(node.token.value) is None:
                self._sem_error(
                    f"use of undeclared variable '{node.token.value}'",
                    node.token.line
                )
            return

        # ── default: recurse into children ────────────────────────────────────
        for c in node.children:
            self._sem_walk(c)

    # public entry point
    def analyse(self, tree: ParseNode):
        self._sem_walk(tree)

    # helper: get line number from a ParseNode
    def _node_line(self, node: ParseNode) -> int:
        if node.is_leaf() and node.token:
            return node.token.line
        for c in node.children:
            line = self._node_line(c)
            if line:
                return line
        return 0

    # print_results – consolidated error report + symbol table
    def print_results(self):
        W = 80

        # ── Symbol table ──────────────────────────────────────────────────────
        print()
        print("=" * W)
        n  = len(self.sym.sym_table)
        pl = "y" if n == 1 else "ies"
        print(f"  SYMBOL TABLE  ({n} entr{pl})")
        print("=" * W)
        print(f"  {'Name':<16}  {'Type':<8}  {'Scope':<7}  {'Offset':<10}  Line")
        print(f"  {'----------------':<16}  {'--------':<8}  {'-------':<7}  {'----------':<10}  ----")
        for s in self.sym.sym_table:
            print(f"  {s.name:<16}  {s.type:<8}  {s.scope:<7}  {s.offset:<10}  {s.line}")
        print("=" * W)
        print()

        # ── Consolidated error report ─────────────────────────────────────────
        print("=" * W)
        if self.sem_errors == 0:
            print("  SEMANTIC ANALYSIS PASSED  –  no errors found.")
            print("=" * W)
            return

        print(f"  SEMANTIC ERROR REPORT  ({self.sem_errors} error(s) found)")
        print("=" * W)

        # Group errors by category, in the order they first appeared
        _EXPLAIN = {
            "undeclared": (
                "UNDECLARED VARIABLE",
                "The variable is used but was never declared with 'int' or 'float'.\n"
                "    Fix : declare it before use, e.g.  int b;  or  float sum;"
            ),
            "duplicate": (
                "DUPLICATE DECLARATION",
                "The same variable name is declared more than once in the same scope.\n"
                "    Fix : remove the second declaration or rename the variable."
            ),
            "type_mismatch": (
                "TYPE MISMATCH",
                "A float expression is being assigned to an int variable.\n"
                "    Fix : change the variable type to 'float', or cast the value."
            ),
            "invalid_bool": (
                "INVALID BOOLEAN CONDITION",
                "The if/while condition does not contain a relational operator\n"
                "    (< > <= >= == !=).  A boolean condition must compare two values.\n"
                "    Fix : use a proper comparison, e.g.  if (x > 0) { ... }"
            ),
            "other": (
                "SEMANTIC ERROR",
                "An unclassified semantic error was detected."
            ),
        }

        # collect unique categories in appearance order
        seen_cats: list = []
        cat_errors: dict = {}
        for (line, cat, detail) in self._error_log:
            if cat not in cat_errors:
                cat_errors[cat] = []
                seen_cats.append(cat)
            cat_errors[cat].append((line, detail))

        error_num = 1
        for cat in seen_cats:
            title, explanation = _EXPLAIN.get(cat, _EXPLAIN["other"])
            occurrences = cat_errors[cat]

            print()
            print(f"  ┌─ ERROR TYPE : {title}")
            print(f"  │  EXPLANATION: {explanation}")
            print(f"  │  OCCURRENCES ({len(occurrences)}):")
            for (line, detail) in occurrences:
                print(f"  │    [{error_num}] Line {line:>3} –  {detail}")
                error_num += 1
            print(f"  └{'─' * (W - 4)}")

        print()
        print("=" * W)
        print(f"  Total: {self.sem_errors} semantic error(s).  "
              f"Fix all errors above and re-run.")
        print("=" * W)
        print()


# ==============================================================================
#  MAIN  (mirrors phase5.c + phase6.c main())
# ==============================================================================

def main():
    W = 80

    if len(sys.argv) < 2:
        print("Usage: python semantic.py <source_file>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)

    # ── Lex ───────────────────────────────────────────────────────────────────
    lexer  = Lexer(source)
    tokens = lexer.tokenize()

    if lexer.errors:
        print("Lexical errors:")
        for e in lexer.errors:
            print(" ", e)
        sys.exit(1)

    # ── Parse ─────────────────────────────────────────────────────────────────
    parser = Parser(tokens)
    try:
        tree = parser.parse_program()
    except Exception as e:
        parser.errors.append(str(e))
        tree = None

    if parser.errors:
        print("=" * W)
        print("  Syntax errors found - semantic analysis aborted.")
        print("=" * W)
        for i, err in enumerate(parser.errors, 1):
            print(f"  [{i}] {err}")
        print("=" * W)
        sys.exit(1)

    # ── Phase 6 header  (mirrors phase6.c main) ───────────────────────────────
    print("=" * W)
    print("  PHASE 6 - SEMANTIC ANALYSIS")
    print("=" * W)
    print(f"  Loaded {len(tokens)} tokens\n")

    print("=" * W)
    print("  SEMANTIC ANALYSIS TRACE")
    print("=" * W)
    print()

    # ── Run semantic analysis ─────────────────────────────────────────────────
    analyser = SemanticAnalyser()
    analyser.analyse(tree)

    # ── Close global scope  (mirrors phase5.c closing line) ───────────────────
    print("  [SCOPE] <<< Leaving scope level 0  (global)")

    # ── Results + symbol table ────────────────────────────────────────────────
    analyser.print_results()

    sys.exit(1 if analyser.sem_errors else 0)


if __name__ == "__main__":
    main()