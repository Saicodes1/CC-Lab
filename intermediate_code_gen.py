"""
================================================================================
  PHASE 7 - INTERMEDIATE CODE GENERATION (THREE-ADDRESS CODE / QUADRUPLES)
================================================================================

  Translates directly from the ParseNode tree produced by parser.py,
  using the SemanticAnalyser from semantic.py.

  Mirrors phase7.c exactly:
    • Quadruple format  : (op, arg1, arg2, result)
    • Temporaries       : t1, t2, …
    • Labels            : L1, L2, …
    • Scope mangling    : global vars keep their name; inner-scope vars → name_sN
    • Saves quads to    : Phase7_quadruples.txt
    • Interactive step-by-step viewer per statement
================================================================================
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from lexer import Lexer
from parser import Parser, ParseNode
from semantic import SemanticAnalyser, SymbolTable, SymbolEntry


# ==============================================================================
#  QUADRUPLE
# ==============================================================================

@dataclass
class Quad:
    op:     str
    arg1:   str
    arg2:   str
    result: str


# ==============================================================================
#  TAC GENERATOR
# ==============================================================================

class TACGenerator:
    """
    Mirrors all the gen_* functions in phase7.c.
    Uses the same SymbolTable from semantic.py to resolve variable scopes.
    """

    def __init__(self, sym: SymbolTable):
        self.sym         = sym          # populated symbol table from semantic pass
        self.quads: List[Quad] = []
        self.temp_count  = 0
        self.label_count = 0

    # ── Counters ──────────────────────────────────────────────────────────────

    def new_temp(self) -> str:
        self.temp_count += 1
        return f"t{self.temp_count}"

    def new_label(self) -> str:
        self.label_count += 1
        return f"L{self.label_count}"

    # ── Emit ──────────────────────────────────────────────────────────────────

    def emit(self, op: str, arg1: str = "_", arg2: str = "_",
             result: str = "_") -> int:
        """Append one quadruple; returns its index."""
        self.quads.append(Quad(op, arg1 or "_", arg2 or "_", result or "_"))
        return len(self.quads) - 1

    def emit_label(self, lbl: str):
        self.emit("label", "_", "_", lbl)

    # ── Scope mangling (mirrors get_mangled_name) ─────────────────────────────

    def _mangle(self, name: str, scope: int) -> str:
        if scope == 0:
            return name
        return f"{name}_s{scope}"

    def _lookup_mangle(self, name: str) -> str:
        """Look up name in symbol table and return its mangled form."""
        s = self.sym.lookup(name)
        scope = s.scope if s else 0
        return self._mangle(name, scope)

    # ── Expression code generation ────────────────────────────────────────────

    def gen_expr(self, node: Optional[ParseNode]) -> str:
        """
        Mirrors gen_expr() in phase7.c.
        Returns the name/temp that holds the value of the expression.
        """
        if node is None:
            return "_"

        label = node.label

        # INTEGER / FLOAT literals
        if label == "INTEGER":
            return node.token.value
        if label in ("FLOAT_LIT", "FLOAT"):
            return node.token.value

        # ID – resolve scope
        if label == "ID":
            return self._lookup_mangle(node.token.value)

        # factor  →  single child
        if label == "factor":
            if len(node.children) == 1:
                return self.gen_expr(node.children[0])
            # parenthesised: ( expr )
            if len(node.children) == 3:
                return self.gen_expr(node.children[1])
            # unary minus: MINUS factor
            if len(node.children) == 2 and node.children[0].label == "MINUS":
                inner = self.gen_expr(node.children[1])
                t = self.new_temp()
                self.emit("-", "0", inner, t)
                return t
            return "_"

        # binary expr / term  (3 children: left  op  right)
        if label in ("expr", "term") and len(node.children) == 3:
            left_node, op_node, right_node = node.children
            # op_node is a leaf whose value is the operator symbol
            op = op_node.token.value if op_node.token else op_node.label
            l  = self.gen_expr(left_node)
            r  = self.gen_expr(right_node)
            t  = self.new_temp()
            self.emit(op, l, r, t)
            return t

        # single-child wrappers (expr/term/factor with 1 child)
        if node.children:
            return self.gen_expr(node.children[0])

        return "_"

    # ── Boolean code generation ───────────────────────────────────────────────

    def gen_bool(self, node: Optional[ParseNode],
                 ltrue: str, lfalse: str):
        """
        Mirrors gen_bool() in phase7.c.
        Emits conditional jumps; control reaches ltrue/lfalse.
        """
        if node is None:
            return

        label = node.label

        # ── NOT: swap true/false branches ────────────────────────────────────
        if label in ("bool_expr", "bool_factor") and node.children \
                and node.children[0].label == "NOT":
            # children: [NOT, bool_factor]
            self.gen_bool(node.children[1], lfalse, ltrue)
            return

        # ── OR / AND: 3-child bool nodes ─────────────────────────────────────
        # parser.py builds:  bool_expr(left, OR_leaf, right)
        #                    bool_term(left, AND_leaf, right)
        if label in ("bool_expr", "bool_term") and len(node.children) == 3:
            op_node = node.children[1]
            op_val  = op_node.token.value if op_node.token else op_node.label
            if op_val == "||":
                lnext = self.new_label()
                self.gen_bool(node.children[0], ltrue, lnext)
                self.emit_label(lnext)
                self.gen_bool(node.children[2], ltrue, lfalse)
                return
            if op_val == "&&":
                lnext = self.new_label()
                self.gen_bool(node.children[0], lnext, lfalse)
                self.emit_label(lnext)
                self.gen_bool(node.children[2], ltrue, lfalse)
                return

        # ── rel_expr with relational operator: expr  rel_op  expr ────────────
        if label == "rel_expr" and len(node.children) == 3:
            left_node, rel_node, right_node = node.children
            # rel_op node wraps a single leaf token
            if rel_node.label == "rel_op" and rel_node.children:
                op = rel_node.children[0].token.value
            elif rel_node.token:
                op = rel_node.token.value
            else:
                op = rel_node.label
            l = self.gen_expr(left_node)
            r = self.gen_expr(right_node)
            t = self.new_temp()
            self.emit(op, l, r, t)
            self.emit("if",   t, "_", ltrue)
            self.emit("goto", "_", "_", lfalse)
            return

        # ── rel_expr with 1 child: bare expression (no rel op) ───────────────
        if label == "rel_expr" and len(node.children) == 1:
            v = self.gen_expr(node.children[0])
            self.emit("if",   v, "_", ltrue)
            self.emit("goto", "_", "_", lfalse)
            return

        # ── Parenthesised bool_factor: ( bool_expr ) ─────────────────────────
        # parser.py: bool_factor([LPAREN_leaf, bool_expr, RPAREN_leaf])
        if label == "bool_factor" and len(node.children) == 3:
            self.gen_bool(node.children[1], ltrue, lfalse)
            return

        # ── Single-child wrapper nodes: recurse into child ───────────────────
        # Covers: bool_expr(1 child), bool_term(1 child), bool_factor(1 child)
        if len(node.children) == 1:
            self.gen_bool(node.children[0], ltrue, lfalse)
            return

        # ── Fallback: treat as an expression and branch on its value ─────────
        v = self.gen_expr(node)
        self.emit("if",   v, "_", ltrue)
        self.emit("goto", "_", "_", lfalse)

    # ── Statement code generation ─────────────────────────────────────────────

    def gen_decl(self, node: ParseNode):
        """Mirrors gen_decl() – only emits TAC if there is an initialiser."""
        var      = ""
        init     = None
        children = node.children
        for i, c in enumerate(children):
            if c.label == "ID":
                var = c.token.value
            elif c.label in ("ASSIGN", "ASSIGNMENT") and i + 1 < len(children):
                init = children[i + 1]
            elif c.label == "expr" and init is None and var:
                init = c  # handle parser style without explicit ASSIGN node
        if init:
            rhs     = self.gen_expr(init)
            mangled = self._lookup_mangle(var)
            self.emit("=", rhs, "_", mangled)

    def gen_assign(self, node: ParseNode):
        """Mirrors gen_assign()."""
        var      = ""
        rhs_node = None
        children = node.children
        for i, c in enumerate(children):
            if c.label == "ID":
                var = c.token.value
            elif c.label in ("ASSIGN", "ASSIGNMENT") and i + 1 < len(children):
                rhs_node = children[i + 1]
            elif c.label == "expr" and rhs_node is None and var:
                rhs_node = c
        if rhs_node:
            rhs     = self.gen_expr(rhs_node)
            mangled = self._lookup_mangle(var)
            self.emit("=", rhs, "_", mangled)

    def gen_if(self, node: ParseNode):
        """Mirrors gen_if()."""
        cond  = None
        then_b = None
        else_b = None
        else_seen = False
        for c in node.children:
            if c.label == "bool_expr":
                cond = c
            elif c.label == "block":
                if not else_seen:
                    then_b = c
                else:
                    else_b = c
            elif c.label in ("ELSE", "else"):
                else_seen = True

        ltrue  = self.new_label()
        lfalse = self.new_label()
        lend   = self.new_label() if else_b else lfalse

        self.gen_bool(cond, ltrue, lfalse)
        self.emit_label(ltrue)
        self.gen_block(then_b)

        if else_b:
            self.emit("goto", "_", "_", lend)
            self.emit_label(lfalse)
            self.gen_block(else_b)
            self.emit_label(lend)
        else:
            self.emit_label(lfalse)

    def gen_while(self, node: ParseNode):
        """Mirrors gen_while()."""
        cond = None
        body = None
        for c in node.children:
            if c.label == "bool_expr":
                cond = c
            elif c.label == "block":
                body = c

        lbegin = self.new_label()
        lbody  = self.new_label()
        lend   = self.new_label()

        self.emit_label(lbegin)
        self.gen_bool(cond, lbody, lend)
        self.emit_label(lbody)
        self.gen_block(body)
        self.emit("goto", "_", "_", lbegin)
        self.emit_label(lend)

    def gen_print(self, node: ParseNode):
        """Mirrors gen_print()."""
        expr_node = None
        for c in node.children:
            if c.label in ("expr", "term", "factor",
                           "INTEGER", "FLOAT_LIT", "ID"):
                expr_node = c
                break
        v = self.gen_expr(expr_node)
        self.emit("param", v, "_", "_")
        self.emit("call",  "print", "1", "_")

    def gen_block(self, node: Optional[ParseNode]):
        """Mirrors gen_block() – increments scope for TAC, then recurses."""
        if node is None:
            return
        saved = self.sym.current_scope
        self.sym.current_scope += 1
        for c in node.children:
            if c.label == "stmt_list":
                self.gen_stmt_list(c)
            elif c.label == "block":
                self.gen_block(c)
            else:
                self.gen_stmt(c)
        self.sym.current_scope = saved

    def gen_stmt_list(self, node: Optional[ParseNode]):
        """Mirrors gen_stmt_list()."""
        if node is None:
            return
        for c in node.children:
            self.gen_stmt(c)

    def gen_stmt(self, node: Optional[ParseNode]):
        """Mirrors gen_stmt() – dispatches to the correct gen_* function."""
        if node is None:
            return
        label = node.label
        if label == "decl_stmt":
            self.gen_decl(node)
        elif label == "assign_stmt":
            self.gen_assign(node)
        elif label == "if_stmt":
            self.gen_if(node)
        elif label == "while_stmt":
            self.gen_while(node)
        elif label == "print_stmt":
            self.gen_print(node)
        elif label == "block":
            self.gen_block(node)
        elif label == "stmt_list":
            self.gen_stmt_list(node)
        elif label == "stmt":
            # wrapper node from parser.py
            for c in node.children:
                self.gen_stmt(c)
        else:
            for c in node.children:
                self.gen_stmt(c)

    def gen_program(self, root: ParseNode):
        """Mirrors gen_program() – entry point for full program TAC."""
        if root is None:
            return
        self.sym.current_scope = 0
        for c in root.children:
            if c.label == "stmt_list":
                self.gen_stmt_list(c)
            else:
                self.gen_stmt(c)

    # ── Per-statement generation (for step-by-step viewer) ───────────────────

    def gen_stmt_isolated(self, node: ParseNode
                          ) -> Tuple[int, int, int, int]:
        """
        Generate TAC for a single statement into the main quad list.
        Returns (quad_start, quad_end, temp_start, label_start).
        Mirrors the per-statement pass inside main() in phase7.c —
        counters reset to 0 before each statement so temps/labels
        always start at t1/L1 for every statement independently.
        """
        # mirrors:  temp_count = 0; label_count = 0;  inside the C loop
        self.temp_count  = 0
        self.label_count = 0
        q_start = len(self.quads)
        t_start = self.temp_count
        l_start = self.label_count
        self.gen_stmt(node)
        q_end = len(self.quads)
        return q_start, q_end, t_start, l_start


# ==============================================================================
#  PRINTING / OUTPUT HELPERS   (mirrors printing functions in phase7.c)
# ==============================================================================

def print_tac_line(q: Quad, out=sys.stdout):
    """Mirrors print_tac_line()."""
    if q.op == "label":
        print(f"{q.result}:", file=out)
    elif q.op == "goto":
        print(f"    goto {q.result}", file=out)
    elif q.op == "if":
        print(f"    if {q.arg1} goto {q.result}", file=out)
    elif q.op == "ifFalse":
        print(f"    ifFalse {q.arg1} goto {q.result}", file=out)
    elif q.op == "=":
        print(f"    {q.result} = {q.arg1}", file=out)
    elif q.op == "param":
        print(f"    param {q.arg1}", file=out)
    elif q.op == "call":
        print(f"    call {q.arg1}, {q.arg2}", file=out)
    else:
        print(f"    {q.result} = {q.arg1} {q.op} {q.arg2}", file=out)


def print_quads(quads: List[Quad], out=sys.stdout):
    """Mirrors print_quads()."""
    W = 80
    print("=" * W, file=out)
    print(" index | op       | arg1            | arg2    | result ", file=out)
    print("=" * W, file=out)
    for i, q in enumerate(quads):
        print(f" {i:4d}  | {q.op:<8} | {q.arg1:<15} | {q.arg2:<7} | {q.result:<15}",
              file=out)
    print("=" * W, file=out)


def save_space_separated(quads: List[Quad], filename: str):
    """Mirrors save_space_separated()."""
    with open(filename, "w") as f:
        for i, q in enumerate(quads):
            f.write(f"{i} {q.op} {q.arg1} {q.arg2} {q.result}\n")
    print(f"  TAC saved to {filename}")


# ==============================================================================
#  STATEMENT COLLECTION   (mirrors collect_tac_statements in phase7.c)
# ==============================================================================

def _has_initialization(node: ParseNode) -> bool:
    """Mirrors has_initialization() – True if decl_stmt has an initialiser."""
    if node.label != "decl_stmt":
        return False
    for c in node.children:
        if c.label in ("ASSIGN", "ASSIGNMENT"):
            return True
    return False


def _generates_tac(node: ParseNode) -> bool:
    """Mirrors generates_tac()."""
    if node.label in ("if_stmt", "while_stmt", "assign_stmt", "print_stmt"):
        return True
    if node.label == "decl_stmt":
        return _has_initialization(node)
    return False


def collect_tac_statements(root: ParseNode,
                            results: Optional[List[ParseNode]] = None
                            ) -> List[ParseNode]:
    """Mirrors collect_tac_statements()."""
    if results is None:
        results = []
    if _generates_tac(root):
        results.append(root)
    for c in root.children:
        collect_tac_statements(c, results)
    return results


# ==============================================================================
#  SOURCE STRING   (mirrors node_source_str)
# ==============================================================================

def node_source_str(node: ParseNode) -> str:
    """
    Gather all token values from the subtree (left-to-right DFS),
    mirroring node_source_str() in phase7.c.
    """
    parts = []
    def _gather(n: ParseNode):
        if n.token is not None:
            parts.append(n.token.value)
        for c in n.children:
            _gather(c)
    _gather(node)
    return " ".join(parts)


# ==============================================================================
#  STEP-BY-STEP EXPLANATION   (mirrors explain_step_by_step)
# ==============================================================================

def explain_step_by_step(node: ParseNode,
                          quads: List[Quad],
                          q_start: int, q_end: int,
                          t_start: int, l_start: int,
                          out=sys.stdout):
    """Mirrors explain_step_by_step() in phase7.c."""
    W = 80
    src = node_source_str(node)
    print("\n" + "=" * W, file=out)
    print("  STEP-BY-STEP TRANSLATION", file=out)
    print("=" * W, file=out)
    print(f"  Source statement: {src}\n", file=out)
    print(f"  Statement type  : {node.label}\n", file=out)

    # Collect temporaries and labels referenced in this statement's quads
    temps_found:  set = set()
    labels_found: set = set()

    for q in quads[q_start:q_end]:
        for field_val in (q.arg1, q.arg2, q.result):
            if field_val.startswith("t") and len(field_val) > 1 and \
                    field_val[1:].isdigit():
                temps_found.add(int(field_val[1:]))
            if field_val.startswith("L") and len(field_val) > 1 and \
                    field_val[1:].isdigit():
                labels_found.add(int(field_val[1:]))

    temp_str  = ", ".join(f"t{n}" for n in sorted(temps_found))  or "none"
    label_str = ", ".join(f"L{n}" for n in sorted(labels_found)) or "none"

    print(f"  Temporaries allocated : {temp_str}", file=out)
    print(f"  Labels allocated      : {label_str}", file=out)

    print(f"\n  Quadruples generated (indices {q_start} to {q_end - 1}):\n",
          file=out)
    print(f"  {'Index':>5} | {'op':<8} | {'arg1':<15} | {'arg2':<7} | {'result':<15}",
          file=out)
    print(f"  {'-----':>5}-+-{'--------':<8}-+-{'---------------':<15}-+-"
          f"{'-------':<7}-+-{'---------------':<15}", file=out)
    for i in range(q_start, q_end):
        q = quads[i]
        print(f"  {i:5d} | {q.op:<8} | {q.arg1:<15} | {q.arg2:<7} | {q.result:<15}",
              file=out)

    print("\nTAC for this statement:\n", file=out)
    for i in range(q_start, q_end):
        print("    ", end="", file=out)
        print_tac_line(quads[i], out)
    print("=" * W, file=out)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    W = 80

    if len(sys.argv) < 2:
        print("Usage: python phase7.py <source_file>")
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
        print("Lexical errors detected:")
        for e in lexer.errors:
            print(" ", e)
        sys.exit(1)

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("=" * W)
    print("  PHASE 7 - INTERMEDIATE CODE GENERATION (Three-Address Code / Quadruples)")
    print("=" * W)
    print(f"  Loaded {len(tokens)} tokens\n")

    parser = Parser(tokens)
    try:
        tree = parser.parse_program()
    except Exception as e:
        parser.errors.append(str(e))
        tree = None

    if parser.errors:
        print("  Syntax errors found - TAC generation aborted.")
        for i, err in enumerate(parser.errors, 1):
            print(f"  [{i}] Syntax Error: {err}")
        sys.exit(1)

    # ── Semantic analysis ─────────────────────────────────────────────────────
    analyser = SemanticAnalyser()
    analyser.analyse(tree)

    if analyser.sem_errors:
        print("\nTAC generation aborted due to semantic errors.")
        analyser.print_results()
        sys.exit(1)

    print("  Parse completed successfully - no syntax or semantic errors.\n")

    print("REPRESENTATION FORMAT: QUADRUPLES")
    print("  Each instruction is stored as a 4-tuple:")
    print("      ( op , arg1 , arg2 , result )\n")

    # ── Collect selectable statements ─────────────────────────────────────────
    selectable = collect_tac_statements(tree)

    # ── Per-statement pass (for step-by-step viewer) ──────────────────────────
    # We run each statement in isolation with fresh counters to record
    # which quads / temps / labels belong to each statement.
    stmt_info: List[Tuple[int, int, int, int]] = []  # (q_start, q_end, t_start, l_start)

    gen_per_stmt = TACGenerator(analyser.sym)

    for stmt in selectable:
        info = gen_per_stmt.gen_stmt_isolated(stmt)
        stmt_info.append(info)

    saved_quads = list(gen_per_stmt.quads)  # snapshot for step-by-step viewer

    # ── Full-program pass ─────────────────────────────────────────────────────
    gen_full = TACGenerator(analyser.sym)
    gen_full.gen_program(tree)

    print()
    print_quads(gen_full.quads)
    print(f"\n  Summary: {len(gen_full.quads)} quadruples,  "
          f"{gen_full.temp_count} temporaries,  {gen_full.label_count} labels.")

    save_space_separated(gen_full.quads, "Phase7_quadruples.txt")

    print()
    print("=" * W)
    print("  THREE-ADDRESS CODE (Full Program)")
    print("=" * W)
    print()
    for q in gen_full.quads:
        print_tac_line(q)
    print()
    print("=" * W)
    print(f"\n  Summary: {len(gen_full.quads)} quadruples,  "
          f"{gen_full.temp_count} temporaries,  {gen_full.label_count} labels.")

    # ── Step-by-step viewer ───────────────────────────────────────────────────
    if not selectable:
        print("\n  No statements that generate TAC found.")
        sys.exit(0)

    print("\n" + "=" * W)
    print("  STEP-BY-STEP TRANSLATION VIEWER")
    print("=" * W)
    print("  Select a statement to see its detailed translation:\n")

    for i, stmt in enumerate(selectable):
        src = node_source_str(stmt)
        if len(src) > 60:
            src = src[:57] + "..."
        print(f"  [{i + 1:2d}]  {stmt.label:<16}  line {_node_line(stmt):<4}  {src}")

    choice = 0
    while choice < 1 or choice > len(selectable):
        try:
            raw    = input(f"\n  Enter statement number (1-{len(selectable)}): ").strip()
            choice = int(raw)
            if choice < 1 or choice > len(selectable):
                print(f"  Invalid choice. Please enter a number between 1 and "
                      f"{len(selectable)}.")
        except ValueError:
            print("  Invalid input — enter a number.")

    idx = choice - 1
    q_start, q_end, t_start, l_start = stmt_info[idx]
    explain_step_by_step(
        selectable[idx],
        saved_quads,
        q_start, q_end,
        t_start, l_start,
    )


# ==============================================================================
#  Utility
# ==============================================================================

def _node_line(node: ParseNode) -> int:
    """Return the line number of the first token in the subtree."""
    if node.token is not None:
        return node.token.line
    for c in node.children:
        line = _node_line(c)
        if line:
            return line
    return 0


if __name__ == "__main__":
    main()