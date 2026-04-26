"""
================================================================================
  PHASE 8 - OPTIMIZATION AND TARGET CODE GENERATION
================================================================================

  Mirrors Phase8.c exactly:
    • Builds on Phase 7 infrastructure (lexer.py, parser.py, semantic.py)
    • Optimization 1 : Constant Folding
    • Optimization 2 : Copy Propagation
    • Target Code     : Pseudo-assembly (.DATA / .CODE sections)
    • Saves assembly  : output.asm
================================================================================
"""

import sys
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from lexer   import Lexer
from parser  import Parser, ParseNode
from semantic import SemanticAnalyser, SymbolTable, SymbolEntry


# ==============================================================================
#  QUADRUPLE  (same as Phase 7)
# ==============================================================================

@dataclass
class Quad:
    op:     str
    arg1:   str
    arg2:   str
    result: str


# ==============================================================================
#  TAC GENERATOR  (identical to phase7.py – included here so phase8.py is
#  self-contained and mirrors the monolithic Phase8.c structure)
# ==============================================================================

class TACGenerator:
    def __init__(self, sym: SymbolTable):
        self.sym         = sym
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
        self.quads.append(Quad(op, arg1 or "_", arg2 or "_", result or "_"))
        return len(self.quads) - 1

    def emit_label(self, lbl: str):
        self.emit("label", "_", "_", lbl)

    # ── Scope mangling ────────────────────────────────────────────────────────

    def _mangle(self, name: str, scope: int) -> str:
        return name if scope == 0 else f"{name}_s{scope}"

    def _lookup_mangle(self, name: str) -> str:
        s = self.sym.lookup(name)
        scope = s.scope if s else 0
        return self._mangle(name, scope)

    # ── Expression generation ─────────────────────────────────────────────────

    def gen_expr(self, node: Optional[ParseNode]) -> str:
        if node is None:
            return "_"
        label = node.label

        if label == "INTEGER":
            return node.token.value
        if label in ("FLOAT_LIT", "FLOAT"):
            return node.token.value
        if label == "ID":
            return self._lookup_mangle(node.token.value)

        if label == "factor":
            if len(node.children) == 1:
                return self.gen_expr(node.children[0])
            if len(node.children) == 3:
                return self.gen_expr(node.children[1])
            if len(node.children) == 2 and node.children[0].label == "MINUS":
                inner = self.gen_expr(node.children[1])
                t = self.new_temp()
                self.emit("-", "0", inner, t)
                return t
            return "_"

        if label in ("expr", "term") and len(node.children) == 3:
            left_node, op_node, right_node = node.children
            op = op_node.token.value if op_node.token else op_node.label
            l  = self.gen_expr(left_node)
            r  = self.gen_expr(right_node)
            t  = self.new_temp()
            self.emit(op, l, r, t)
            return t

        if node.children:
            return self.gen_expr(node.children[0])
        return "_"

    # ── Boolean generation ────────────────────────────────────────────────────

    def gen_bool(self, node: Optional[ParseNode], ltrue: str, lfalse: str):
        if node is None:
            return
        label = node.label

        # NOT – swap branches
        if label in ("bool_expr", "bool_factor") and node.children \
                and node.children[0].label == "NOT":
            self.gen_bool(node.children[1], lfalse, ltrue)
            return

        # OR / AND
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

        # Relational expression: expr  rel_op  expr
        if label == "rel_expr" and len(node.children) == 3:
            left_node, rel_node, right_node = node.children
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

        # Bare expression inside rel_expr (no relational op)
        if label == "rel_expr" and len(node.children) == 1:
            v = self.gen_expr(node.children[0])
            self.emit("if",   v, "_", ltrue)
            self.emit("goto", "_", "_", lfalse)
            return

        # Parenthesised bool_factor: ( bool_expr )
        if label == "bool_factor" and len(node.children) == 3:
            self.gen_bool(node.children[1], ltrue, lfalse)
            return

        # Mirrors Phase8.c gen_bool: handle bool_expr with embedded REL_OP
        # (Phase8.c parser puts REL_OP directly inside bool_expr)
        if label == "bool_expr":
            # find REL_OP child
            rel_idx = -1
            for i, c in enumerate(node.children):
                if c.label == "REL_OP":
                    rel_idx = i
                    break
            if rel_idx > 0 and rel_idx + 1 < len(node.children):
                l = self.gen_expr(node.children[rel_idx - 1])
                r = self.gen_expr(node.children[rel_idx + 1])
                op = node.children[rel_idx].token.value if node.children[rel_idx].token \
                     else node.children[rel_idx].label
                t = self.new_temp()
                self.emit(op, l, r, t)
                self.emit("if",   t, "_", ltrue)
                self.emit("goto", "_", "_", lfalse)
                return

        # Single-child wrapper
        if len(node.children) == 1:
            self.gen_bool(node.children[0], ltrue, lfalse)
            return

        # Fallback
        v = self.gen_expr(node)
        self.emit("if",   v, "_", ltrue)
        self.emit("goto", "_", "_", lfalse)

    # ── Statement generators ──────────────────────────────────────────────────

    def gen_decl(self, node: ParseNode):
        var  = ""
        init = None
        children = node.children
        for i, c in enumerate(children):
            if c.label == "ID":
                var = c.token.value
            elif c.label in ("ASSIGN", "ASSIGNMENT") and i + 1 < len(children):
                init = children[i + 1]
            elif c.label == "expr" and init is None and var:
                init = c
        if init:
            rhs     = self.gen_expr(init)
            mangled = self._lookup_mangle(var)
            self.emit("=", rhs, "_", mangled)

    def gen_assign(self, node: ParseNode):
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
        cond      = None
        then_b    = None
        else_b    = None
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
        if node is None:
            return
        for c in node.children:
            self.gen_stmt(c)

    def gen_stmt(self, node: Optional[ParseNode]):
        if node is None:
            return
        label = node.label
        if label == "decl_stmt":    self.gen_decl(node)
        elif label == "assign_stmt": self.gen_assign(node)
        elif label == "if_stmt":     self.gen_if(node)
        elif label == "while_stmt":  self.gen_while(node)
        elif label == "print_stmt":  self.gen_print(node)
        elif label == "block":       self.gen_block(node)
        elif label == "stmt_list":   self.gen_stmt_list(node)
        elif label == "stmt":
            for c in node.children:
                self.gen_stmt(c)
        else:
            for c in node.children:
                self.gen_stmt(c)

    def gen_program(self, root: ParseNode):
        if root is None:
            return
        self.sym.current_scope = 0
        for c in root.children:
            if c.label == "stmt_list":
                self.gen_stmt_list(c)
            else:
                self.gen_stmt(c)


# ==============================================================================
#  OPTIMIZATION 1 – CONSTANT FOLDING
#  Mirrors constant_folding_optimization() in Phase8.c
# ==============================================================================

def _is_number(s: str) -> bool:
    """True iff s is a numeric literal (int or float). Mirrors is_constant_number()."""
    if not s or s == "_":
        return False
    try:
        int(s)
        return True
    except ValueError:
        pass
    try:
        float(s)
        return True
    except ValueError:
        return False


def constant_folding(quads: List[Quad]) -> int:
    """
    Mirrors constant_folding_optimization() in Phase8.c.
    Modifies quads in-place.  Returns the number of folds applied.
    """
    folds = 0
    arith_ops = {"+", "-", "*", "/", "%"}

    for q in quads:
        if q.op not in arith_ops:
            continue
        if not (_is_number(q.arg1) and _is_number(q.arg2)):
            continue

        # Both args are numeric constants → fold
        try:
            v1 = int(q.arg1)
            v2 = int(q.arg2)
            is_float = False
        except ValueError:
            v1 = float(q.arg1)
            v2 = float(q.arg2)
            is_float = True

        if q.op == "+":
            result = v1 + v2
        elif q.op == "-":
            result = v1 - v2
        elif q.op == "*":
            result = v1 * v2
        elif q.op == "/":
            if v2 == 0:
                continue            # avoid div-by-zero
            result = v1 / v2 if is_float else v1 // v2
        elif q.op == "%":
            if v2 == 0:
                continue
            result = v1 % v2

        const_val = str(int(result)) if not is_float and result == int(result) \
                    else str(result)

        print(f"  Folding: {q.result} = {q.arg1} {q.op} {q.arg2}  →  "
              f"{q.result} = {const_val}")

        q.op   = "="
        q.arg1 = const_val
        q.arg2 = "_"
        folds += 1

    return folds


# ==============================================================================
#  OPTIMIZATION 2 – COPY PROPAGATION
#  Mirrors copy_propagation() in Phase8.c
# ==============================================================================

def _is_reassigned_later(quads: List[Quad], var: str, start: int) -> bool:
    """Mirrors is_reassigned_later() in Phase8.c."""
    for i in range(start + 1, len(quads)):
        if quads[i].result == var:
            return True
    return False


def _is_inside_loop(quads: List[Quad], idx: int) -> bool:
    """Return True if quad at idx is inside a loop (a goto jumps back past it)."""
    for j, q in enumerate(quads):
        if q.op == "goto" and q.result != "_":
            # find where this goto jumps
            target_label = q.result
            for k, qk in enumerate(quads):
                if qk.op == "label" and qk.result == target_label:
                    # if the jump goes backwards and idx is inside that range
                    if k < j and k <= idx <= j:
                        return True
    return False


def copy_propagation(quads: List[Quad]) -> int:
    """
    Mirrors copy_propagation() in Phase8.c.
    Modifies quads in-place.  Returns the number of replacements made.
    """
    props = 0

    for i, q in enumerate(quads):
        # Only consider simple copy assignments where the source is a variable
        if q.op != "=":
            continue
        if _is_number(q.arg1) or q.arg1 == "_":
            continue

        src = q.arg1
        dst = q.result

        # Skip if either variable is reassigned later (safety check)
        if _is_reassigned_later(quads, dst, i):
            continue
        if _is_reassigned_later(quads, src, i):
            continue

        # Skip if the assignment is inside a loop body — the src temp may hold
        # a stale value from a previous iteration when referenced after the loop.
        if _is_inside_loop(quads, i):
            continue

        print(f"  Safe copy: {dst} = {src}")

        # Propagate forward: replace uses of dst with src
        for j in range(i + 1, len(quads)):
            nxt = quads[j]
            if nxt.arg1 == dst:
                nxt.arg1 = src
                props += 1
            if nxt.arg2 == dst:
                nxt.arg2 = src
                props += 1
            # Stop if src is redefined
            if nxt.result == src:
                break

    return props


# ==============================================================================
#  TAC PRINTER (mirrors print_tac_line / printing in Phase8.c)
# ==============================================================================

def print_tac_line(q: Quad, out=sys.stdout):
    """Mirrors print_tac_line() in Phase8.c."""
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


def print_tac_numbered(quads: List[Quad], out=sys.stdout):
    """Mirrors the numbered TAC dump in Phase8.c main()."""
    for i, q in enumerate(quads):
        print(f"  {i:4d}: ", end="", file=out)
        if q.op == "label":
            print(f"{q.result}:", file=out)
        elif q.op == "=":
            print(f"{q.result} = {q.arg1}", file=out)
        elif q.op == "param":
            print(f"param {q.arg1}", file=out)
        elif q.op == "call":
            print(f"call {q.arg1}, {q.arg2}", file=out)
        else:
            print(f"{q.result} = {q.arg1} {q.op} {q.arg2}", file=out)


# ==============================================================================
#  TARGET CODE GENERATION  (mirrors generate_target_code() in Phase8.c)
# ==============================================================================

def generate_target_code(quads: List[Quad], sym: SymbolTable,
                          temp_count: int, out=sys.stdout):
    """
    Generates pseudo-assembly code mirroring generate_target_code() in Phase8.c.

    Sections:
      .DATA  – one entry per symbol-table variable and temporary
      .CODE  – one pseudo-assembly instruction per quadruple
    """

    # ── .DATA section ─────────────────────────────────────────────────────────
    print(".DATA", file=out)

    # Build a map of variable → initial constant value from the quads.
    # A quad  (op="=", arg1=<number>, result=<var>)  is an initialisation.
    # We take only the FIRST such assignment per variable (the declaration init).
    init_vals: dict = {}
    for q in quads:
        if q.op == "=" and _is_number(q.arg1) and q.result not in init_vals:
            init_vals[q.result] = q.arg1

    # Emit all symbol-table entries (mirrors the sym_table loop in Phase8.c)
    emitted_vars = set()
    for entry in sym.sym_table:                 # SymbolEntry objects
        vtype = getattr(entry, "type", "int")
        init  = init_vals.get(entry.name)
        if vtype == "float":
            val = init if init is not None else "0.0"
            print(f"    {entry.name} .FLOAT {val}", file=out)
        else:
            val = init if init is not None else "0"
            print(f"    {entry.name} .WORD {val}", file=out)
        emitted_vars.add(entry.name)

    # Emit temporaries  t1 … t<temp_count>
    for i in range(1, temp_count + 1):
        tname = f"t{i}"
        init  = init_vals.get(tname)
        val   = init if init is not None else "0"
        print(f"    {tname} .WORD {val}", file=out)

    # ── .CODE section ─────────────────────────────────────────────────────────
    print("\n.CODE", file=out)
    print("main:\n", file=out)

    for q in quads:

        # ── label ──────────────────────────────────────────────────────────
        if q.op == "label":
            # Emit all labels exactly as generated (no renaming)
            print(f"{q.result}:", file=out)

        # ── simple copy / load constant ────────────────────────────────────
        elif q.op == "=":
            if _is_number(q.arg1):
                print(f"    MOV {q.result}, #{q.arg1}", file=out)
            else:
                print(f"    MOV R0, {q.arg1}", file=out)
                print(f"    MOV {q.result}, R0", file=out)

        # ── arithmetic ─────────────────────────────────────────────────────
        elif q.op == "+":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    ADD R0, {a2}", file=out)
            print(f"    MOV {q.result}, R0", file=out)

        elif q.op == "-":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    SUB R0, {a2}", file=out)
            print(f"    MOV {q.result}, R0", file=out)

        elif q.op == "*":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    MUL R0, {a2}", file=out)
            print(f"    MOV {q.result}, R0", file=out)

        elif q.op == "/":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    DIV R0, {a2}", file=out)
            print(f"    MOV {q.result}, R0", file=out)

        elif q.op == "%":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    MOD R0, {a2}", file=out)
            print(f"    MOV {q.result}, R0", file=out)

        # ── relational operators ────────────────────────────────────────────
        elif q.op == ">":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    CMP R0, {a2}", file=out)
            print(f"    JG  {q.result}_true", file=out)
            print(f"    MOV {q.result}, #0", file=out)
            print(f"    JMP {q.result}_end", file=out)
            print(f"{q.result}_true:", file=out)
            print(f"    MOV {q.result}, #1", file=out)
            print(f"{q.result}_end:", file=out)

        elif q.op == "<":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    CMP R0, {a2}", file=out)
            print(f"    JL  {q.result}_true", file=out)
            print(f"    MOV {q.result}, #0", file=out)
            print(f"    JMP {q.result}_end", file=out)
            print(f"{q.result}_true:", file=out)
            print(f"    MOV {q.result}, #1", file=out)
            print(f"{q.result}_end:", file=out)

        elif q.op == ">=":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    CMP R0, {a2}", file=out)
            print(f"    JGE {q.result}_true", file=out)
            print(f"    MOV {q.result}, #0", file=out)
            print(f"    JMP {q.result}_end", file=out)
            print(f"{q.result}_true:", file=out)
            print(f"    MOV {q.result}, #1", file=out)
            print(f"{q.result}_end:", file=out)

        elif q.op == "<=":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    CMP R0, {a2}", file=out)
            print(f"    JLE {q.result}_true", file=out)
            print(f"    MOV {q.result}, #0", file=out)
            print(f"    JMP {q.result}_end", file=out)
            print(f"{q.result}_true:", file=out)
            print(f"    MOV {q.result}, #1", file=out)
            print(f"{q.result}_end:", file=out)

        elif q.op == "==":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    CMP R0, {a2}", file=out)
            print(f"    JE  {q.result}_true", file=out)
            print(f"    MOV {q.result}, #0", file=out)
            print(f"    JMP {q.result}_end", file=out)
            print(f"{q.result}_true:", file=out)
            print(f"    MOV {q.result}, #1", file=out)
            print(f"{q.result}_end:", file=out)

        elif q.op == "!=":
            a1 = f"#{q.arg1}" if _is_number(q.arg1) else q.arg1
            a2 = f"#{q.arg2}" if _is_number(q.arg2) else q.arg2
            print(f"    MOV R0, {a1}", file=out)
            print(f"    CMP R0, {a2}", file=out)
            print(f"    JNE {q.result}_true", file=out)
            print(f"    MOV {q.result}, #0", file=out)
            print(f"    JMP {q.result}_end", file=out)
            print(f"{q.result}_true:", file=out)
            print(f"    MOV {q.result}, #1", file=out)
            print(f"{q.result}_end:", file=out)

        # ── conditional / unconditional jump ───────────────────────────────
        elif q.op == "if":
            print(f"    MOV R0, {q.arg1}", file=out)
            print(f"    CMP R0, #1", file=out)
            print(f"    JE  {q.result}", file=out)

        elif q.op == "goto":
            print(f"    JMP {q.result}", file=out)

        # ── function call / param ──────────────────────────────────────────
        elif q.op == "param":
            print(f"    PUSH {q.arg1}", file=out)

        elif q.op == "call":
            print(f"    CALL {q.arg1}", file=out)

    print("\n    HALT", file=out)
    print("\nprint:", file=out)
    print("    RET", file=out)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    W = 80

    if len(sys.argv) < 2:
        print("Usage: python phase8.py <source_file>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)

    print("=" * W)
    print("  FINAL PHASE : OPTIMIZATION AND TARGET CODE GENERATION")
    print("=" * W)

    # ── Lex ───────────────────────────────────────────────────────────────────
    lexer  = Lexer(source)
    tokens = lexer.tokenize()
    print(f"\n  Loaded {len(tokens)} tokens\n")

    if lexer.errors:
        print("Lexical errors detected:")
        for e in lexer.errors:
            print(" ", e)
        sys.exit(1)

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("  Parsing program...")
    parser = Parser(tokens)
    try:
        tree = parser.parse_program()
    except Exception as e:
        parser.errors.append(str(e))
        tree = None

    if parser.errors:
        print("\n  Syntax errors found")
        for i, err in enumerate(parser.errors, 1):
            print(f"  [{i}] Syntax Error: {err}")
        sys.exit(1)

    # ── Semantic analysis ─────────────────────────────────────────────────────
    print("  Performing semantic analysis...")
    analyser = SemanticAnalyser()
    analyser.analyse(tree)

    if analyser.sem_errors:
        print("\n  Semantic errors found")
        analyser.print_results()
        sys.exit(1)

    print("  Parse completed successfully\n")

    # ── TAC generation ────────────────────────────────────────────────────────
    print("  Generating Three-Address Code...")
    gen = TACGenerator(analyser.sym)
    gen.gen_program(tree)
    print(f"  Generated {len(gen.quads)} quadruples")

    # Keep a pristine copy for the "before" display
    original_quads = deepcopy(gen.quads)

    # ── Print original TAC ────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  THREE-ADDRESS CODE")
    print("=" * W)
    print()
    print_tac_numbered(original_quads)

    # ── Apply optimizations ───────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  APPLYING OPTIMIZATIONS")
    print("=" * W)

    print("\n  [1] Constant Folding:")
    cf_count = constant_folding(gen.quads)
    print(f"      {cf_count} operations folded")

    print("\n  [2] Copy Propagation:")
    cp_count = copy_propagation(gen.quads)
    print(f"      {cp_count} replacements made")

    # ── Print optimized TAC ───────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  OPTIMIZED THREE-ADDRESS CODE")
    print("=" * W)
    print()
    print_tac_numbered(gen.quads)

    print()
    print("=" * W)
    print("  THREE-ADDRESS CODE (readable)")
    print("=" * W)
    print()
    for q in gen.quads:
        print_tac_line(q)

    # ── Generate target code ──────────────────────────────────────────────────
    asm_filename = "output.asm"
    try:
        with open(asm_filename, "w") as asm_file:
            generate_target_code(gen.quads, analyser.sym, gen.temp_count, asm_file)
        print(f"\n  Target assembly code written to {asm_filename}")
    except OSError as e:
        print(f"\n  Warning: could not write {asm_filename}: {e}")

    print()
    print("=" * W)
    print("  TARGET CODE (Pseudo-Assembly)")
    print("=" * W)
    print()
    generate_target_code(gen.quads, analyser.sym, gen.temp_count)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  OPTIMIZATION SUMMARY")
    print("=" * W)
    print(f"  Constant Folding:    {cf_count:2d} operations")
    print(f"  Copy Propagation:    {cp_count:2d} replacements")
    print(f"  Total Optimizations: {cf_count + cp_count:2d}")
    print("=" * W)


if __name__ == "__main__":
    main()