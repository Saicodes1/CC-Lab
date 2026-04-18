"""
Recursive-Descent Parser  +  LL(1) Parser  +  Shift-Reduce Parser
Produces a full Concrete Parse Tree (not an AST) for the language
defined by the CFG below.

CFG (Context-Free Grammar)
===========================
program     -> stmt_list EOF
stmt_list   -> stmt stmt_list | ε
stmt        -> decl_stmt | assign_stmt | if_stmt | while_stmt | print_stmt
decl_stmt   -> TYPE ID ; | TYPE ID = expr ;
assign_stmt -> ID = expr ;
if_stmt     -> if ( bool_expr ) block else block
            | if ( bool_expr ) block
while_stmt  -> while ( bool_expr ) block
print_stmt  -> print ( expr ) ;
block       -> { stmt_list }

bool_expr   -> bool_expr || bool_term | bool_term
bool_term   -> bool_term && bool_factor | bool_factor
bool_factor -> ! bool_factor | ( bool_expr ) | rel_expr
rel_expr    -> expr rel_op expr
rel_op      -> < | > | <= | >= | == | !=

expr        -> expr + term | expr - term | term
term        -> term * factor | term / factor | term % factor | factor
factor      -> ( expr ) | INTEGER | FLOAT | ID | - factor
TYPE        -> int | float

Error Recovery — Panic Mode
============================
Recovery happens at the statement level inside parse_stmt().
On a ParseError the parser:
  1. Records the error message.
  2. Calls _synchronise() to skip ahead to a safe restart token.
  3. Resumes parse_stmt_list() so every error is collected.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional

from lexer import Lexer, Token


# ─────────────────────────────────────────────
#  Parse-Tree Node
# ─────────────────────────────────────────────

@dataclass
class ParseNode:
    label: str
    token: Optional[Token] = None
    children: List["ParseNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return self.token is not None

    def add(self, child: "ParseNode") -> "ParseNode":
        self.children.append(child)
        return child

    def pretty(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└── " if is_last else "├── "
        if self.is_leaf():
            display = f"{self.label}({self.token.value!r})"
        else:
            display = self.label
        lines = [prefix + connector + display]
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            lines.append(child.pretty(child_prefix, i == len(self.children) - 1))
        return "\n".join(lines)

    def __str__(self):
        return self.pretty("", True)


# ─────────────────────────────────────────────
#  Parser
# ─────────────────────────────────────────────

class ParseError(Exception):
    pass


_STMT_START_TOKENS = {
    "INT", "FLOAT",
    "IDENTIFIER",
    "IF",
    "WHILE",
    "PRINT",
    "LBRACE",
}

_EXPR_START_TOKENS = {
    "INT_LITERAL", "FLOAT_LITERAL", "IDENTIFIER", "LPAREN", "MINUS",
}

_BOOL_FACTOR_START_TOKENS = _EXPR_START_TOKENS | {"NOT"}

MAX_LOOKAHEAD = 512


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos    = 0
        self.errors: List[str] = []

    # ── Token accessors ───────────────────────

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        return self.tokens[min(idx, len(self.tokens) - 1)]

    # ── Error location ────────────────────────

    def _error_location(self) -> tuple:
        cur = self.current
        if self.pos > 0:
            prev = self.tokens[self.pos - 1]
            if prev.line < cur.line:
                return prev.line, prev.col + len(str(prev.value))
        return cur.line, cur.col

    # ── Block skipper / scanner ────────────────

    def _skip_block(self):
        if self.current.type != "LBRACE":
            return
        depth = 0
        while self.current.type != "EOF":
            if self.current.type == "LBRACE":
                depth += 1
            elif self.current.type == "RBRACE":
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    return
            self.pos += 1

    def _scan_block_for_errors(self):
        if self.current.type != "LBRACE":
            return
        self.pos += 1
        self.parse_stmt_list()
        if self.current.type == "RBRACE":
            self.pos += 1
        elif self.current.type != "EOF":
            line, col = self._error_location()
            self.errors.append(
                f"Line {line}, col {col}: "
                f"expected 'RBRACE' to close block but got "
                f"{self.current.type!r} ({self.current.value!r})"
            )

    # ── Panic-mode recovery ───────────────────

    def _is_real_stmt_start(self) -> bool:
        t   = self.current.type
        nxt = self.peek(1).type
        if t == "IDENTIFIER":
            return nxt in ("ASSIGN", "SEMICOLON")
        if t in ("IF", "WHILE", "PRINT"):
            return nxt == "LPAREN"
        if t in ("INT", "FLOAT"):
            return nxt == "IDENTIFIER"
        if t == "LBRACE":
            return True
        return False

    def _synchronise(self):
        while True:
            t = self.current.type
            if t == "EOF":
                return
            if t == "RBRACE":
                return
            if t == "SEMICOLON":
                self.pos += 1
                if self.current.type in ("EOF", "RBRACE"):
                    return
                if self.current.type in _STMT_START_TOKENS and \
                        self._is_real_stmt_start():
                    return
                continue
            if t == "LBRACE":
                self._scan_block_for_errors()
                if self.current.type in ("EOF", "RBRACE"):
                    return
                if self.current.type in _STMT_START_TOKENS and \
                        self._is_real_stmt_start():
                    return
                continue
            if t in _STMT_START_TOKENS and self._is_real_stmt_start():
                return
            self.pos += 1

    # ── Hang-prevention guards ────────────────

    def _assert_expr_start(self):
        tok = self.current
        if tok.type not in _EXPR_START_TOKENS:
            line, col = self._error_location()
            raise ParseError(
                f"Line {line}, col {col}: "
                f"unexpected token in expression: {tok.type!r} ({tok.value!r})"
            )

    def _assert_bool_factor_start(self):
        tok = self.current
        if tok.type not in _BOOL_FACTOR_START_TOKENS:
            line, col = self._error_location()
            raise ParseError(
                f"Line {line}, col {col}: "
                f"unexpected token in condition: {tok.type!r} ({tok.value!r})"
            )

    # ── Token helpers ─────────────────────────

    def consume(self, expected_type: str) -> Token:
        tok = self.current
        if tok.type != expected_type:
            line, col = self._error_location()
            raise ParseError(
                f"Line {line}, col {col}: "
                f"expected {expected_type!r} but got {tok.type!r} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    def match(self, *types: str) -> bool:
        return self.current.type in types

    def leaf(self, token: Token) -> ParseNode:
        return ParseNode(label=token.type, token=token)

    # ══════════════════════════════════════════
    #  Grammar rules
    # ══════════════════════════════════════════

    def parse_program(self) -> ParseNode:
        node = ParseNode("program")
        node.add(self.parse_stmt_list())
        while not self.match("EOF"):
            tok = self.current
            line, col = self._error_location()
            self.errors.append(
                f"Line {line}, col {col}: "
                f"unexpected token {tok.type!r} ({tok.value!r}) at top level"
            )
            self.pos += 1
            if self.current.type in _STMT_START_TOKENS:
                node.add(self.parse_stmt_list())
        node.add(self.leaf(self.consume("EOF")))
        return node

    def parse_stmt_list(self) -> ParseNode:
        node = ParseNode("stmt_list")
        while not self.match("EOF", "RBRACE"):
            node.add(self.parse_stmt())
        if not node.children:
            node.add(ParseNode("ε"))
        return node

    def parse_stmt(self) -> ParseNode:
        node = ParseNode("stmt")
        tok  = self.current
        try:
            if tok.type in ("INT", "FLOAT"):
                node.add(self.parse_decl_stmt())
            elif tok.type == "IDENTIFIER":
                node.add(self.parse_assign_stmt())
            elif tok.type == "IF":
                node.add(self.parse_if_stmt())
            elif tok.type == "WHILE":
                node.add(self.parse_while_stmt())
            elif tok.type == "PRINT":
                node.add(self.parse_print_stmt())
            elif tok.type == "LBRACE":
                node.add(self.parse_block())
            else:
                line, col = self._error_location()
                raise ParseError(
                    f"Line {line}, col {col}: "
                    f"unexpected token {tok.type!r} ({tok.value!r})"
                )
        except ParseError as err:
            self.errors.append(str(err))
            self._synchronise()
            node.add(ParseNode("ERROR"))

        return node

    def parse_decl_stmt(self) -> ParseNode:
        node = ParseNode("decl_stmt")
        type_node = ParseNode("TYPE")
        type_node.add(self.leaf(self.consume(self.current.type)))
        node.add(type_node)
        if not self.match("IDENTIFIER"):
            bad = self.current
            line, col = bad.line, bad.col
            while not self.match("SEMICOLON", "EOF", "RBRACE"):
                self.pos += 1
            if self.match("SEMICOLON"):
                self.pos += 1
            raise ParseError(
                f"Line {line}, col {col}: "
                f"expected 'IDENTIFIER' but got "
                f"{bad.type!r} ({bad.value!r})"
            )
        node.add(ParseNode("ID", token=self.consume("IDENTIFIER")))
        if self.match("ASSIGN"):
            node.add(self.leaf(self.consume("ASSIGN")))
            node.add(self.parse_expr())
        node.add(self.leaf(self.consume("SEMICOLON")))
        return node

    def parse_assign_stmt(self) -> ParseNode:
        node = ParseNode("assign_stmt")
        node.add(ParseNode("ID", token=self.consume("IDENTIFIER")))
        node.add(self.leaf(self.consume("ASSIGN")))
        node.add(self.parse_expr())
        node.add(self.leaf(self.consume("SEMICOLON")))
        return node

    def parse_if_stmt(self) -> ParseNode:
        node = ParseNode("if_stmt")
        node.add(self.leaf(self.consume("IF")))
        node.add(self.leaf(self.consume("LPAREN")))
        node.add(self.parse_bool_expr())
        node.add(self.leaf(self.consume("RPAREN")))
        node.add(self.parse_block())
        if self.match("ELSE"):
            node.add(self.leaf(self.consume("ELSE")))
            node.add(self.parse_block())
        return node

    def parse_while_stmt(self) -> ParseNode:
        node = ParseNode("while_stmt")
        node.add(self.leaf(self.consume("WHILE")))
        node.add(self.leaf(self.consume("LPAREN")))
        node.add(self.parse_bool_expr())
        node.add(self.leaf(self.consume("RPAREN")))
        node.add(self.parse_block())
        return node

    def parse_print_stmt(self) -> ParseNode:
        node = ParseNode("print_stmt")
        node.add(self.leaf(self.consume("PRINT")))
        node.add(self.leaf(self.consume("LPAREN")))
        node.add(self.parse_expr())
        node.add(self.leaf(self.consume("RPAREN")))
        if not self.match("SEMICOLON"):
            prev = self.tokens[self.pos - 1]
            line, col = prev.line, prev.col + len(str(prev.value))
            raise ParseError(
                f"Line {line}, col {col}: "
                f"expected 'SEMICOLON' after print statement but got "
                f"{self.current.type!r} ({self.current.value!r})"
            )
        node.add(self.leaf(self.consume("SEMICOLON")))
        return node

    def parse_block(self) -> ParseNode:
        node = ParseNode("block")
        node.add(self.leaf(self.consume("LBRACE")))
        node.add(self.parse_stmt_list())
        if self.current.type != "RBRACE":
            line, col = self._error_location()
            raise ParseError(
                f"Line {line}, col {col}: "
                f"expected 'RBRACE' but got {self.current.type!r} "
                f"({self.current.value!r})"
            )
        node.add(self.leaf(self.consume("RBRACE")))
        return node

    # ── Boolean expressions ────────────────────

    def parse_bool_expr(self) -> ParseNode:
        self._assert_bool_factor_start()
        left = ParseNode("bool_expr")
        left.add(self.parse_bool_term())
        while self.match("OR"):
            op = self.leaf(self.consume("OR"))
            self._assert_bool_factor_start()
            right = self.parse_bool_term()
            parent = ParseNode("bool_expr")
            parent.add(left); parent.add(op); parent.add(right)
            left = parent
        return left

    def parse_bool_term(self) -> ParseNode:
        self._assert_bool_factor_start()
        left = ParseNode("bool_term")
        left.add(self.parse_bool_factor())
        while self.match("AND"):
            op = self.leaf(self.consume("AND"))
            self._assert_bool_factor_start()
            right = self.parse_bool_factor()
            parent = ParseNode("bool_term")
            parent.add(left); parent.add(op); parent.add(right)
            left = parent
        return left

    def parse_bool_factor(self) -> ParseNode:
        node = ParseNode("bool_factor")
        if self.match("NOT"):
            node.add(self.leaf(self.consume("NOT")))
            self._assert_bool_factor_start()
            node.add(self.parse_bool_factor())
        elif self.match("LPAREN") and self._lookahead_is_bool():
            node.add(self.leaf(self.consume("LPAREN")))
            node.add(self.parse_bool_expr())
            node.add(self.leaf(self.consume("RPAREN")))
        else:
            node.add(self.parse_rel_expr())
        return node

    def _lookahead_is_bool(self) -> bool:
        depth = 0
        i     = self.pos
        limit = min(self.pos + MAX_LOOKAHEAD, len(self.tokens))
        while i < limit:
            t = self.tokens[i].type
            if t in ("LBRACE", "EOF"):
                return False
            if t == "LPAREN":
                depth += 1
            elif t == "RPAREN":
                depth -= 1
                if depth == 0:
                    nxt = self.tokens[i + 1].type if i + 1 < len(self.tokens) else "EOF"
                    return nxt in ("AND", "OR", "RPAREN", "EOF", "LBRACE")
            elif depth == 1 and t in ("LT", "GT", "LEQ", "GEQ", "EQ", "NEQ",
                                      "OR", "AND", "NOT"):
                return True
            i += 1
        return False

    def parse_rel_expr(self) -> ParseNode:
        node = ParseNode("rel_expr")
        self._assert_expr_start()
        node.add(self.parse_expr())
        # rel_op is OPTIONAL: bare expressions like  if (b)  pass the parser
        # and are caught by the semantic analyser instead (invalid bool condition)
        if self.current.type in ("LT", "GT", "LEQ", "GEQ", "EQ", "NEQ"):
            node.add(self.parse_rel_op())
            self._assert_expr_start()
            node.add(self.parse_expr())
        return node

    def parse_rel_op(self) -> ParseNode:
        node = ParseNode("rel_op")
        tok  = self.current
        if tok.type in ("LT", "GT", "LEQ", "GEQ", "EQ", "NEQ"):
            node.add(self.leaf(self.consume(tok.type)))
        else:
            line, col = self._error_location()
            raise ParseError(
                f"Line {line}, col {col}: "
                f"expected relational operator, got {tok.type!r}"
            )
        return node

    # ── Arithmetic expressions ─────────────────

    def parse_expr(self) -> ParseNode:
        self._assert_expr_start()
        left = ParseNode("expr")
        left.add(self.parse_term())
        while self.match("PLUS", "MINUS"):
            op = self.leaf(self.consume(self.current.type))
            self._assert_expr_start()
            right = self.parse_term()
            parent = ParseNode("expr")
            parent.add(left); parent.add(op); parent.add(right)
            left = parent
        return left

    def parse_term(self) -> ParseNode:
        self._assert_expr_start()
        left = ParseNode("term")
        left.add(self.parse_factor())
        while self.match("STAR", "SLASH", "PERCENT"):
            op = self.leaf(self.consume(self.current.type))
            self._assert_expr_start()
            right = self.parse_factor()
            parent = ParseNode("term")
            parent.add(left); parent.add(op); parent.add(right)
            left = parent
        return left

    def parse_factor(self) -> ParseNode:
        node = ParseNode("factor")
        tok  = self.current
        if tok.type == "LPAREN":
            node.add(self.leaf(self.consume("LPAREN")))
            self._assert_expr_start()
            node.add(self.parse_expr())
            node.add(self.leaf(self.consume("RPAREN")))
        elif tok.type == "INT_LITERAL":
            node.add(ParseNode("INTEGER", token=self.consume("INT_LITERAL")))
        elif tok.type == "FLOAT_LITERAL":
            node.add(ParseNode("FLOAT_LIT", token=self.consume("FLOAT_LITERAL")))
        elif tok.type == "IDENTIFIER":
            node.add(ParseNode("ID", token=self.consume("IDENTIFIER")))
        elif tok.type == "MINUS":
            node.add(self.leaf(self.consume("MINUS")))
            self._assert_expr_start()
            node.add(self.parse_factor())
        else:
            line, col = self._error_location()
            raise ParseError(
                f"Line {line}, col {col}: "
                f"unexpected token in expression: {tok.type!r} ({tok.value!r})"
            )
        return node


# ─────────────────────────────────────────────
#  Token -> Grammar Terminal mapping
#  (mirrors token_to_terminal() in C)
# ─────────────────────────────────────────────

def token_to_terminal(tok: Token) -> str:
    """Map a lexer Token to the grammar terminal symbol used by LL(1)/SR."""
    if tok is None:
        return "$"
    t, v = tok.type, tok.value
    if t in ("INT", "FLOAT"):
        return "TYPE"
    if t == "IF":
        return "IF"
    if t == "ELSE":
        return "ELSE"
    if t == "WHILE":
        return "WHILE"
    if t == "PRINT":
        return "PRINT"
    if t == "IDENTIFIER":
        return "ID"
    if t == "INT_LITERAL":
        return "INTEGER"
    if t == "FLOAT_LITERAL":
        return "FLOAT"
    if t == "ASSIGN":
        return "ASSIGNMENT"
    if t in ("LT", "GT", "LEQ", "GEQ", "EQ", "NEQ"):
        return "REL_OP"
    if t == "PLUS":
        return "PLUS"
    if t == "MINUS":
        return "MINUS"
    if t == "STAR":
        return "MULT"
    if t == "SLASH":
        return "DIV"
    if t == "PERCENT":
        return "MOD"
    if t == "NOT":
        return "NOT"
    if t == "AND":
        return "LOGICAL_AND"
    if t == "OR":
        return "LOGICAL_OR"
    if t == "LPAREN":
        return "LPAREN"
    if t == "RPAREN":
        return "RPAREN"
    if t == "SEMICOLON":
        return "SEMI"
    if t == "LBRACE":
        return "LBRACE"
    if t == "RBRACE":
        return "RBRACE"
    return t


# ─────────────────────────────────────────────
#  Grammar / FIRST / FOLLOW / LL(1) table
# ─────────────────────────────────────────────

EPS = "ε"

PRODUCTIONS = [
    ( 1, "program",         ["stmt_list"]),
    ( 2, "stmt_list",       ["stmt", "stmt_list"]),
    ( 3, "stmt_list",       [EPS]),
    ( 4, "stmt",            ["decl_stmt"]),
    ( 5, "stmt",            ["assign_stmt"]),
    ( 6, "stmt",            ["if_stmt"]),
    ( 7, "stmt",            ["while_stmt"]),
    ( 8, "stmt",            ["print_stmt"]),
    ( 9, "stmt",            ["block"]),
    (10, "decl_stmt",       ["TYPE", "ID", "decl_prime"]),
    (11, "decl_prime",      ["ASSIGNMENT", "arith_expr", "SEMI"]),
    (12, "decl_prime",      ["SEMI"]),
    (13, "assign_stmt",     ["ID", "ASSIGNMENT", "arith_expr", "SEMI"]),
    (14, "if_stmt",         ["IF", "LPAREN", "bool_expr", "RPAREN", "block", "else_prime"]),
    (15, "else_prime",      ["ELSE", "block"]),
    (16, "else_prime",      [EPS]),
    (17, "while_stmt",      ["WHILE", "LPAREN", "bool_expr", "RPAREN", "block"]),
    (18, "print_stmt",      ["PRINT", "LPAREN", "arith_expr", "RPAREN", "SEMI"]),
    (19, "block",           ["LBRACE", "stmt_list", "RBRACE"]),
    (20, "arith_expr",      ["term", "arith_expr_tail"]),
    (21, "arith_expr_tail", ["PLUS",  "term", "arith_expr_tail"]),
    (22, "arith_expr_tail", ["MINUS", "term", "arith_expr_tail"]),
    (23, "arith_expr_tail", [EPS]),
    (24, "term",            ["factor", "term_tail"]),
    (25, "term_tail",       ["MULT",  "factor", "term_tail"]),
    (26, "term_tail",       ["DIV",   "factor", "term_tail"]),
    (27, "term_tail",       ["MOD",   "factor", "term_tail"]),
    (28, "term_tail",       [EPS]),
    (29, "factor",          ["LPAREN", "arith_expr", "RPAREN"]),
    (30, "factor",          ["ID"]),
    (31, "factor",          ["INTEGER"]),
    (32, "factor",          ["FLOAT"]),
    (33, "bool_expr",       ["bool_term", "bool_expr_tail"]),
    (34, "bool_expr_tail",  ["LOGICAL_OR", "bool_term", "bool_expr_tail"]),
    (35, "bool_expr_tail",  [EPS]),
    (36, "bool_term",       ["bool_factor", "bool_term_tail"]),
    (37, "bool_term_tail",  ["LOGICAL_AND", "bool_factor", "bool_term_tail"]),
    (38, "bool_term_tail",  [EPS]),
    (39, "bool_factor",     ["arith_expr", "rel_op", "arith_expr"]),
    (40, "bool_factor",     ["LPAREN", "bool_expr", "RPAREN"]),
    (41, "bool_factor",     ["NOT", "bool_factor"]),
    (42, "rel_op",          ["REL_OP"]),
]

TERMINALS = [
    "TYPE", "ID", "SEMI", "ASSIGNMENT", "IF", "LPAREN", "RPAREN",
    "ELSE", "WHILE", "PRINT", "LBRACE", "RBRACE",
    "LOGICAL_OR", "LOGICAL_AND", "NOT", "REL_OP",
    "PLUS", "MINUS", "MULT", "DIV", "MOD",
    "INTEGER", "FLOAT", "$",
]

_TERMINAL_SET = set(TERMINALS) | {EPS}


def _is_terminal(s: str) -> bool:
    return s in _TERMINAL_SET


def _build_symbol_lists():
    nts = []
    seen = set()
    for _, lhs, _ in PRODUCTIONS:
        if lhs not in seen:
            nts.append(lhs)
            seen.add(lhs)
    return nts


NON_TERMINALS = _build_symbol_lists()
_NT_INDEX  = {s: i for i, s in enumerate(NON_TERMINALS)}
_TER_INDEX = {s: i for i, s in enumerate(TERMINALS)}


def _compute_first_sets():
    first = {nt: set() for nt in NON_TERMINALS}
    changed = True
    while changed:
        changed = False
        for _, lhs, rhs in PRODUCTIONS:
            before = len(first[lhs])
            result = _first_of_sequence(rhs, first)
            first[lhs] |= result
            if len(first[lhs]) != before:
                changed = True
    return first


def _first_of_sequence(seq, first_sets):
    result = set()
    all_eps = True
    for sym in seq:
        if sym == EPS:
            continue
        if _is_terminal(sym):
            result.add(sym)
            all_eps = False
            break
        else:
            fs = first_sets.get(sym, set())
            result |= (fs - {EPS})
            if EPS not in fs:
                all_eps = False
                break
    if all_eps:
        result.add(EPS)
    return result


def _compute_follow_sets(first_sets):
    follow = {nt: set() for nt in NON_TERMINALS}
    follow["program"].add("$")
    changed = True
    while changed:
        changed = False
        for _, lhs, rhs in PRODUCTIONS:
            for j, B in enumerate(rhs):
                if _is_terminal(B):
                    continue
                if B not in follow:
                    continue
                before = len(follow[B])
                beta = rhs[j+1:]
                if beta:
                    beta_first = _first_of_sequence(beta, first_sets)
                    follow[B] |= (beta_first - {EPS})
                    if EPS in beta_first:
                        follow[B] |= follow[lhs]
                else:
                    follow[B] |= follow[lhs]
                if len(follow[B]) != before:
                    changed = True
    return follow


def _build_ll1_table(first_sets, follow_sets):
    # table[nt_index][term_index] = production_number or -1
    n_nt  = len(NON_TERMINALS)
    n_ter = len(TERMINALS)
    table = [[-1] * n_ter for _ in range(n_nt)]
    for pnum, lhs, rhs in PRODUCTIONS:
        ni = _NT_INDEX.get(lhs)
        if ni is None:
            continue
        first_rhs = _first_of_sequence(rhs, first_sets)
        for sym in first_rhs:
            if sym == EPS:
                for f in follow_sets[lhs]:
                    ti = _TER_INDEX.get(f)
                    if ti is not None:
                        table[ni][ti] = pnum
            else:
                ti = _TER_INDEX.get(sym)
                if ti is not None:
                    table[ni][ti] = pnum
    return table


# ─────────────────────────────────────────────
#  Left-Recursion Detector
# ─────────────────────────────────────────────

def _detect_left_recursion(productions):
    """
    Detect direct AND indirect left recursion in the grammar.

    Direct left recursion:  A -> A ...
    Indirect left recursion: A -> B ..., B -> A ...  (via nullable first symbols)

    Raises SystemExit immediately if any left recursion is found so that
    the parser stops completely at start-up before attempting to parse.
    """

    # ── 1. Collect all non-terminals and their RHS lists ──────────────────
    rhs_map: dict = {}          # NT -> list of RHS lists
    nt_order: list = []         # preserve declaration order for messages
    for _, lhs, rhs in productions:
        if lhs not in rhs_map:
            rhs_map[lhs] = []
            nt_order.append(lhs)
        rhs_map[lhs].append(rhs)

    all_nts = set(rhs_map.keys())

    # ── 2. Build a nullable set (NTs that can derive ε in one step) ───────
    nullable: set = set()
    for nt, rhs_list in rhs_map.items():
        for rhs in rhs_list:
            if rhs == [EPS] or rhs == ["ε"] or rhs == []:
                nullable.add(nt)
                break

    # Fixed-point: propagate nullable through rules whose entire RHS is nullable
    changed = True
    while changed:
        changed = False
        for nt, rhs_list in rhs_map.items():
            if nt in nullable:
                continue
            for rhs in rhs_list:
                if all(sym in nullable for sym in rhs if sym not in (EPS, "ε")):
                    nullable.add(nt)
                    changed = True
                    break

    # ── 3. Compute "left-corner" reachability  ────────────────────────────
    # left_corners[A] = set of NTs that can appear as the leftmost derived
    # symbol of A (including A itself via direct rules, and transitively).
    #
    # B is a left-corner of A if there is a production  A -> B ...
    # OR  A -> X1 X2 ... Xk B ...  where every Xi is nullable.

    def immediate_left_corners(nt):
        """NTs that appear as an immediate leftmost symbol in any RHS of nt."""
        corners = set()
        for rhs in rhs_map.get(nt, []):
            for sym in rhs:
                if sym in (EPS, "ε"):
                    break
                if sym in all_nts:
                    corners.add(sym)
                if sym not in nullable:
                    break           # sym is not nullable, so it blocks further
        return corners

    # Transitive closure via BFS/fixed-point
    left_reachable: dict = {nt: set() for nt in all_nts}
    for nt in all_nts:
        left_reachable[nt] = immediate_left_corners(nt)

    changed = True
    while changed:
        changed = False
        for nt in all_nts:
            before = len(left_reachable[nt])
            extra = set()
            for corner in list(left_reachable[nt]):
                extra |= left_reachable[corner]
            left_reachable[nt] |= extra
            if len(left_reachable[nt]) != before:
                changed = True

    # ── 4. Report any left-recursive non-terminals ────────────────────────
    W = 72
    recursive_nts = []
    direct_nts    = []
    indirect_nts  = []

    for nt in nt_order:
        is_direct = any(
            rhs and rhs[0] == nt          # first symbol is itself
            for rhs in rhs_map[nt]
        )
        is_indirect = (not is_direct) and (nt in left_reachable[nt])

        if is_direct:
            direct_nts.append(nt)
            recursive_nts.append((nt, "direct"))
        elif is_indirect:
            indirect_nts.append(nt)
            recursive_nts.append((nt, "indirect"))

    if not recursive_nts:
        return          # ── Grammar is safe; continue normally ──

    # ── 5. Fatal error: print a clear report and halt ─────────────────────
    print("\n" + "=" * W)
    print("  FATAL: LEFT RECURSION DETECTED IN GRAMMAR")
    print("=" * W)
    print()
    print("  This parser requires a left-recursion-free grammar.")
    print("  The following non-terminal(s) are left-recursive:\n")

    for nt, kind in recursive_nts:
        print(f"  ✘  {nt}  ({kind} left recursion)")
        # Show the offending production(s)
        for rhs in rhs_map[nt]:
            rhs_str = " ".join(rhs) if rhs not in ([EPS], ["ε"], []) else "ε"
            arrow = "  →  "
            if kind == "direct" and rhs and rhs[0] == nt:
                print(f"       {nt}{arrow}{rhs_str}   ← left-recursive rule")
            elif kind == "indirect":
                # show the left-corner chain
                for corner in sorted(left_reachable[nt]):
                    if nt in left_reachable.get(corner, set()) or corner == nt:
                        pass
                if rhs and rhs[0] in all_nts and (
                    rhs[0] in left_reachable[nt] or rhs[0] == nt
                ):
                    print(f"       {nt}{arrow}{rhs_str}   ← contributes to indirect cycle")

    print()
    if direct_nts:
        print("  How to fix DIRECT left recursion:")
        print("    Replace  A -> A α | β")
        print("    With     A -> β A'")
        print("             A' -> α A' | ε")
        print()
    if indirect_nts:
        print("  How to fix INDIRECT left recursion:")
        print("    Substitute productions to expose the direct recursion,")
        print("    then apply the direct left-recursion elimination above.")
        print()

    print("  Parser cannot continue.  Fix the grammar and re-run.")
    print("=" * W + "\n")
    sys.exit(1)


# ── Run the check immediately at module load, before building tables ──────
_detect_left_recursion(PRODUCTIONS)


# Build everything once at module load
_FIRST_SETS  = _compute_first_sets()
_FOLLOW_SETS = _compute_follow_sets(_FIRST_SETS)
_LL1_TABLE   = _build_ll1_table(_FIRST_SETS, _FOLLOW_SETS)

_PROD_BY_NUM = {pnum: (lhs, rhs) for pnum, lhs, rhs in PRODUCTIONS}


# ─────────────────────────────────────────────
#  Print helpers  (grammar info)
# ─────────────────────────────────────────────

def print_productions():
    W = 80
    print("\n" + "=" * W)
    print("LEFT-FACTORED GRAMMAR PRODUCTIONS")
    print("=" * W + "\n")
    for pnum, lhs, rhs in PRODUCTIONS:
        rhs_str = "eps" if rhs == [EPS] else " ".join(rhs)
        print(f"  [{pnum:2d}]  {lhs:<20} -> {rhs_str}")
    print()


def print_first_sets():
    W = 80
    print("\n" + "=" * W)
    print("FIRST SETS")
    print("=" * W + "\n")
    for nt in NON_TERMINALS:
        items = " ".join("eps" if s == EPS else s for s in sorted(_FIRST_SETS[nt]))
        print(f"  FIRST({nt:<18}) = {{ {items} }}")
    print()


def print_follow_sets():
    W = 80
    print("\n" + "=" * W)
    print("FOLLOW SETS")
    print("=" * W + "\n")
    for nt in NON_TERMINALS:
        items = " ".join(sorted(_FOLLOW_SETS[nt]))
        print(f"  FOLLOW({nt:<18}) = {{ {items} }}")
    print()


def print_ll1_table():
    W = 80
    print("\n" + "=" * W)
    print("LL(1) PARSING TABLE (production numbers; blank = error)")
    print("=" * W + "\n")
    NT_W = 20
    COL_W = 12
    COLS_PER_PASS = 6
    n_cols = len(TERMINALS)
    passes = (n_cols + COLS_PER_PASS - 1) // COLS_PER_PASS
    for p in range(passes):
        col_start = p * COLS_PER_PASS
        col_end   = min(col_start + COLS_PER_PASS, n_cols)
        sep_top = "+-" + "-" * NT_W + "-+" + "".join("-" * COL_W + "+" for _ in range(col_start, col_end))
        print(sep_top)
        hdr = f"| {' Non-Terminal':<{NT_W}} |" + "".join(f" {TERMINALS[j]:<{COL_W-1}}|" for j in range(col_start, col_end))
        print(hdr)
        print(sep_top)
        for i, nt in enumerate(NON_TERMINALS):
            row = f"| {nt:<{NT_W}} |"
            for j in range(col_start, col_end):
                v = _LL1_TABLE[i][j]
                cell = f"P{v}" if v != -1 else ""
                row += f" {cell:<{COL_W-1}}|"
            print(row)
        print(sep_top)
        print()
        if p < passes - 1:
            print("  (continued)\n")
    print("  Note: P<n> = apply production n  |  blank = syntax error\n")


# ─────────────────────────────────────────────
#  Node -> token list  (mirrors node_to_tokens in C)
# ─────────────────────────────────────────────

def node_to_tokens(node: ParseNode) -> List[Token]:
    """Flatten a parse-tree node into its leaf tokens (in order)."""
    result = []
    def _gather(n: ParseNode):
        if n.is_leaf():
            result.append(n.token)
        for c in n.children:
            _gather(c)
    _gather(node)
    return result


# ─────────────────────────────────────────────
#  LL(1) Parser  (mirrors ll1_parse_statement in C)
# ─────────────────────────────────────────────

def ll1_parse_statement(stmt: ParseNode, quiet: bool = False) -> bool:
    toks = node_to_tokens(stmt)
    if not quiet:
        print("\n" + "=" * 80)
        print("LL(1) PARSER - STATEMENT TRACE")
        print("=" * 80 + "\n")
        print("Input: " + " ".join(t.value for t in toks) + "\n")

    input_syms  = [token_to_terminal(t) for t in toks] + ["$"]
    input_lines = [t.line for t in toks] + [toks[-1].line if toks else 0]

    stack  = ["$", "stmt"]   # top is last element
    ip     = 0
    step   = 1
    success   = True
    ll1_errors = 0

    if not quiet:
        print(f"{'Step':<5}  {'Stack (top→bottom)':<42}  {'Remaining Input':<30}  Action")
        print("-" * 5 + "  " + "-" * 42 + "  " + "-" * 30 + "  " + "-" * 33)

    while stack and step < 300:
        X = stack[-1]
        a = input_syms[ip] if ip < len(input_syms) else "$"

        stack_str = " ".join(reversed(stack))[:42]
        remain    = " ".join(input_syms[ip:ip+5])
        if ip + 5 < len(input_syms):
            remain += " ..."
        remain = remain[:30]

        if not quiet:
            print(f"{step:<5}  {stack_str:<42}  {remain:<30}  ", end="")
        step += 1

        if X == "$" and a == "$":
            if not quiet:
                print("ACCEPT")
            break

        if _is_terminal(X):
            if X == a:
                if not quiet:
                    print(f"Match '{X}'")
                stack.pop()
                ip += 1
            elif X == EPS:
                if not quiet:
                    print("Pop eps")
                stack.pop()
            else:
                ln = input_lines[ip] if ip < len(input_lines) else 0
                if not quiet:
                    print(f"*** ERROR: expected '{X}', found '{a}' (line {ln})")
                ll1_errors += 1
                success = False
                break
        else:
            ni = _NT_INDEX.get(X, -1)
            ti = _TER_INDEX.get(a, -1)
            if ni < 0 or ti < 0 or _LL1_TABLE[ni][ti] == -1:
                ln = input_lines[ip] if ip < len(input_lines) else 0
                if not quiet:
                    print(f"*** ERROR: no production for [{X},{a}] (line {ln})")
                ll1_errors += 1
                success = False
                break
            pnum = _LL1_TABLE[ni][ti]
            _, rhs = _PROD_BY_NUM[pnum]
            rhs_str = "eps" if rhs == [EPS] else " ".join(rhs)
            if not quiet:
                print(f"Apply P{pnum}: {X} -> {rhs_str}")
            stack.pop()
            if rhs != [EPS]:
                for sym in reversed(rhs):
                    stack.append(sym)

    if not quiet:
        print()
        if ll1_errors > 0:
            print(f"[FAILED] LL(1) Parser: {ll1_errors} error(s) detected.\n")
        elif success:
            print("[SUCCESS] LL(1) Parser accepted the statement.\n")
        else:
            print("[FAILED] LL(1) Parser: step limit reached.\n")

    return success and ll1_errors == 0


# ─────────────────────────────────────────────
#  Shift-Reduce Parser  (mirrors sr_parse_statement in C)
# ─────────────────────────────────────────────

class _SRStack:
    """Simple stack of (symbol, line) pairs."""
    def __init__(self):
        self._data: List[tuple] = []   # list of (sym, line)

    @property
    def top(self) -> int:
        return len(self._data) - 1

    def push(self, sym: str, line: int = 0):
        self._data.append((sym, line))

    def pop(self):
        if self._data:
            self._data.pop()

    def check(self, offset: int, sym: str) -> bool:
        idx = self.top - offset
        return idx >= 0 and self._data[idx][0] == sym

    def sym_at(self, offset: int) -> str:
        idx = self.top - offset
        return self._data[idx][0] if 0 <= idx <= self.top else ""

    def line_at(self, offset: int) -> int:
        idx = self.top - offset
        return self._data[idx][1] if 0 <= idx <= self.top else 0

    def as_str(self) -> str:
        return " ".join(s for s, _ in self._data) if self._data else "(empty)"

    def __len__(self):
        return len(self._data)


def _sr_try_reduce(sr: _SRStack, la: str, desc_holder: list) -> bool:
    """
    Attempt one reduction.  Returns True if a reduction was made and
    writes a description string into desc_holder[0].
    Mirrors sr_try_reduce() in the C code exactly.
    """

    def desc(s):
        desc_holder[0] = s

    top = sr.top   # index of stack top

    # 0. Empty block
    if sr.check(0, "LBRACE") and la == "RBRACE":
        ln = sr.line_at(0)
        sr.push("stmt_list", ln)
        desc("[3] stmt_list -> eps (empty block)")
        return True

    # A. Declaration
    if (top >= 4 and sr.check(4,"TYPE") and sr.check(3,"ID") and
            sr.check(2,"ASSIGNMENT") and sr.check(1,"arith_expr") and sr.check(0,"SEMI")):
        ln = sr.line_at(2)
        sr.pop(); sr.pop(); sr.pop()
        sr.push("decl_prime", ln)
        desc("[11] decl_prime -> = arith_expr ;"); return True

    if (top >= 2 and sr.check(2,"TYPE") and sr.check(1,"ID") and sr.check(0,"SEMI")):
        ln = sr.line_at(0)
        sr.pop()
        sr.push("decl_prime", ln)
        desc("[12] decl_prime -> ;"); return True

    if (top >= 2 and sr.check(2,"TYPE") and sr.check(1,"ID") and sr.check(0,"decl_prime")):
        ln = sr.line_at(2)
        sr.pop(); sr.pop(); sr.pop()
        sr.push("decl_stmt", ln)
        desc("[10] decl_stmt -> TYPE ID decl_prime"); return True

    if sr.check(0,"decl_stmt"):
        ln = sr.line_at(0); sr.pop(); sr.push("stmt", ln)
        desc("[4] stmt -> decl_stmt"); return True

    # B. Assignment
    if (top >= 3 and sr.check(3,"ID") and sr.check(2,"ASSIGNMENT") and
            sr.check(1,"arith_expr") and sr.check(0,"SEMI")):
        if not (top >= 4 and sr.check(4,"TYPE")):
            ln = sr.line_at(3)
            sr.pop(); sr.pop(); sr.pop(); sr.pop()
            sr.push("assign_stmt", ln)
            desc("[13] assign_stmt -> ID = arith_expr ;"); return True

    if sr.check(0,"assign_stmt"):
        ln = sr.line_at(0); sr.pop(); sr.push("stmt", ln)
        desc("[5] stmt -> assign_stmt"); return True

    # C. Factor from terminal atoms
    if sr.check(0,"ID"):
        if top >= 1 and sr.check(1,"TYPE"):
            return False
        if la == "ASSIGNMENT":
            return False
        ln = sr.line_at(0); sr.pop(); sr.push("factor", ln)
        desc("[30] factor -> ID"); return True

    if sr.check(0,"INTEGER"):
        ln = sr.line_at(0); sr.pop(); sr.push("factor", ln)
        desc("[31] factor -> INTEGER"); return True

    if sr.check(0,"FLOAT"):
        ln = sr.line_at(0); sr.pop(); sr.push("factor", ln)
        desc("[32] factor -> FLOAT"); return True

    # D. factor -> ( arith_expr )
    if (top >= 2 and sr.check(2,"LPAREN") and sr.check(1,"arith_expr") and sr.check(0,"RPAREN")):
        is_ctrl = top >= 3 and (sr.check(3,"IF") or sr.check(3,"WHILE") or sr.check(3,"PRINT"))
        if not is_ctrl:
            ln = sr.line_at(2)
            sr.pop(); sr.pop(); sr.pop()
            sr.push("factor", ln)
            desc("[29] factor -> ( arith_expr )"); return True

    # E. term_tail
    if top >= 2 and sr.check(2,"MULT") and sr.check(1,"factor") and sr.check(0,"term_tail"):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("term_tail", ln); desc("[25] term_tail -> * factor term_tail"); return True

    if top >= 2 and sr.check(2,"DIV") and sr.check(1,"factor") and sr.check(0,"term_tail"):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("term_tail", ln); desc("[26] term_tail -> / factor term_tail"); return True

    if top >= 2 and sr.check(2,"MOD") and sr.check(1,"factor") and sr.check(0,"term_tail"):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("term_tail", ln); desc("[27] term_tail -> %% factor term_tail"); return True

    if sr.check(0,"factor"):
        if la not in ("MULT","DIV","MOD"):
            ln = sr.line_at(0)
            sr.push("term_tail", ln)
            desc("[28] term_tail -> eps"); return True

    # F. term
    if top >= 1 and sr.check(1,"factor") and sr.check(0,"term_tail"):
        ln = sr.line_at(1); sr.pop(); sr.pop()
        sr.push("term", ln); desc("[24] term -> factor term_tail"); return True

    # G. arith_expr_tail
    if top >= 2 and sr.check(2,"PLUS") and sr.check(1,"term") and sr.check(0,"arith_expr_tail"):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("arith_expr_tail", ln); desc("[21] arith_expr_tail -> + term arith_expr_tail"); return True

    if top >= 2 and sr.check(2,"MINUS") and sr.check(1,"term") and sr.check(0,"arith_expr_tail"):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("arith_expr_tail", ln); desc("[22] arith_expr_tail -> - term arith_expr_tail"); return True

    if sr.check(0,"term"):
        if la not in ("PLUS","MINUS"):
            ln = sr.line_at(0)
            sr.push("arith_expr_tail", ln)
            desc("[23] arith_expr_tail -> eps"); return True

    # H. arith_expr
    if top >= 1 and sr.check(1,"term") and sr.check(0,"arith_expr_tail"):
        ln = sr.line_at(1); sr.pop(); sr.pop()
        sr.push("arith_expr", ln); desc("[20] arith_expr -> term arith_expr_tail"); return True

    # I. Boolean expressions
    if sr.check(0,"REL_OP"):
        ln = sr.line_at(0); sr.pop(); sr.push("rel_op", ln)
        desc("[42] rel_op -> REL_OP"); return True

    if (top >= 2 and sr.check(2,"arith_expr") and sr.check(1,"rel_op") and sr.check(0,"arith_expr")):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("bool_factor", ln)
        desc("[39] bool_factor -> arith_expr rel_op arith_expr"); return True

    if top >= 1 and sr.check(1,"NOT") and sr.check(0,"bool_factor"):
        ln = sr.line_at(1); sr.pop(); sr.pop()
        sr.push("bool_factor", ln)
        desc("[41] bool_factor -> NOT bool_factor"); return True

    if (top >= 2 and sr.check(2,"LPAREN") and sr.check(1,"bool_expr") and sr.check(0,"RPAREN")):
        is_control = top >= 3 and (sr.check(3,"IF") or sr.check(3,"WHILE"))
        if not is_control:
            ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
            sr.push("bool_factor", ln)
            desc("[40] bool_factor -> ( bool_expr )"); return True

    if sr.check(0,"bool_factor"):
        if la != "LOGICAL_AND":
            ln = sr.line_at(0)
            sr.push("bool_term_tail", ln)
            desc("[38] bool_term_tail -> eps"); return True

    if (top >= 2 and sr.check(2,"LOGICAL_AND") and sr.check(1,"bool_factor") and sr.check(0,"bool_term_tail")):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("bool_term_tail", ln)
        desc("[37] bool_term_tail -> && bool_factor bool_term_tail"); return True

    if top >= 1 and sr.check(1,"bool_factor") and sr.check(0,"bool_term_tail"):
        ln = sr.line_at(1); sr.pop(); sr.pop()
        sr.push("bool_term", ln)
        desc("[36] bool_term -> bool_factor bool_term_tail"); return True

    if sr.check(0,"bool_term"):
        if la != "LOGICAL_OR":
            ln = sr.line_at(0)
            sr.push("bool_expr_tail", ln)
            desc("[35] bool_expr_tail -> eps"); return True

    if (top >= 2 and sr.check(2,"LOGICAL_OR") and sr.check(1,"bool_term") and sr.check(0,"bool_expr_tail")):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("bool_expr_tail", ln)
        desc("[34] bool_expr_tail -> || bool_term bool_expr_tail"); return True

    if top >= 1 and sr.check(1,"bool_term") and sr.check(0,"bool_expr_tail"):
        ln = sr.line_at(1); sr.pop(); sr.pop()
        sr.push("bool_expr", ln)
        desc("[33] bool_expr -> bool_term bool_expr_tail"); return True

    # J. Print statement
    if (top >= 4 and sr.check(4,"PRINT") and sr.check(3,"LPAREN") and sr.check(2,"arith_expr") and
            sr.check(1,"RPAREN") and sr.check(0,"SEMI")):
        ln = sr.line_at(4)
        sr.pop(); sr.pop(); sr.pop(); sr.pop(); sr.pop()
        sr.push("print_stmt", ln)
        desc("[18] print_stmt -> print ( arith_expr ) ;"); return True

    if sr.check(0,"print_stmt"):
        ln = sr.line_at(0); sr.pop(); sr.push("stmt", ln)
        desc("[8] stmt -> print_stmt"); return True

    # K. Block
    if (top >= 2 and sr.check(2,"LBRACE") and sr.check(1,"stmt_list") and sr.check(0,"RBRACE")):
        ln = sr.line_at(2); sr.pop(); sr.pop(); sr.pop()
        sr.push("block", ln)
        desc("[19] block -> { stmt_list }"); return True

    # L. Statement list
    if top >= 1 and sr.check(1,"stmt") and sr.check(0,"stmt_list"):
        ln = sr.line_at(1); sr.pop(); sr.pop()
        sr.push("stmt_list", ln)
        desc("[2] stmt_list -> stmt stmt_list"); return True

    if sr.check(0,"stmt") and la in ("RBRACE","$"):
        ln = sr.line_at(0)
        sr.push("stmt_list", ln)
        sr.pop(); sr.pop()
        sr.push("stmt_list", ln)
        desc("[2+3] stmt_list -> stmt eps"); return True

    # M. While statement
    if (top >= 4 and sr.check(4,"WHILE") and sr.check(3,"LPAREN") and sr.check(2,"bool_expr") and
            sr.check(1,"RPAREN") and sr.check(0,"block")):
        ln = sr.line_at(4); sr.pop(); sr.pop(); sr.pop(); sr.pop(); sr.pop()
        sr.push("while_stmt", ln)
        desc("[17] while_stmt -> while ( bool_expr ) block"); return True

    if sr.check(0,"while_stmt"):
        ln = sr.line_at(0); sr.pop(); sr.push("stmt", ln)
        desc("[7] stmt -> while_stmt"); return True

    # N. If statement
    if top >= 1 and sr.check(1,"ELSE") and sr.check(0,"block"):
        ln = sr.line_at(1); sr.pop(); sr.pop()
        sr.push("else_prime", ln)
        desc("[15] else_prime -> else block"); return True

    if (top >= 5 and sr.check(5,"IF") and sr.check(4,"LPAREN") and sr.check(3,"bool_expr") and
            sr.check(2,"RPAREN") and sr.check(1,"block") and sr.check(0,"else_prime")):
        ln = sr.line_at(5)
        sr.pop(); sr.pop(); sr.pop(); sr.pop(); sr.pop(); sr.pop()
        sr.push("if_stmt", ln)
        desc("[14] if_stmt -> if ( bool_expr ) block else_prime"); return True

    if (top >= 4 and sr.check(4,"IF") and sr.check(3,"LPAREN") and sr.check(2,"bool_expr") and
            sr.check(1,"RPAREN") and sr.check(0,"block")):
        if la != "ELSE":
            ln = sr.line_at(4)
            sr.pop(); sr.pop(); sr.pop(); sr.pop(); sr.pop()
            sr.push("if_stmt", ln)
            desc("[14+16] if_stmt -> if ( bool_expr ) block (no else)"); return True

    if sr.check(0,"if_stmt"):
        ln = sr.line_at(0); sr.pop(); sr.push("stmt", ln)
        desc("[6] stmt -> if_stmt"); return True

    # O. Block -> stmt
    if sr.check(0,"block"):
        dominated = False
        for offset in range(1, sr.top + 1):
            s = sr.sym_at(offset)
            if s in ("IF","WHILE","ELSE"):
                dominated = True; break
            if s == "LBRACE":
                break
        if not dominated and la in ("RBRACE","$","TYPE","ID","IF","WHILE","PRINT","LBRACE"):
            ln = sr.line_at(0); sr.pop(); sr.push("stmt", ln)
            desc("[9] stmt -> block"); return True

    return False


def sr_parse_statement(stmt: ParseNode, quiet: bool = False) -> bool:
    toks = node_to_tokens(stmt)
    if not quiet:
        print("\n" + "=" * 80)
        print("SHIFT-REDUCE PARSER - STATEMENT TRACE")
        print("=" * 80 + "\n")
        print("Input: " + " ".join(t.value for t in toks) + "\n")

    input_syms  = [token_to_terminal(t) for t in toks] + ["$"]
    input_lines = [t.line for t in toks] + [toks[-1].line if toks else 0]

    sr = _SRStack()
    ip        = 0
    step      = 1
    success   = True
    sr_errors = 0

    if not quiet:
        print(f"{'Step':<5}  {'Stack (bottom→top)':<45}  {'Remaining Input':<30}  Action")
        print("-" * 5 + "  " + "-" * 45 + "  " + "-" * 30 + "  " + "-" * 33)

    while step < 500 and ip <= len(input_syms):
        la = input_syms[ip] if ip < len(input_syms) else "$"

        stack_str = sr.as_str()[:45]
        remain    = " ".join(input_syms[ip:ip+6])
        if ip + 6 < len(input_syms):
            remain += " ..."
        remain = remain[:30]

        if not quiet:
            print(f"{step:<5}  {stack_str:<45}  {remain:<30}  ", end="")
        step += 1

        # Accept check
        if len(sr) == 1 and sr.check(0,"stmt") and la == "$":
            if not quiet:
                print("ACCEPT")
            break

        desc_holder = [""]
        reduced = False
        while _sr_try_reduce(sr, la, desc_holder):
            if not quiet:
                print(f"Reduce: {desc_holder[0]}")
            reduced = True
            la = input_syms[ip] if ip < len(input_syms) else "$"
            # Recheck accept after each reduction
            if len(sr) == 1 and sr.check(0,"stmt") and la == "$":
                if not quiet:
                    print("ACCEPT")
                goto_accept = True
                break
            # Print next step number & blank columns for continued reductions
            if not quiet:
                print(f"{'':5}  {'':45}  {'':30}  ", end="")
            desc_holder = [""]
        else:
            goto_accept = False

        if goto_accept:
            break

        if reduced:
            continue

        if la != "$":
            if not quiet:
                print(f"Shift: {la}")
            ln = input_lines[ip] if ip < len(input_lines) else 0
            sr.push(la, ln)
            ip += 1
        else:
            if not quiet:
                print("*** ERROR: cannot reduce or shift at '$'")
            sr_errors += 1
            success = False
            break

    if step >= 500:
        if not quiet:
            print("*** ERROR: step limit reached")
        success = False
        sr_errors += 1

    if not quiet:
        print()
        if sr_errors > 0:
            print(f"[FAILED] Shift-Reduce Parser: {sr_errors} error(s) detected.\n")
        elif success:
            print("[SUCCESS] Shift-Reduce Parser accepted the statement.\n")
        else:
            print("[FAILED] Shift-Reduce Parser did not accept.\n")

    return success and sr_errors == 0


# ─────────────────────────────────────────────
#  Derivation utilities
# ─────────────────────────────────────────────

def find_first_node(root: ParseNode, label: str) -> Optional[ParseNode]:
    if root.label == label:
        return root
    for c in root.children:
        found = find_first_node(c, label)
        if found:
            return found
    return None


def _render(sentential: List[ParseNode]) -> str:
    parts = []
    for n in sentential:
        if n.is_leaf():
            parts.append(n.token.value)
        elif n.label == "ε":
            pass
        else:
            parts.append(n.label)
    return " ".join(parts) if parts else ""


def leftmost_derivation(root: ParseNode, max_steps: int = 200) -> List[str]:
    steps: List[str] = []
    sentential: List[ParseNode] = [root]
    steps.append(_render(sentential))
    for _ in range(max_steps):
        idx = next(
            (i for i, n in enumerate(sentential)
             if not n.is_leaf() and n.label != "ε" and n.children),
            None
        )
        if idx is None:
            break
        node = sentential[idx]
        sentential = sentential[:idx] + node.children + sentential[idx + 1:]
        rendered = _render(sentential)
        if rendered != steps[-1]:
            steps.append(rendered)
    return steps


def rightmost_derivation(root: ParseNode, max_steps: int = 200) -> List[str]:
    steps: List[str] = []
    sentential: List[ParseNode] = [root]
    steps.append(_render(sentential))
    for _ in range(max_steps):
        idx = next(
            (i for i in range(len(sentential) - 1, -1, -1)
             if not sentential[i].is_leaf()
             and sentential[i].label != "ε"
             and sentential[i].children),
            None
        )
        if idx is None:
            break
        node = sentential[idx]
        sentential = sentential[:idx] + node.children + sentential[idx + 1:]
        rendered = _render(sentential)
        if rendered != steps[-1]:
            steps.append(rendered)
    return steps


def print_derivation(steps: List[str], direction: str):
    W = 60
    print("\n" + "=" * W)
    print(f"  {direction} DERIVATION")
    print("=" * W)
    for i, step in enumerate(steps):
        print(f"  Step {i + 1} => {step}")
    print("=" * W)


# ─────────────────────────────────────────────
#  Statement collection helpers
# ─────────────────────────────────────────────

def collect_stmts(node: ParseNode, results: List[ParseNode] = None) -> List[ParseNode]:
    if results is None:
        results = []
    if node.label == "stmt":
        results.append(node)
    for child in node.children:
        collect_stmts(child, results)
    return results


def collect_valid_stmts(node: ParseNode, results: List[ParseNode] = None) -> List[ParseNode]:
    if results is None:
        results = []
    if node.label == "stmt":
        has_error = any(c.label == "ERROR" for c in node.children)
        if not has_error:
            results.append(node)
        return results
    for child in node.children:
        collect_valid_stmts(child, results)
    return results


def stmt_source(node: ParseNode) -> str:
    parts = []
    def gather(n: ParseNode):
        if n.is_leaf():
            parts.append(n.token.value)
        for c in n.children:
            gather(c)
    gather(node)
    return " ".join(parts)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python parser.py <source_file>")
        print("Example: python parser.py test_program.src")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)

    # ── Lex ──────────────────────────────────────
    lexer  = Lexer(source)
    tokens = lexer.tokenize()

    if lexer.errors:
        print("Lexical errors detected:")
        for e in lexer.errors:
            print(" ", e)
        sys.exit(1)

    lexer.print_token_stream()

    # ── Print grammar info ────────────────────────
    print_productions()
    print_first_sets()
    print_follow_sets()
    print_ll1_table()

    # ── Parse (with panic-mode recovery) ─────────
    tree   = None
    parser = Parser(tokens)
    try:
        tree = parser.parse_program()
    except ParseError as e:
        parser.errors.append(str(e))

    W = 80

    # ── Report ALL syntax errors ──────────────────
    if parser.errors:
        print("\n" + "=" * W)
        print("  SYNTAX ERRORS DETECTED")
        print("=" * W)
        for i, err in enumerate(parser.errors, 1):
            print(f"  [{i}] Syntax Error: {err}")
        print("=" * W)
        print(f"\n  ✘  {len(parser.errors)} syntax error(s) found. "
              f"Fix the errors above and re-run.\n")
        sys.exit(1)

    # ── Full parse tree (no errors) ──────────────
    print("\n" + "=" * W)
    print("  PARSE TREE — ENTIRE PROGRAM")
    print("=" * W)
    print("  program")
    for i, child in enumerate(tree.children):
        print(child.pretty("  ", i == len(tree.children) - 1))
    print("=" * W)

    # ── Collect all statements ────────────────────
    all_stmts = collect_valid_stmts(tree)

    if not all_stmts:
        print("\n  No valid statements available.\n")
        sys.exit(0)

    # ── Silent pre-pass: run LL(1) + SR on all stmts ──
    stmt_ll1_ok = []
    stmt_sr_ok  = []
    total_errors_found = 0

    for s in all_stmts:
        has_error = any(c.label == "ERROR" for c in s.children)
        if has_error:
            stmt_ll1_ok.append(False)
            stmt_sr_ok.append(False)
            total_errors_found += 1
        else:
            l_ok = ll1_parse_statement(s, quiet=True)
            r_ok = sr_parse_statement(s, quiet=True)
            stmt_ll1_ok.append(l_ok)
            stmt_sr_ok.append(r_ok)
            if not l_ok or not r_ok:
                total_errors_found += 1

    # ── Print stack traces for erroneous statements only ──
    if total_errors_found > 0:
        print("\n" + "=" * W)
        print("DETAILED STACK TRACES FOR ERRONEOUS STATEMENTS")
        print("=" * W)

        first_ll1 = True
        for i, s in enumerate(all_stmts):
            if not stmt_ll1_ok[i]:
                if first_ll1:
                    print("\n--- LL(1) Parser Traces ---")
                    first_ll1 = False
                src = stmt_source(s)
                inner = s.children[0] if s.children else s
                print(f"\n--- Statement {i+1} (line {s.children[0].token.line if s.children and s.children[0].is_leaf() else '?'}): {src} ---")
                ll1_parse_statement(s, quiet=False)

        first_sr = True
        for i, s in enumerate(all_stmts):
            if not stmt_sr_ok[i]:
                if first_sr:
                    print("\n--- Shift-Reduce Parser Traces ---")
                    first_sr = False
                src = stmt_source(s)
                print(f"\n--- Statement {i+1}: {src} ---")
                sr_parse_statement(s, quiet=False)

        print()

    # ── Statements listing ────────────────────────
    print("\n" + "=" * W)
    print(f"STATEMENTS FOUND ({len(all_stmts)} total)")
    print("=" * W)
    for i, s in enumerate(all_stmts):
        inner   = s.children[0] if s.children else s
        src     = stmt_source(s)
        both_ok = stmt_ll1_ok[i] and stmt_sr_ok[i]
        has_err = any(c.label == "ERROR" for c in s.children)
        tag = "  [SYNTAX ERROR]" if has_err else ("  [PARSE ERROR]" if not both_ok else "")
        print(f"  [{i+1:2d}]  {inner.label:<15}  {src}{tag}")
    print("=" * W)

    # ── Statement selection ───────────────────────
    while True:
        try:
            raw    = input(f"\n  Enter statement number (1-{len(all_stmts)}): ").strip()
            choice = int(raw)
            if 1 <= choice <= len(all_stmts):
                break
            print(f"  Please enter a number between 1 and {len(all_stmts)}.")
        except ValueError:
            print("  Invalid input — enter a number.")

    selected = all_stmts[choice - 1]
    src_text = stmt_source(selected)

    print("\n" + "=" * W)
    print("  SELECTED: " + src_text)
    print("=" * W + "\n")

    # Run parsers verbosely on selected statement
    ll1_ok = ll1_parse_statement(selected, quiet=False)
    sr_ok  = sr_parse_statement(selected, quiet=False)

    both_ok = ll1_ok and sr_ok

    if both_ok:
        print("=" * W)
        print("BOTH PARSERS ACCEPTED - Showing derivations and statement parse tree")
        print("=" * W + "\n")

        print("--- LEFTMOST DERIVATION ---")
        lm_steps = leftmost_derivation(selected)
        print_derivation(lm_steps, "LEFTMOST")

        print("\n--- RIGHTMOST DERIVATION ---")
        rm_steps = rightmost_derivation(selected)
        print_derivation(rm_steps, "RIGHTMOST")

        inner = selected.children[0] if selected.children else selected
        print(f"\n--- STATEMENT PARSE TREE ---")
        print("=" * W)
        print(f"  PARSE TREE — {inner.label}  [ {src_text} ]")
        print("=" * W)
        print(f"  {selected.label}")
        for i, child in enumerate(selected.children):
            print(child.pretty("  ", i == len(selected.children) - 1))
        print("=" * W + "\n")
    else:
        print("=" * W)
        print("PARSER(S) REJECTED THE STATEMENT")
        print("=" * W)
        print(f"  LL(1) parser : {'ACCEPTED' if ll1_ok else 'REJECTED'}")
        print(f"  S/R  parser  : {'ACCEPTED' if sr_ok  else 'REJECTED'}")
        print()
        print("  Parse tree and derivations are only shown for syntactically correct")
        print("  statements that are accepted by BOTH parsers.")

    print("\n" + "=" * W)
    print("All phases complete.")
    print("=" * W)


if __name__ == "__main__":
    main()