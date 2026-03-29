"""
Recursive-Descent Parser
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

_synchronise() strategy:
  - Skip tokens until a STATEMENT-START token or EOF is reached.
  - STATEMENT-START tokens: INT, FLOAT, IDENTIFIER, IF, WHILE, PRINT
  - SEMICOLON is consumed and then skipped past (end of broken stmt).
  - RBRACE is left alone — it belongs to parse_block's closing check.
  - LBRACE that appears after a broken condition is consumed and then
    the block content is PARSED (not blindly skipped) via
    _scan_block_for_errors(), so every interior error (missing semicolons,
    bad expressions, etc.) is collected before recovery continues.
  - EOF always terminates recovery.

Hang prevention
===============
  - _assert_expr_start() / _assert_bool_factor_start() bail out
    immediately when a non-expression token appears mid-expression
    (e.g. LBRACE or EOF when ')' is missing).
  - _lookahead_is_bool() is capped at MAX_LOOKAHEAD tokens and returns
    False immediately on LBRACE / EOF so it never scans past a block.
  - _skip_block() is used inside _synchronise() to swallow an orphaned
    block that resulted from a broken while/if header.
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


# Tokens that can start a fresh statement — used by _synchronise() to
# know when it is safe to hand control back to parse_stmt_list().
_STMT_START_TOKENS = {
    "INT", "FLOAT",      # declaration
    "IDENTIFIER",        # assignment
    "IF",                # if-statement
    "WHILE",             # while-statement
    "PRINT",             # print-statement
}

# Tokens that can legally begin a factor / expression.
_EXPR_START_TOKENS = {
    "INT_LITERAL", "FLOAT_LITERAL", "IDENTIFIER", "LPAREN", "MINUS",
}

# Tokens that can legally begin a boolean factor.
_BOOL_FACTOR_START_TOKENS = _EXPR_START_TOKENS | {"NOT"}

# Hard cap for _lookahead_is_bool() scan.
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
        """
        If the current token is on a different line from the previous one,
        the error belongs to the END of the previous line (e.g. missing ';').
        """
        cur = self.current
        if self.pos > 0:
            prev = self.tokens[self.pos - 1]
            if prev.line < cur.line:
                return prev.line, prev.col + len(str(prev.value))
        return cur.line, cur.col

    # ── Block skipper / scanner ────────────────

    def _skip_block(self):
        """
        Fast path: skip an entire block without reporting interior errors.
        Only used when we cannot safely descend (e.g. deeply nested recovery).
        Consumes tokens until the matching RBRACE, then consumes the RBRACE.
        """
        if self.current.type != "LBRACE":
            return
        depth = 0
        while self.current.type != "EOF":
            if self.current.type == "LBRACE":
                depth += 1
            elif self.current.type == "RBRACE":
                depth -= 1
                if depth == 0:
                    self.pos += 1   # consume the closing }
                    return
            self.pos += 1

    def _scan_block_for_errors(self):
        """
        Called by _synchronise() when it encounters an orphaned LBRACE
        that results from a broken while/if header.

        Instead of silently swallowing the block, we PARSE its contents
        so that every interior error (missing semicolons, bad expressions,
        etc.) is collected — exactly as parse_block / parse_stmt_list would
        do for a well-formed header.

        Strategy
        --------
        1. Consume the opening '{'.
        2. Call parse_stmt_list() — this reports all interior statement
           errors through the normal panic-mode mechanism.
        3. Consume the closing '}' if present; if it is missing, record
           that error too.
        4. If the current token is NOT an LBRACE at all, return immediately
           (same guard as the old _skip_block).
        """
        if self.current.type != "LBRACE":
            return
        self.pos += 1                       # consume '{'
        self.parse_stmt_list()              # report all interior errors
        if self.current.type == "RBRACE":
            self.pos += 1                   # consume '}'
        elif self.current.type != "EOF":
            # Missing closing brace — record the error and move on.
            line, col = self._error_location()
            self.errors.append(
                f"Line {line}, col {col}: "
                f"expected 'RBRACE' to close block but got "
                f"{self.current.type!r} ({self.current.value!r})"
            )

    # ── Panic-mode recovery ───────────────────

    def _is_real_stmt_start(self) -> bool:
        """
        Return True only when the current token genuinely opens a new
        statement, not when it merely appears mid-expression/condition.

        Rules per token type:
          IDENTIFIER   -> next token must be '=' or ';'  (assignment start)
          IF / WHILE   -> next token must be '('         (condition start)
          PRINT        -> next token must be '('         (argument start)
          INT / FLOAT  -> next token must be IDENTIFIER  (declaration start)
          anything else -> False  (should not happen; caller already checked
                                   that current type is in _STMT_START_TOKENS)
        """
        t    = self.current.type
        nxt  = self.peek(1).type
        if t == "IDENTIFIER":
            return nxt in ("ASSIGN", "SEMICOLON")
        if t in ("IF", "WHILE", "PRINT"):
            return nxt == "LPAREN"
        if t in ("INT", "FLOAT"):
            return nxt == "IDENTIFIER"
        return False

    def _synchronise(self):
        """
        Skip tokens until a safe statement-start token or EOF is reached.

        Token handling during the skip:
          SEMICOLON  -> consume it (ends the broken statement), then stop
                        only if the next token is a real stmt-start or EOF.
          LBRACE     -> the broken while/if left an orphaned block;
                        descend into it with _scan_block_for_errors() so
                        every interior error is reported before continuing.
          RBRACE     -> stop WITHOUT consuming; belongs to parse_block.
          EOF        -> stop always.
          stmt-start -> stop only when _is_real_stmt_start() confirms the
                        token genuinely opens a new statement, not a
                        mid-condition identifier/keyword that would cause
                        the broken line to be re-entered and re-errored.
          anything else -> skip and continue.
        """
        while True:
            t = self.current.type

            if t == "EOF":
                return

            if t == "RBRACE":
                # Let parse_block's closing-brace check handle it.
                return

            if t == "SEMICOLON":
                self.pos += 1          # consume the semicolon
                # Stop here only if we are now at a real statement boundary.
                if self.current.type in ("EOF", "RBRACE"):
                    return
                if self.current.type in _STMT_START_TOKENS and \
                        self._is_real_stmt_start():
                    return
                continue

            if t == "LBRACE":
                # Orphaned block from a broken header — descend into it
                # so that interior errors (missing semicolons, bad exprs,
                # etc.) are all collected rather than silently swallowed.
                self._scan_block_for_errors()
                # After the block, check whether we are at a clean point.
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
        """Raise ParseError if the current token cannot start a factor."""
        tok = self.current
        if tok.type not in _EXPR_START_TOKENS:
            line, col = self._error_location()
            raise ParseError(
                f"Line {line}, col {col}: "
                f"unexpected token in expression: {tok.type!r} ({tok.value!r})"
            )

    def _assert_bool_factor_start(self):
        """Raise ParseError if the current token cannot start a bool factor."""
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

    # program -> stmt_list EOF
    def parse_program(self) -> ParseNode:
        node = ParseNode("program")
        node.add(self.parse_stmt_list())
        # After the top-level stmt_list, skip any stray tokens (e.g. an extra
        # RBRACE from a malformed block) so that the parser continues to reach
        # subsequent statements rather than dying here and masking later errors.
        while not self.match("EOF"):
            tok = self.current
            line, col = self._error_location()
            self.errors.append(
                f"Line {line}, col {col}: "
                f"unexpected token {tok.type!r} ({tok.value!r}) at top level"
            )
            self.pos += 1
            # Resume stmt_list parsing if we land on a statement-start token
            if self.current.type in _STMT_START_TOKENS:
                node.add(self.parse_stmt_list())
        node.add(self.leaf(self.consume("EOF")))
        return node

    # stmt_list -> stmt stmt_list | ε
    # Stops on EOF or RBRACE only — LBRACE is never a valid stmt start
    # and is handled by _synchronise() so it never reaches here naked.
    def parse_stmt_list(self) -> ParseNode:
        node = ParseNode("stmt_list")
        while not self.match("EOF", "RBRACE"):
            node.add(self.parse_stmt())
        if not node.children:
            node.add(ParseNode("ε"))
        return node

    # stmt -> decl_stmt | assign_stmt | if_stmt | while_stmt | print_stmt
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

    # decl_stmt -> TYPE ID ; | TYPE ID = expr ;
    def parse_decl_stmt(self) -> ParseNode:
        node = ParseNode("decl_stmt")
        type_node = ParseNode("TYPE")
        type_node.add(self.leaf(self.consume(self.current.type)))
        node.add(type_node)
        # If the token after the type keyword is not an identifier (e.g. a
        # numeric literal like '123' in 'int 123abc'), skip everything up to
        # and including the semicolon before raising so that leftover fragments
        # of the same bad declaration (e.g. 'abc') are not mistaken for the
        # start of a new statement and don't produce a second spurious error.
        if not self.match("IDENTIFIER"):
            bad = self.current
            line, col = bad.line, bad.col
            while not self.match("SEMICOLON", "EOF", "RBRACE"):
                self.pos += 1
            if self.match("SEMICOLON"):
                self.pos += 1   # consume the semicolon too
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

    # assign_stmt -> ID = expr ;
    def parse_assign_stmt(self) -> ParseNode:
        node = ParseNode("assign_stmt")
        node.add(ParseNode("ID", token=self.consume("IDENTIFIER")))
        node.add(self.leaf(self.consume("ASSIGN")))
        node.add(self.parse_expr())
        node.add(self.leaf(self.consume("SEMICOLON")))
        return node

    # if_stmt -> if ( bool_expr ) block [ else block ]
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

    # while_stmt -> while ( bool_expr ) block
    def parse_while_stmt(self) -> ParseNode:
        node = ParseNode("while_stmt")
        node.add(self.leaf(self.consume("WHILE")))
        node.add(self.leaf(self.consume("LPAREN")))
        node.add(self.parse_bool_expr())
        node.add(self.leaf(self.consume("RPAREN")))
        node.add(self.parse_block())
        return node

    # print_stmt -> print ( expr ) ;
    def parse_print_stmt(self) -> ParseNode:
        node = ParseNode("print_stmt")
        node.add(self.leaf(self.consume("PRINT")))
        node.add(self.leaf(self.consume("LPAREN")))
        node.add(self.parse_expr())
        node.add(self.leaf(self.consume("RPAREN")))
        # Detect and report a missing semicolon immediately, before the error
        # can be masked by a later cascade (e.g. an EOF/RBRACE mismatch that
        # causes _synchronise() to bail out before this statement is reached).
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

    # block -> { stmt_list }
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

    # bool_expr -> bool_expr || bool_term | bool_term
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

    # bool_term -> bool_term && bool_factor | bool_factor
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

    # bool_factor -> ! bool_factor | ( bool_expr ) | rel_expr
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
        """
        Decide whether the current '(' opens a bool sub-expression.
        Returns False immediately on LBRACE/EOF (unclosed parens).
        Hard-limited to MAX_LOOKAHEAD tokens.
        """
        depth = 0
        i     = self.pos
        limit = min(self.pos + MAX_LOOKAHEAD, len(self.tokens))
        while i < limit:
            t = self.tokens[i].type
            if t in ("LBRACE", "EOF"):
                return False          # parens never closed — bail out
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

    # rel_expr -> expr rel_op expr
    def parse_rel_expr(self) -> ParseNode:
        node = ParseNode("rel_expr")
        self._assert_expr_start()
        node.add(self.parse_expr())
        node.add(self.parse_rel_op())
        self._assert_expr_start()
        node.add(self.parse_expr())
        return node

    # rel_op -> < | > | <= | >= | == | !=
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

    # expr -> expr + term | expr - term | term
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

    # term -> term * factor | term / factor | term % factor | factor
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

    # factor -> ( expr ) | INTEGER | FLOAT | ID | - factor
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

    # ── Parse (with panic-mode recovery) ─────────
    tree   = None
    parser = Parser(tokens)
    try:
        tree = parser.parse_program()
    except ParseError as e:
        parser.errors.append(str(e))

    W = 70

    # ── Report ALL syntax errors — no partial tree ──
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

    # ── CFG ──────────────────────────────────────
    print("\n" + "=" * W)
    print("  CONTEXT-FREE GRAMMAR (CFG)")
    print("=" * W)
    print("""\
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
  TYPE        -> int | float""")
    print("=" * W)

    # ── Statement selection ───────────────────────
    all_stmts = collect_valid_stmts(tree)

    if not all_stmts:
        print("\n  No valid statements available for derivation.\n")
        sys.exit(0)

    print("\n" + "=" * W)
    print("  SELECT A STATEMENT for LMD / RMD / Parse Tree")
    print("=" * W)
    for idx, s in enumerate(all_stmts):
        inner = s.children[0]
        print(f"  [{idx + 1}]  {inner.label:<15}  {stmt_source(s)}")
    print("=" * W)

    while True:
        try:
            raw    = input("\n  Enter statement number: ").strip()
            choice = int(raw)
            if 1 <= choice <= len(all_stmts):
                break
            print(f"  Please enter a number between 1 and {len(all_stmts)}.")
        except ValueError:
            print("  Invalid input — enter a number.")

    selected_stmt  = all_stmts[choice - 1]
    selected_inner = selected_stmt.children[0]
    selected_text  = stmt_source(selected_stmt)

    print("\n" + "=" * W)
    print("  SELECTED STATEMENT")
    print("=" * W)
    print(f"  {selected_text}")
    print("=" * W)

    lm_steps = leftmost_derivation(selected_stmt)
    print_derivation(lm_steps, "LEFTMOST")

    rm_steps = rightmost_derivation(selected_stmt)
    print_derivation(rm_steps, "RIGHTMOST")

    print("\n" + "=" * W)
    print(f"  PARSE TREE — {selected_inner.label}  [ {selected_text} ]")
    print("=" * W)
    print(f"  {selected_stmt.label}")
    for i, child in enumerate(selected_stmt.children):
        print(child.pretty("  ", i == len(selected_stmt.children) - 1))
    print("=" * W)

    print(f"\n  ✔  Parse successful. No syntax errors.")
    print(f"     Tokens consumed : {len(tokens)}\n")


if __name__ == "__main__":
    main()