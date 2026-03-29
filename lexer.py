"""
Lexical Analyzer
Tokenizes source code into a stream of tokens.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────────
#  Token definitions
# ─────────────────────────────────────────────

TOKEN_TYPES = [
    # Multi-char operators (must come before single-char)
    ("FLOAT_LITERAL",  r'\d+\.\d+'),
    ("INT_LITERAL",    r'\d+'),
    ("AND",            r'&&'),
    ("OR",             r'\|\|'),
    ("LEQ",            r'<='),
    ("GEQ",            r'>='),
    ("EQ",             r'=='),
    ("NEQ",            r'!='),
    # Single-char operators
    ("NOT",            r'!'),
    ("LT",             r'<'),
    ("GT",             r'>'),
    ("ASSIGN",         r'='),
    ("PLUS",           r'\+'),
    ("MINUS",          r'-'),
    ("STAR",           r'\*'),
    ("SLASH",          r'/'),
    ("PERCENT",        r'%'),
    # Delimiters
    ("LPAREN",         r'\('),
    ("RPAREN",         r'\)'),
    ("LBRACE",         r'\{'),
    ("RBRACE",         r'\}'),
    ("SEMICOLON",      r';'),
    # Keywords / identifiers (keywords matched first via post-processing)
    ("IDENTIFIER",     r'[a-zA-Z_][a-zA-Z0-9_]*'),
    # Whitespace / comments (skipped)
    ("NEWLINE",        r'\n'),
    ("SKIP",           r'[ \t\r]+'),
    ("COMMENT",        r'//[^\n]*'),
    ("MCOMMENT",       r'/\[\s\S]?\*/'),
    # Catch-all for unknown characters
    ("UNKNOWN",        r'.'),
]

KEYWORDS = {"int", "float", "if", "else", "while", "print", "true", "false"}

# Compile a single master regex
MASTER_PATTERN = re.compile(
    '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_TYPES)
)


@dataclass
class Token:
    type: str
    value: str
    line: int
    col: int

    def _repr_(self):
        return f"Token({self.type:15s} | {self.value!r:15s} | line {self.line}, col {self.col})"


class LexerError(Exception):
    pass


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.errors: List[str] = []

    def tokenize(self) -> List[Token]:
        line = 1
        line_start = 0

        for mo in MASTER_PATTERN.finditer(self.source):
            kind  = mo.lastgroup
            value = mo.group()
            col   = mo.start() - line_start + 1

            # Track line numbers
            if kind == "NEWLINE":
                line += 1
                line_start = mo.end()
                continue
            if kind in ("SKIP", "COMMENT", "MCOMMENT"):
                # Count newlines inside multi-line comments
                newlines = value.count('\n')
                if newlines:
                    line += newlines
                    line_start = mo.start() + value.rfind('\n') + 1
                continue
            if kind == "UNKNOWN":
                err = f"LexicalError at line {line}, col {col}: unexpected character {value!r}"
                self.errors.append(err)
                continue

            # Promote keywords
            if kind == "IDENTIFIER" and value in KEYWORDS:
                kind = value.upper()        # e.g. "int" → "INT"

            self.tokens.append(Token(kind, value, line, col))

        self.tokens.append(Token("EOF", "", line, 0))
        return self.tokens

    def print_token_stream(self):
        print("\n" + "="*60)
        print("  TOKEN STREAM")
        print("="*60)
        print(f"  {'TYPE':<18} {'VALUE':<18} {'LINE':>5}  {'COL':>4}")
        print("-"*60)
        for tok in self.tokens:
            print(f"  {tok.type:<18} {tok.value!r:<18} {tok.line:>5}  {tok.col:>4}")
        print("="*60)
        if self.errors:
            print("\nLEXICAL ERRORS:")
            for e in self.errors:
                print(" ", e)
        print()
# ─────────────────────────────────────────────
#  Standalone mode
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lexer.py <source_file>")
        print("Example: python lexer.py test_program.src")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename, "r") as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)

    print(f"\n  Source file: {filename}")

    print("\n" + "="*60)
    print("  REGULAR EXPRESSIONS FOR EACH TOKEN CLASS")
    print("="*60)
    print(f"  {'TOKEN TYPE':<18} {'REGEX'}")
    print("-"*60)
    for name, pattern in TOKEN_TYPES:
        if name not in ("NEWLINE", "SKIP", "UNKNOWN"):
            print(f"  {name:<18} {pattern}")
    print("="*60)

    lexer = Lexer(source)
    lexer.tokenize()
    lexer.print_token_stream()

    if lexer.errors:
        print(f"  {len(lexer.errors)} lexical error(s) found.")
    else:
        print(f"  ✔ No lexical errors. Total tokens: {len(lexer.tokens)}")