"""
================================================================================
  ASM SIMULATOR  –  Executes the pseudo-assembly produced by phase8.py
================================================================================

  Supported instructions:
    MOV  dst, #imm      load immediate into dst
    MOV  dst, src       copy register/memory → dst
    ADD  dst, src       dst = dst + src
    SUB  dst, src       dst = dst - src
    MUL  dst, src       dst = dst * src
    DIV  dst, src       dst = dst // src  (integer div)
    MOD  dst, src       dst = dst %  src
    CMP  a, b           set flags from a - b  (or a - #imm)
    CMP  a, #imm        same with immediate
    JMP  label          unconditional jump
    JE   label          jump if equal        (flag == 0)
    JNE  label          jump if not equal    (flag != 0)
    JG   label          jump if greater      (flag >  0)
    JL   label          jump if less         (flag <  0)
    JGE  label          jump if >=           (flag >= 0)
    JLE  label          jump if <=           (flag <= 0)
    PUSH src            push value onto param stack
    CALL print          pop top of param stack → print it
    HALT                stop execution
    RET                 return from subroutine (used in print:)

  Usage:
      python asm_simulator.py              # reads output.asm by default
      python asm_simulator.py myfile.asm  # or specify a file
================================================================================
"""

import sys
import re
from typing import Dict, List, Optional, Tuple


# ==============================================================================
#  INSTRUCTION  (parsed representation of one line)
# ==============================================================================

class Instruction:
    def __init__(self, op: str, args: List[str], source: str):
        self.op     = op        # e.g. "MOV", "ADD", "JMP" …
        self.args   = args      # list of operand strings
        self.source = source    # original source line (for trace output)

    def __repr__(self):
        return f"Instruction({self.op}, {self.args})"


# ==============================================================================
#  PARSER  –  reads .asm text → list of instructions + label map
# ==============================================================================

def parse_asm(text: str) -> Tuple[List[Instruction], Dict[str, int]]:
    """
    Returns:
        instructions  – flat list of Instruction objects (labels excluded)
        labels        – { label_name : instruction_index }
    """
    instructions: List[Instruction] = []
    labels:       Dict[str, int]    = {}

    in_code = False   # ignore .DATA section

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Skip blank lines and comments ─────────────────────────────────────
        if not line or line.startswith(";"):
            continue

        # ── Section markers ───────────────────────────────────────────────────
        if line == ".DATA":
            in_code = False
            continue
        if line == ".CODE":
            in_code = True
            continue

        # ── .DATA declarations (e.g.  x .WORD 0) – skip ─────────────────────
        if not in_code:
            continue

        # ── Label definitions  (e.g. "L3:"  "main:"  "t1_true:") ─────────────
        if line.endswith(":"):
            label_name = line[:-1].strip()
            labels[label_name] = len(instructions)
            continue

        # ── Regular instruction ────────────────────────────────────────────────
        # Split on first space to get mnemonic, then split remainder on comma
        parts = line.split(None, 1)          # ["MOV", "R0, x"]
        op    = parts[0].upper()
        args  = []
        if len(parts) > 1:
            # split on comma, strip whitespace from each token
            args = [a.strip() for a in parts[1].split(",")]

        instructions.append(Instruction(op, args, raw_line))

    return instructions, labels


# ==============================================================================
#  SIMULATOR
# ==============================================================================

class Simulator:
    """
    Executes the pseudo-assembly instruction list produced by parse_asm().
    """

    MAX_STEPS = 100_000   # safety limit to catch infinite loops

    def __init__(self, instructions: List[Instruction],
                 labels: Dict[str, int], trace: bool = False):
        self.instructions = instructions
        self.labels       = labels
        self.trace        = trace

        # ── Machine state ──────────────────────────────────────────────────────
        self.memory: Dict[str, float] = {}   # variables + temporaries
        self.R0:     float            = 0    # general-purpose register
        self.flag:   float            = 0    # result of last CMP (a - b)
        self.pc:     int              = 0    # program counter (index into instructions)
        self.stack:  List[float]      = []   # param stack for CALL print
        self.output: List[str]        = []   # collected print() output
        self.halted: bool             = False
        self.steps:  int              = 0

    # ── Value resolution ───────────────────────────────────────────────────────

    def _read(self, operand: str) -> float:
        """Resolve an operand to a numeric value."""
        operand = operand.strip()
        # Immediate  #123  or  #3.14
        if operand.startswith("#"):
            return float(operand[1:])
        # Bare numeric literal without # (e.g. MOV R0, 2  or  CMP R0, 5.0)
        try:
            return float(operand)
        except ValueError:
            pass
        # R0 register
        if operand == "R0":
            return self.R0
        # Memory / variable
        return self.memory.get(operand, 0.0)

    def _write(self, dst: str, value: float):
        """Write value to dst (R0 or a named memory location)."""
        dst = dst.strip()
        if dst == "R0":
            self.R0 = value
        else:
            self.memory[dst] = value

    def _fmt(self, v: float) -> str:
        """Pretty-print a number: show as int if it is whole."""
        if v == int(v):
            return str(int(v))
        return str(v)

    # ── Single-step execution ──────────────────────────────────────────────────

    def _step(self):
        if self.pc >= len(self.instructions):
            self.halted = True
            return

        instr = self.instructions[self.pc]
        op    = instr.op
        args  = instr.args

        if self.trace:
            print(f"  [PC={self.pc:4d}]  {instr.source.strip()}")

        self.pc += 1   # advance BEFORE executing (jumps overwrite this)

        # ── MOV ───────────────────────────────────────────────────────────────
        if op == "MOV":
            dst, src = args[0], args[1]
            self._write(dst, self._read(src))

        # ── Arithmetic ────────────────────────────────────────────────────────
        elif op == "ADD":
            dst, src = args[0], args[1]
            self._write(dst, self._read(dst) + self._read(src))

        elif op == "SUB":
            dst, src = args[0], args[1]
            self._write(dst, self._read(dst) - self._read(src))

        elif op == "MUL":
            dst, src = args[0], args[1]
            self._write(dst, self._read(dst) * self._read(src))

        elif op == "DIV":
            dst, src = args[0], args[1]
            divisor = self._read(src)
            if divisor == 0:
                print("  [RUNTIME ERROR] Division by zero — halting.")
                self.halted = True
                return
            result = self._read(dst) / divisor
            # integer division if both operands are whole numbers
            if self._read(dst) == int(self._read(dst)) and divisor == int(divisor):
                result = int(self._read(dst)) // int(divisor)
            self._write(dst, result)

        elif op == "MOD":
            dst, src = args[0], args[1]
            divisor = self._read(src)
            if divisor == 0:
                print("  [RUNTIME ERROR] Modulo by zero — halting.")
                self.halted = True
                return
            self._write(dst, self._read(dst) % divisor)

        # ── CMP ───────────────────────────────────────────────────────────────
        elif op == "CMP":
            a = self._read(args[0])
            b = self._read(args[1])
            self.flag = a - b

        # ── Unconditional jump ────────────────────────────────────────────────
        elif op == "JMP":
            target = args[0]
            if target not in self.labels:
                print(f"  [RUNTIME ERROR] Undefined label '{target}' — halting.")
                self.halted = True
                return
            self.pc = self.labels[target]

        # ── Conditional jumps ─────────────────────────────────────────────────
        elif op == "JE":
            if self.flag == 0:
                self.pc = self.labels[args[0]]
        elif op == "JNE":
            if self.flag != 0:
                self.pc = self.labels[args[0]]
        elif op == "JG":
            if self.flag > 0:
                self.pc = self.labels[args[0]]
        elif op == "JL":
            if self.flag < 0:
                self.pc = self.labels[args[0]]
        elif op == "JGE":
            if self.flag >= 0:
                self.pc = self.labels[args[0]]
        elif op == "JLE":
            if self.flag <= 0:
                self.pc = self.labels[args[0]]

        # ── Stack / call ──────────────────────────────────────────────────────
        elif op == "PUSH":
            self.stack.append(self._read(args[0]))

        elif op == "CALL":
            fn = args[0].strip()
            if fn == "print":
                if self.stack:
                    val = self.stack.pop()
                    out = self._fmt(val)
                    self.output.append(out)
                    print(f"  print → {out}")
                else:
                    print("  print → (empty stack)")
            # For any other CALL: just skip (no user-defined functions supported)

        # ── HALT / RET ────────────────────────────────────────────────────────
        elif op in ("HALT", "RET"):
            self.halted = True

        else:
            # Unknown instruction — warn but continue
            print(f"  [WARNING] Unknown instruction: {op} — skipped.")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self):
        """Execute until HALT or MAX_STEPS reached."""
        # Jump to "main" label if it exists, otherwise start at 0
        if "main" in self.labels:
            self.pc = self.labels["main"]

        while not self.halted and self.steps < self.MAX_STEPS:
            self._step()
            self.steps += 1

        if self.steps >= self.MAX_STEPS:
            print(f"\n  [WARNING] Reached step limit ({self.MAX_STEPS}) — "
                  f"possible infinite loop, execution stopped.")


# ==============================================================================
#  PRETTY PRINTER  –  final memory dump
# ==============================================================================

def print_memory(sim: Simulator):
    W = 60
    print("\n" + "=" * W)
    print("  FINAL MEMORY STATE")
    print("=" * W)

    # Separate user variables (no underscore-digit suffix) from temporaries
    user_vars = {}
    temps     = {}

    for name, val in sorted(sim.memory.items()):
        # temporaries: t1, t2, t3 …  or  t1_true, t1_end (label helpers)
        if re.match(r'^t\d', name):
            temps[name] = val
        else:
            user_vars[name] = val

    if user_vars:
        print(f"\n  {'Variable':<16}  {'Value'}")
        print(f"  {'--------':<16}  {'-----'}")
        for name, val in sorted(user_vars.items()):
            print(f"  {name:<16}  {sim._fmt(val)}")

    if temps:
        print(f"\n  {'Temporary':<16}  {'Value'}")
        print(f"  {'---------':<16}  {'-----'}")
        for name, val in sorted(temps.items(),
                                key=lambda x: int(re.search(r'\d+', x[0]).group()
                                                  if re.search(r'\d+', x[0]) else 0)):
            print(f"  {name:<16}  {sim._fmt(val)}")

    print("=" * W)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    W = 60

    # ── Load file ─────────────────────────────────────────────────────────────
    filename = sys.argv[1] if len(sys.argv) > 1 else "output.asm"
    try:
        with open(filename) as f:
            asm_text = f.read()
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        print("Run phase8.py first to generate output.asm")
        sys.exit(1)

    # ── Ask for trace mode ────────────────────────────────────────────────────
    trace = False
    if len(sys.argv) > 2 and sys.argv[2] in ("-t", "--trace"):
        trace = True

    # ── Parse ─────────────────────────────────────────────────────────────────
    instructions, labels = parse_asm(asm_text)

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * W)
    print("  ASM SIMULATOR")
    print("=" * W)
    print(f"  File       : {filename}")
    print(f"  Instructions loaded : {len(instructions)}")
    print(f"  Labels found        : {list(labels.keys())}")
    print(f"  Trace mode : {'ON' if trace else 'OFF'  } "
          f"(pass --trace to enable)")
    print("=" * W)

    # ── Run ───────────────────────────────────────────────────────────────────
    sim = Simulator(instructions, labels, trace=trace)

    print("\n  --- EXECUTION OUTPUT ---\n")
    sim.run()

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n  --- EXECUTION COMPLETE ---")
    print(f"  Steps executed : {sim.steps}")

    if sim.output:
        print(f"\n  Program output ({len(sim.output)} value(s)):")
        for i, v in enumerate(sim.output, 1):
            print(f"    [{i}] {v}")
    else:
        print("\n  No output produced (no print statements executed).")

    print_memory(sim)


if __name__ == "__main__":
    main()