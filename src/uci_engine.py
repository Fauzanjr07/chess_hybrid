import subprocess
import time
import re
from typing import List, Tuple, Optional

class UCIEngine:
    """
    Minimal synchronous UCI wrapper.
    Assumes engine is a local executable (ex: stockfish.exe).
    """

    def __init__(self, path: str, name: str = "engine"):
        self.path = path
        self.name = name

        # start process
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # init UCI
        self._write("uci")
        self._read_until("uciok")

        # make sure it's ready
        self._write("isready")
        self._read_until("readyok")

    def _write(self, cmd: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str, timeout: float = 10.0) -> List[str]:
        assert self.proc.stdout is not None
        start = time.time()
        lines: List[str] = []
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"{self.name}: timeout waiting for '{token}'")
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.rstrip("\n")
            lines.append(line)
            if token in line:
                return lines

    def setoption(self, name: str, value: str):
        # e.g. set Threads, Hash, etc
        self._write(f"setoption name {name} value {value}")

    def ucinewgame(self):
        self._write("ucinewgame")
        self._write("isready")
        self._read_until("readyok")

    def position_fen(self, fen: str, moves: Optional[List[str]] = None):
        if moves:
            ms = " ".join(moves)
            self._write(f"position fen {fen} moves {ms}")
        else:
            self._write(f"position fen {fen}")

    def go_and_get(self,
                   movetime_ms: Optional[int] = None,
                   depth: Optional[int] = None
                   ) -> Tuple[int, Optional[int], str]:
        """
        Return:
          (cp_score, mate_score, bestmove)
        cp_score  : centipawn eval from side-to-move POV
        mate_score: moves to mate (if mate found) or None
        bestmove  : bestmove in UCI notation
        """
        # Note on time complexity:
        # The actual search work is performed by the external UCI engine (e.g. Stockfish).
        # Typical classical engines use alpha-beta/negamax search with various pruning and extensions.
        # Worst-case complexity is O(b^d) where b is branching factor and d is search depth.
        # With alpha-beta and good move ordering the effective complexity can approach O(b^(d/2)) in best cases.
        # When movetime_ms is used the engine decides how much depth to search within that time.
        # We denote a single engine evaluation cost as T_sf(depth) to emphasize it depends on chosen depth/time.

        if movetime_ms is not None:
            self._write(f"go movetime {movetime_ms}")
        elif depth is not None:
            self._write(f"go depth {depth}")
        else:
            self._write("go")

        lines = self._read_until("bestmove")
        bestmove = "(none)"
        cp_score: int = 0
        mate_score: Optional[int] = None
        score_re = re.compile(r"score (cp|mate) (-?\d+)")

        for line in lines:
            if line.startswith("info "):
                m = score_re.search(line)
                if m:
                    kind, val = m.group(1), int(m.group(2))
                    if kind == "cp":
                        cp_score = val
                        mate_score = None
                    else:
                        mate_score = val
                        # convert mate score into a very large cp magnitude so ordering still works
                        cp_score = 100000 if val > 0 else -100000
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]

        return cp_score, mate_score, bestmove

    def quit(self):
        try:
            self._write("quit")
        except Exception:
            pass
        if self.proc:
            self.proc.terminate()
