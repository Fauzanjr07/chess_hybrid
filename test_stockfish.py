import subprocess

proc = subprocess.Popen(
    ["./engines/stockfish-17.1/stockfish"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    universal_newlines=True,
    bufsize=1
)

proc.stdin.write("uci\n")
proc.stdin.flush()

for _ in range(20):
    line = proc.stdout.readline().strip()
    print(line)
    if "uciok" in line:
        break

proc.stdin.write("position startpos\n")
proc.stdin.write("go depth 8\n")
proc.stdin.flush()

for _ in range(30):
    print(proc.stdout.readline().strip())