import os
import sys
import chess
import time
import datetime
import csv
import statistics
import shutil

from .uci_engine import UCIEngine
from .stockfish_filter import StockfishFilter
from .mcts_core import MCTS, MCTSNode

# EDIT path ini sesuai lokasi stockfish.exe kamu
STOCKFISH_PATH = r"D:\stockfish-windows-x86-64-avx2\stockfish-windows-x86-64-avx2.exe"

MAX_DEPTH = 8          # batas depth sesuai proposal (<= 8 ply)
N_SIMULATIONS = 200    # naikkan nanti jadi 800, 1600 saat eksperimen
THRESHOLD_CP = -100    # pruning langkah sangat buruk
TOP_K = 8              # hanya evaluasi beberapa kandidat teratas
USE_MOVETIME = False   # False = pakai depth fix
SF_MOVETIME_MS = 20

TEST_FENS = [
    # posisi awal
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # midgame contoh
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10",
]

def run_demo():
    # 1. start engine stockfish
    sf = UCIEngine(STOCKFISH_PATH, name="stockfish")
    sf.setoption("Threads", "1")

    # 2. buat filter stockfish
    sf_filter = StockfishFilter(
        engine=sf,
        threshold_cp=THRESHOLD_CP,
        top_k=TOP_K,
        use_movetime=USE_MOVETIME,
        movetime_ms=SF_MOVETIME_MS,
        depth=MAX_DEPTH,
    )

    # 3. buat MCTS hybrid
    mcts = MCTS(sf_filter=sf_filter, ucb_c=1.5)

    # 4. jalankan untuk tiap FEN
    for fen in TEST_FENS:
        print("=================================")
        print("FEN:", fen)

        board = chess.Board(fen)
        root = MCTSNode(board=board)

        # quick micro-benchmark: measure average uncached Stockfish eval (T_sf)
        sf_T = None
        try:
            repeats = 3
            times = []
            for _ in range(repeats):
                # clear cache to force engine evaluation
                try:
                    sf_filter.cache.clear()
                except Exception:
                    pass
                t0 = time.perf_counter()
                _ = sf_filter.eval_board(board)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            sf_T = statistics.mean(times) if times else None
        except Exception:
            sf_T = None

        # jalankan simulasi (measure observed runtime)
        t_start = time.perf_counter()
        mcts.run_simulations(root, n_sim=N_SIMULATIONS)
        t_end = time.perf_counter()
        observed_total_s = t_end - t_start
        observed_per_sim_s = observed_total_s / N_SIMULATIONS if N_SIMULATIONS else None

        # langkah terbaik menurut hybrid
        bm = mcts.best_move(root)
        if bm is None:
            print("No legal move / terminal position.")
        else:
            print("Hybrid best move:", bm.uci())
            # Log time-complexity info for this search run as CSV
            try:
                logs_dir = os.path.join(os.getcwd(), "logs")
                os.makedirs(logs_dir, exist_ok=True)
                csv_path = os.path.join(logs_dir, "run_log.csv")

                ts = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
                # approximate branching factor at root
                b = len(root.legal_moves())

                fieldnames = [
                    "timestamp",
                    "fen",
                    "n_sim",
                    "top_k",
                    "depth",
                    "threshold_cp",
                    "branching_factor_at_root",
                    "measured_T_sf_s",
                    "observed_total_s",
                    "observed_per_sim_s",
                    "estimated_time_per_sim_s",
                    "estimated_total_s",
                    "bestmove_hybrid",
                    "cp_hybrid",
                    "bestmove_stockfish",
                    "cp_stockfish",
                    "theoretical_time_per_sim",
                    "theoretical_total_time",
                ]

                write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

                with open(csv_path, "a", encoding="utf-8", newline="") as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    est_time_per_sim = None
                    est_total = None
                    try:
                        if sf_T is not None:
                            est_time_per_sim = float(sf_T) * float(TOP_K)
                            est_total = est_time_per_sim * float(N_SIMULATIONS)
                    except Exception:
                        est_time_per_sim = None
                        est_total = None

                    # compute Stockfish's standalone best move & cp for comparison
                    best_sf_move = ""
                    cp_sf = ""
                    try:
                        # set position and ask stockfish for its best move using same mode as filter
                        sf.position_fen(fen)
                        cp_sf_val, mate_sf, best_sf = sf.go_and_get(
                            movetime_ms=SF_MOVETIME_MS if USE_MOVETIME else None,
                            depth=None if USE_MOVETIME else MAX_DEPTH,
                        )
                        best_sf_move = best_sf
                        cp_sf = cp_sf_val
                    except Exception:
                        best_sf_move = ""
                        cp_sf = ""

                    # compute cp for hybrid best move by evaluating the resulting position
                    best_hyb_move = bm.uci() if bm is not None else ""
                    cp_hyb = ""
                    try:
                        if bm is not None:
                            tmp = board.copy()
                            tmp.push(bm)
                            cp_hyb = sf_filter.eval_board(tmp)
                    except Exception:
                        cp_hyb = ""

                    writer.writerow({
                        "timestamp": ts,
                        "fen": fen,
                        "n_sim": N_SIMULATIONS,
                        "top_k": TOP_K,
                        "depth": MAX_DEPTH,
                        "threshold_cp": THRESHOLD_CP,
                        "branching_factor_at_root": b,
                        "measured_T_sf_s": round(sf_T, 6) if sf_T is not None else "",
                        "observed_total_s": round(observed_total_s, 6) if observed_total_s is not None else "",
                        "observed_per_sim_s": round(observed_per_sim_s, 6) if observed_per_sim_s is not None else "",
                        "estimated_time_per_sim_s": round(est_time_per_sim, 6) if est_time_per_sim is not None else "",
                        "estimated_total_s": round(est_total, 6) if est_total is not None else "",
                        "bestmove_hybrid": best_hyb_move,
                        "cp_hybrid": cp_hyb,
                        "bestmove_stockfish": best_sf_move,
                        "cp_stockfish": cp_sf,
                        "theoretical_time_per_sim": "O(b log b + top_k * T_sf + d)",
                        "theoretical_total_time": "O(n_sim * (b log b + top_k * T_sf + d))",
                    })
            except Exception:
                # don't fail the demo if logging fails
                pass

    # 5. shutdown engine
    sf.quit()

if __name__ == "__main__":
    def locate_stockfish():
        # Candidates: configured path, env var, PATH, common engines folder
        candidates = []
        if STOCKFISH_PATH:
            candidates.append(STOCKFISH_PATH)
        env_path = os.environ.get("STOCKFISH_PATH") or os.environ.get("STOCKFISH_BIN")
        if env_path:
            candidates.append(env_path)
        # which() can return None
        candidates.append(shutil.which("stockfish"))
        candidates.append(shutil.which("stockfish.exe"))
        # common repo engines location
        candidates.append(os.path.join(os.getcwd(), "engines", "stockfish", "stockfish"))
        candidates.append(os.path.join(os.getcwd(), "engines", "stockfish", "stockfish.exe"))

        for c in candidates:
            try:
                if c and os.path.exists(c):
                    return c
            except Exception:
                continue
        return None

    found = locate_stockfish()
    if not found:
        print("ERROR: STOCKFISH not found. Set the correct path in STOCKFISH_PATH or set the environment variable STOCKFISH_PATH/STOCKFISH_BIN, or install stockfish on PATH.")
        sys.exit(1)

    # override with discovered path and run demo
    STOCKFISH_PATH = found
    run_demo()