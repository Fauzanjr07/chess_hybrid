"""
Hybrid MCTS + Stockfish Filter Skeleton
- Gunakan ini sebagai entry point teknis buat prototype.
"""

import os
import sys
import chess

from .uci_engine import UCIEngine
from .stockfish_filter import StockfishFilter
from .mcts_core import MCTS, MCTSNode

# ==== KONFIGURASI UTAMA ====
STOCKFISH_PATH = r"D:\stockfish-windows-x86-64-avx2\stockfish-windows-x86-64-avx2.exe"
# ↑ ganti dengan path stockfish.exe kamu yang sudah terbukti jalan

MAX_DEPTH = 8            # sesuai batasan depth <= 8 ply
N_SIMULATIONS = 200      # naikin nanti (800, 1600...) buat eksperimen serius
THRESHOLD_CP = -100      # prune langkah jelek
TOP_K = 8                # evaluasi hanya K langkah terbaik dulu
USE_MOVETIME = False     # False = pakai depth fix
SF_MOVETIME_MS = 20

EXAMPLE_FENS = [
    # posisi awal
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # midgame contoh
    "r2q1rk1/pp2bppp/2n1pn2/2bp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10",
]

def run_demo():
    # init engine stockfish
    sf = UCIEngine(STOCKFISH_PATH, name="stockfish")

    # (opsional) supaya cepat / konsisten
    sf.setoption("Threads", "1")

    sf_filter = StockfishFilter(
        engine=sf,
        threshold_cp=THRESHOLD_CP,
        top_k=TOP_K,
        use_movetime=USE_MOVETIME,
        movetime_ms=SF_MOVETIME_MS,
        depth=MAX_DEPTH,
    )

    mcts = MCTS(
        sf_filter=sf_filter,
        ucb_c=1.5,
    )

    for fen in EXAMPLE_FENS:
        print("=====================================")
        print("FEN :", fen)

        board = chess.Board(fen)
        root = MCTSNode(board=board)

        # jalankan simulasi MCTS hybrid
        mcts.run_simulations(root, n_sim=N_SIMULATIONS)

        # pilih langkah terbaik versi MCTS
        best = mcts.best_move(root)
        if best is None:
            print("No move found (terminal?)")
        else:
            print("Best move (UCI):", best.uci())

    sf.quit()


if __name__ == "__main__":
    if not os.path.exists(STOCKFISH_PATH):
        print("ERROR: STOCKFISH_PATH belum benar. Edit path di atas.")
        sys.exit(1)

    run_demo()
