from src.uci_engine import UCIEngine

STOCKFISH_PATH = r"D:\stockfish-windows-x86-64-avx2\stockfish-windows-x86-64-avx2.exe"
# ganti sesuai lokasi file engine kamu

def main():
    sf = UCIEngine(STOCKFISH_PATH, name="stockfish")
    sf.setoption("Threads", "1")

    # tes posisi awal
    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    sf.position_fen(startpos_fen)
    cp, mate, best = sf.go_and_get(depth=8)

    print("Evaluasi depth 8:")
    print("  cp score :", cp)
    print("  mate     :", mate)
    print("  bestmove :", best)

    sf.quit()

if __name__ == "__main__":
    main()
