import chess
from uci_engine import UCIEngine
from stockfish_filter import StockfishFilter

STOCKFISH_PATH = r"D:\path\to\stockfish.exe"  # path kamu

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

engine = UCIEngine(STOCKFISH_PATH)
engine.setoption("Threads", "1")

flt = StockfishFilter(
    engine=engine,
    threshold_cp=-100,  # buang langkah yg sangat buruk
    top_k=8,
    use_movetime=False,
    depth=8
)

board = chess.Board(fen)
moves = list(board.legal_moves)

# priors uniform sementara
priors = {m: 1/len(moves) for m in moves}

kept = flt.filter_moves(board, moves, priors)

print("All legal moves:")
print([m.uci() for m in moves])
print("\nMoves kept after Stockfish filter:")
print([m.uci() for m in kept])

engine.quit()
