from typing import Dict, List
import chess
from .uci_engine import UCIEngine

class StockfishFilter:
    def __init__(
        self,
        engine: UCIEngine,
        threshold_cp: int = -100,
        top_k: int = 8,
        use_movetime: bool = False,
        movetime_ms: int = 20,
        depth: int = 8,
    ):
        """
        threshold_cp : langkah dengan eval < threshold_cp akan dipruning
        top_k        : hanya evaluasi K langkah teratas (hemat waktu)
        depth        : batas kedalaman pencarian (ply) -> sesuai batas proposal (<= 8)
        """
        self.engine = engine
        self.threshold_cp = threshold_cp
        self.top_k = top_k
        self.use_movetime = use_movetime
        self.movetime_ms = movetime_ms
        self.depth = depth
        self.cache: Dict[str, int] = {}  # caching evaluasi per-FEN

    def eval_board(self, board: chess.Board) -> int:
        fen = board.fen()
        if fen in self.cache:
            return self.cache[fen]

        self.engine.position_fen(fen)
        cp_score, mate_score, _ = self.engine.go_and_get(
            movetime_ms=self.movetime_ms if self.use_movetime else None,
            depth=None if self.use_movetime else self.depth
        )

        self.cache[fen] = cp_score
        return cp_score

    # Complexity notes for eval_board and filter_moves:
    # - eval_board: single Stockfish evaluation triggered via UCI. Cost depends on engine search mode
    #   (depth or movetime). Classical alpha-beta search is exponential in search depth, so we denote
    #   a single evaluation cost as T_sf (function of depth/movetime and engine internals).
    # - filter_moves:
    #    * ordering by prior: O(m log m) where m = len(candidate_moves)
    #    * evaluating top_k moves: O(top_k * T_sf)
    #    * overall: O(m log m + top_k * T_sf)
    #  In this project top_k is typically small (e.g. 8) to keep filter cost manageable.

    def filter_moves(
        self,
        board: chess.Board,
        candidate_moves: List[chess.Move],
        priors: Dict[chess.Move, float],
    ) -> List[chess.Move]:
        """
        Kembalikan subset dari candidate_moves yang "layak"
        menurut evaluasi Stockfish.
        """
        # urutkan kandidat berdasarkan prior (kalau ada), biar kita evaluasi top-K dulu
        if priors:
            ordered = sorted(candidate_moves, key=lambda m: priors.get(m, 0.0), reverse=True)
        else:
            ordered = candidate_moves[:]

        scored = []
        for mv in ordered[: self.top_k]:
            board.push(mv)
            cp_val = self.eval_board(board)
            board.pop()
            scored.append((mv, cp_val))

        kept = [mv for (mv, cp_val) in scored if cp_val >= self.threshold_cp]

        # fallback: kalau semua ke-prune, ambil move terbaik cp_val tertinggi
        if not kept and scored:
            kept = [max(scored, key=lambda t: t[1])[0]]

        return kept
