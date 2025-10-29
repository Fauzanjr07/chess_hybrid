import math
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import chess

from .stockfish_filter import StockfishFilter

# ---- Prior Provider ----
# Untuk sekarang masih uniform.
# Nanti kalau kamu mau pakai Lc0 sebagai policy prior,
# kamu bisa ganti ini dengan output probabilitas langkah dari Lc0.
class PriorProvider:
    def __init__(self):
        pass

    def get_prior(self, board: chess.Board, moves: List[chess.Move]) -> Dict[chess.Move, float]:
        if not moves:
            return {}
        p = 1.0 / len(moves)
        return {m: p for m in moves}

# ---- Node Struktur untuk MCTS ----
@dataclass
class MCTSNode:
    board: chess.Board
    parent: Optional["MCTSNode"] = None
    move: Optional[chess.Move] = None  # langkah dari parent -> node ini
    children: Dict[chess.Move, "MCTSNode"] = field(default_factory=dict)

    N: int = 0      # visit count
    W: float = 0.0  # total value (dari perspektif player yg akan jalan di parent)
    P: Dict[chess.Move, float] = field(default_factory=dict)

    terminal: bool = False

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def Q(self, mv: chess.Move) -> float:
        """
        Nilai rata-rata dari child move tsb.
        Kalau belum pernah dikunjungi, return 0.0
        """
        child = self.children.get(mv)
        if (not child) or child.N == 0:
            return 0.0
        return child.W / child.N

    def N_move(self, mv: chess.Move) -> int:
        """
        Berapa kali move ini dieksplor.
        """
        child = self.children.get(mv)
        return child.N if child else 0

# ---- MCTS Inti ----
class MCTS:
    def __init__(self, sf_filter: StockfishFilter, ucb_c: float = 1.5):
        self.sf_filter = sf_filter
        self.ucb_c = ucb_c
        self.priors = PriorProvider()

    # Time complexity notes:
    # - Let b be the average branching factor (legal moves per position).
    # - Let d be the average depth traversed during selection/expansion per simulation.
    # - let top_k be the number of moves evaluated by the StockfishFilter.
    # - Let T_sf be the cost of a single Stockfish evaluation (depends on engine and depth; typically exponential in search depth).
    #
    # Per simulation (one MCTS iteration) the main costs are:
    #  1) Selection: sorting legal moves by UCB -> O(b log b) to sort, then calling StockfishFilter
    #  2) StockfishFilter.filter_moves: evaluates up to top_k moves -> O(top_k * T_sf)
    #  3) Recursing down the tree during selection/expansion roughly d steps -> O(d) (not counting SF evals)
    #  4) Backpropagation: O(d)
    #
    # Therefore, time per simulation = O(b log b + top_k * T_sf + d).
    # For n_sim simulations: O(n_sim * (b log b + top_k * T_sf + d)).
    # In practice the dominant term is top_k * T_sf if Stockfish is invoked at each selection step.

    def _ucb_score(self, node: MCTSNode, mv: chess.Move) -> float:
        """
        UCB = Q + C * P * sqrt(N_parent) / (1 + N_move)
        """
        Q = node.Q(mv)
        P = node.P.get(mv, 0.0)
        N_parent = max(1, node.N)
        N_child  = node.N_move(mv)
        return Q + self.ucb_c * P * math.sqrt(N_parent) / (1 + N_child)

    def _select_child_filtered(self, node: MCTSNode) -> Optional[chess.Move]:
        """
        Tahap Selection:
        1. Urutkan move berdasarkan skor UCB (ekspansi MCTS).
        2. Kirim kandidat ke StockfishFilter untuk buang langkah buruk.
        3. Ambil langkah terbaik pertama yang masih 'boleh'.
        Ini bagian yang bikin sistemmu 'hybrid MCTS + Alpha-Beta'.
        """
        legal = node.legal_moves()
        if not legal:
            return None

        # Pastikan prior tersedia
        if not node.P:
            node.P = self.priors.get_prior(node.board, legal)

        # Urutkan moves dari yang paling menjanjikan menurut UCB
        sorted_by_ucb = sorted(
            legal,
            key=lambda mv: self._ucb_score(node, mv),
            reverse=True
        )

        # Minta filter Stockfish untuk prune langkah jelek
        kept = self.sf_filter.filter_moves(
            node.board,
            sorted_by_ucb,
            node.P
        )

        if not kept:
            return None

        # pilih langkah pertama yang lolos filter
        for mv in sorted_by_ucb:
            if mv in kept:
                return mv

        # fallback (harusnya jarang kejadian)
        return kept[0]

    def _rollout_value(self, node: MCTSNode) -> float:
        """
        Rollout/value function.
        Versi paling sederhana:
        - Jika posisi terminal, kasih +1 / -1 / 0.
        - Kalau belum terminal, kita pakai 0 sementara.
        Catatan:
        Value selalu dari sudut pandang side-to-move di node ini.
        """
        if node.board.is_game_over():
            res = node.board.result()  # "1-0", "0-1", "1/2-1/2"
            if res == "1-0":
                return 1.0 if node.board.turn == chess.WHITE else -1.0
            elif res == "0-1":
                return 1.0 if node.board.turn == chess.BLACK else -1.0
            else:
                return 0.0
        # placeholder: nanti bisa diganti static eval cepat
        return 0.0

    def _backpropagate(self, leaf: MCTSNode, value: float):
        """
        Kembali dari leaf ke root, update:
        - N (visit count)
        - W (total value)
        dan bolak-balik tanda value karena giliran pemain berganti.
        """
        node = leaf
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            v = -v
            node = node.parent

    def run_simulations(self, root: MCTSNode, n_sim: int = 200):
        for _ in range(n_sim):
            node = root

            # ===== 1. SELECTION (dengan filter Stockfish) =====
            while True:
                # kalau posisi game over, stop
                if node.board.is_game_over():
                    node.terminal = True
                    break

                mv = self._select_child_filtered(node)
                if mv is None:
                    # tidak ada langkah yang lolos filter
                    break

                # node child belum pernah dibuat → ini frontier
                if mv not in node.children:
                    new_board = node.board.copy()
                    new_board.push(mv)
                    child = MCTSNode(board=new_board, parent=node, move=mv)
                    node.children[mv] = child
                    node = child
                    break
                else:
                    # sudah ada, lanjut lebih dalam
                    node = node.children[mv]

            # ===== 2. (EXPANSION implicit di atas saat bikin child baru) =====

            # ===== 3. ROLLOUT / STATIC EVAL =====
            value = self._rollout_value(node)

            # ===== 4. BACKPROP =====
            self._backpropagate(node, value)

    def best_move(self, root: MCTSNode) -> Optional[chess.Move]:
        """
        Ambil langkah dengan visit count terbesar.
        Ini langkah yang nanti kamu klaim sebagai
        keputusan akhir sistem hybrid.
        """
        if not root.children:
            return None
        return max(root.children.items(), key=lambda kv: kv[1].N)[0]
