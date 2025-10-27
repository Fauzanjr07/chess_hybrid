import math
from typing import Dict, Optional, List, Iterable
import chess
from dataclasses import dataclass, field
from .stockfish_filter import StockfishFilter

# provider prior (sementara uniform)
class PriorProvider:
    def __init__(self):
        pass

    def get_prior(self, board: chess.Board, moves: List[chess.Move]) -> Dict[chess.Move, float]:
        if not moves:
            return {}
        p = 1.0 / len(moves)
        return {m: p for m in moves}

@dataclass
class MCTSNode:
    board: chess.Board
    parent: Optional["MCTSNode"] = None
    move: Optional[chess.Move] = None  # langkah dari parent -> node ini
    children: Dict[chess.Move, "MCTSNode"] = field(default_factory=dict)
    N: int = 0     # visit count
    W: float = 0.0 # total value dari sudut pandang pemain di parent
    P: Dict[chess.Move, float] = field(default_factory=dict)
    terminal: bool = False

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def Q(self, mv: chess.Move) -> float:
        child = self.children.get(mv)
        if (not child) or child.N == 0:
            return 0.0
        return child.W / child.N

    def N_move(self, mv: chess.Move) -> int:
        child = self.children.get(mv)
        return child.N if child else 0

class MCTS:
    def __init__(self, sf_filter: StockfishFilter, ucb_c: float = 1.5):
        self.sf_filter = sf_filter
        self.ucb_c = ucb_c
        self.prior_provider = PriorProvider()

    def _ucb_score(self, node: MCTSNode, mv: chess.Move) -> float:
        Q = node.Q(mv)
        P = node.P.get(mv, 0.0)
        total_N = max(1, node.N)
        n_mv = node.N_move(mv)
        return Q + self.ucb_c * P * math.sqrt(total_N) / (1 + n_mv)

    def _select_child_filtered(self, node: MCTSNode) -> Optional[chess.Move]:
        moves = node.legal_moves()
        if not moves:
            return None

        # ensure priors
        if not node.P:
            node.P = self.prior_provider.get_prior(node.board, moves)

        # urutkan moves berdasarkan skor UCB
        sorted_by_ucb = sorted(moves, key=lambda mv: self._ucb_score(node, mv), reverse=True)

        # minta StockfishFilter prune langkah jelek
        kept = self.sf_filter.filter_moves(node.board, sorted_by_ucb, node.P)

        if not kept:
            return None

        # pilih langkah terbaik pertama yang lolos filter
        for mv in sorted_by_ucb:
            if mv in kept:
                return mv
        return kept[0]

    def _rollout_value(self, node: MCTSNode) -> float:
        """
        Sederhana: kalau game sudah selesai, nilai +1/-1/0.
        Kalau belum selesai, 0.0 (placeholder).
        Nilai selalu dari sudut pandang side-to-move di node tsb.
        """
        if node.board.is_game_over():
            res = node.board.result()  # "1-0", "0-1", "1/2-1/2"
            if res == "1-0":
                return 1.0 if node.board.turn == chess.WHITE else -1.0
            elif res == "0-1":
                return 1.0 if node.board.turn == chess.BLACK else -1.0
            else:
                return 0.0
        return 0.0

    def _backpropagate(self, leaf: MCTSNode, value: float):
        node = leaf
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            # ganti tanda value setiap naik level (ganti perspektif pemain)
            v = -v
            node = node.parent

    def run_simulations(self, root: MCTSNode, n_sim: int = 200):
        for _ in range(n_sim):
            node = root

            # 1. SELECTION
            path = [node]
            while True:
                if node.board.is_game_over():
                    node.terminal = True
                    break
                mv = self._select_child_filtered(node)
                if mv is None:
                    break
                if mv not in node.children:
                    # frontier baru, break untuk expansion
                    child_board = node.board.copy()
                    child_board.push(mv)
                    child = MCTSNode(board=child_board, parent=node, move=mv)
                    node.children[mv] = child
                    node = child
                    path.append(node)
                    break
                else:
                    node = node.children[mv]
                    path.append(node)

            # 2. (optional) EXPANSION sudah dilakukan inline saat frontier

            # 3. ROLLOUT
            val = self._rollout_value(node)

            # 4. BACKPROP
            self._backpropagate(node, val)

    def best_move(self, root: MCTSNode) -> Optional[chess.Move]:
        if not root.children:
            return None
        # pilih child dengan N (visit count) terbesar
        return max(root.children.items(), key=lambda kv: kv[1].N)[0]
