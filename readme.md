# chess_hybrid
Hybrid MCTS + Alpha-Beta (Stockfish) Evaluation
------------------------------------------------

Tujuan
- Sistem ini menggabungkan eksplorasi Monte Carlo Tree Search (MCTS)
  dengan evaluasi Stockfish berbasis Alpha-Beta sebagai filter langkah buruk.
- Implementasi ini sesuai proposal skripsi:
  depth pencarian dibatasi (<= 8 ply),
  input berupa posisi FEN,
  dan evaluasi kinerja pakai MAE, MSE, R², waktu komputasi, serta ELO.

Struktur Direktori
- `src/` : kode sumber (UCI wrapper, filter Stockfish, MCTS hybrid)
- `engines/` : binary Stockfish 17.1 dan (opsional) Lc0
- `data/` : dataset posisi FEN untuk eksperimen
- `logs/` : hasil eksperimen (tabel, waktu eksekusi, dsb)
- `weights/` : bobot evaluasi (NNUE Stockfish, policy/value Lc0)
- `notebooks/` : analisis MAE/MSE/R², plotting grafik untuk Bab 4
- `main.py` : entry point uji jalan

Versi Engine
- Stockfish 17.1
- Lc0 v0.31.2 (opsional, untuk prior/policy nanti)

Langkah Cepat Jalan
1. Pastikan `STOCKFISH_PATH` di `src/mcts_stockfish_hybrid_skeleton.py` menunjuk ke stockfish.exe yang benar.
2. Install dependency:
   ```bash
   python -m pip install -r requirements.txt
# chess_hybrid
