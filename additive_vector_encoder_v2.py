"""
多次元加算ベクトルエンコーダー（Fraction × ケイリー・ハミルトン法）

原理:
  基底行列 B (n×n)、重みベクトル w
  エンコード : v = B @ w         (合成)
  加算操作   : v' = v + B @ Δw  (差分加算)
  デコード   : w = B⁻¹ @ v      (CH法 × Fraction で厳密逆行列)

  → 加算を何回重ねても復元誤差ゼロ
  → 実用上限: n ≤ 8（快適）、n ≤ 10（許容）
"""

from fractions import Fraction
import time, tracemalloc
import numpy as np


# ── Fraction行列演算 ─────────────────────────

def _I(n):
    return [[Fraction(1 if i==j else 0) for j in range(n)] for i in range(n)]

def _mul(A, B, n):
    C = [[Fraction(0)]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n): C[i][j] += A[i][k]*B[k][j]
    return C

def _add(A, B, n): return [[A[i][j]+B[i][j] for j in range(n)] for i in range(n)]
def _scl(s, A, n): return [[s*A[i][j] for j in range(n)] for i in range(n)]
def _tr(A, n):     return sum(A[i][i] for i in range(n))
def _mv(A, v, n):  return [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
def _fmat(A):      n=len(A); return [[Fraction(A[i][j]) for j in range(n)] for i in range(n)]
def _fvec(v):      return [Fraction(x) for x in v]


# ── CH逆行列（Fraction厳密版） ───────────────

def ch_inverse(A_raw):
    n = len(A_raw)
    A = _fmat(A_raw)
    I = _I(n); M = _I(n); coeffs = []
    for k in range(1, n+1):
        AM = _mul(A, M, n)
        ck = -Fraction(_tr(AM, n), k)
        coeffs.append(ck)
        M = _add(AM, _scl(ck, I, n), n)
    c0 = coeffs[-1]
    if c0 == 0: return None
    Q = _I(n)
    for k in range(n-1):
        Q = _add(_mul(A, Q, n), _scl(coeffs[k], I, n), n)
    return _scl(Fraction(-1)/c0, Q, n)


# ── エンコーダー本体 ─────────────────────────

class AdditiveVectorEncoder:
    """
    多次元加算ベクトルエンコーダー

    Parameters
    ----------
    basis : n×n の基底行列（整数リスト）
    """

    def __init__(self, basis):
        self.n   = len(basis)
        self.B   = _fmat(basis)
        self.Inv = ch_inverse(basis)
        if self.Inv is None:
            raise ValueError("基底行列が特異です")

    def encode(self, weights) -> list:
        """w → v = B @ w"""
        return _mv(self.B, _fvec(weights), self.n)

    def add(self, v, delta_w) -> list:
        """v' = v + B @ Δw"""
        dv = _mv(self.B, _fvec(delta_w), self.n)
        vf = v if isinstance(v[0], Fraction) else _fvec(v)
        return [vf[i]+dv[i] for i in range(self.n)]

    def decode(self, v) -> list:
        """w = B⁻¹ @ v  （厳密）"""
        vf = v if isinstance(v[0], Fraction) else _fvec(v)
        return _mv(self.Inv, vf, self.n)

    def inner_product(self, v, q) -> Fraction:
        """内積 vᵀ @ q"""
        vf = v if isinstance(v[0], Fraction) else _fvec(v)
        return sum(vf[i]*Fraction(q[i]) for i in range(self.n))


# ── リソース計測 ─────────────────────────────

def benchmark(n, seed=42):
    rng = np.random.default_rng(seed)
    B = (rng.integers(-3, 4, (n, n)) + np.eye(n)*n*2).tolist()

    w0      = [Fraction(i+1, i+2) for i in range(n)]
    dws     = [[Fraction(1,3)]*n, [Fraction(-1,2)]*n, [Fraction(2,5)]*n]
    w_cum   = [w0[i]+Fraction(1,3)+Fraction(-1,2)+Fraction(2,5) for i in range(n)]

    tracemalloc.start()
    t0 = time.perf_counter()

    enc = AdditiveVectorEncoder(B)
    v   = enc.encode(w0)
    for dw in dws: v = enc.add(v, dw)
    w_rec = enc.decode(v)

    ms = (time.perf_counter()-t0)*1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exact   = all(w_rec[i]==w_cum[i] for i in range(n))
    maxbits = max(max(x.numerator.bit_length(), x.denominator.bit_length())
                  for row in enc.Inv for x in row)
    return ms, peak/1024, exact, maxbits


# ── デモ実行 ─────────────────────────────────

def demo():
    print("="*62)
    print("  多次元ベクトルエンコーダー  Fraction × CH法")
    print("="*62)

    # ── 2次詳細例（x1=9/5, x2=7/5 の系と同じ基底）──
    print("\n【詳細デモ: 2次】")
    enc = AdditiveVectorEncoder([[2,1],[3,-1]])
    w0  = [Fraction(9,5), Fraction(7,5)]
    v   = enc.encode(w0)
    print(f"  初期重み  w = {w0}")
    print(f"  合成ベクトル v = B@w = {v}  ({[float(x) for x in v]})")

    dw = [Fraction(1,3), Fraction(-1,4)]
    v2 = enc.add(v, dw)
    w2 = enc.decode(v2)
    w2_true = [w0[i]+dw[i] for i in range(2)]
    print(f"  加算 Δw = {dw}")
    print(f"  加算後 v' = {v2}")
    print(f"  復元 w' = {w2}  （真値: {w2_true}）")
    print(f"  誤差 = {max(abs(w2[i]-w2_true[i]) for i in range(2))}  ← 厳密ゼロ ✓")
    print(f"  内積 v·[1,1] = {enc.inner_product(v, [1,1])}")

    # ── n次スケーリング表 ──
    print()
    print(f"  {'n':>3}  {'時間(ms)':>10}  {'メモリ(KB)':>11}  "
          f"{'逆行列ビット長':>14}  {'厳密復元':>8}  ゾーン")
    print(f"  {'':─>3}  {'':─>10}  {'':─>11}  {'':─>14}  {'':─>8}  {'':─>6}")

    zones = {**{n:"🟢 快適" for n in range(2,7)},
             **{n:"🟡 許容" for n in range(7,11)},
             **{n:"🔴 重い" for n in range(11,14)}}

    for n in range(2, 13):
        ms, kb, exact, bits = benchmark(n)
        mark = "✓ ゼロ" if exact else "× 誤差"
        zone = zones.get(n, "")
        print(f"  {n:>3}  {ms:>10.1f}  {kb:>11.1f}  {bits:>14}  {mark:>8}  {zone}")

    # ── 推奨まとめ ──
    print()
    print("─"*62)
    print("  推奨まとめ")
    print("─"*62)
    rows = [
        ("n ≤ 6",  "🟢 快適", "<20ms / <25KB",  "ヒル暗号・組み込み・教育用"),
        ("n ≤ 10", "🟡 許容", "<110ms / <65KB", "小規模NLP・制御系・秘密分散"),
        ("n ≤ 12", "🔴 重い", "<210ms / <90KB", "精度最優先・オフライン処理"),
        ("n > 12", "⛔ 非推奨","指数的増大",    "numpy/scipy に切替推奨"),
    ]
    for r in rows:
        print(f"  {r[0]:<8} {r[1]}  {r[2]:<20}  → {r[3]}")
    print("─"*62)
    print("  全次数で残差 = 厳密ゼロ（Fraction保証）")
    print("="*62)


if __name__ == "__main__":
    demo()
