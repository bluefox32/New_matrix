"""
Fraction（有理数）× ケイリー・ハミルトン法 精度検証
浮動小数点版・numpy版・Fraction版の3者比較
"""

from fractions import Fraction
import numpy as np


# ─────────────────────────────────────────
# 共通: 行列演算（Fraction対応）
# ─────────────────────────────────────────

def mat_mul(A, B, n):
    """n×n 行列積（Fraction / float 両対応）"""
    zero = type(A[0][0])(0)
    C = [[zero] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def mat_add(A, B, n):
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def mat_scale(s, A, n):
    return [[s * A[i][j] for j in range(n)] for i in range(n)]

def identity(n, dtype=Fraction):
    return [[dtype(1) if i == j else dtype(0) for j in range(n)] for i in range(n)]

def trace(A, n):
    return sum(A[i][i] for i in range(n))

def mat_vec(A, v, n):
    """行列×ベクトル"""
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]


# ─────────────────────────────────────────
# Faddeev-LeVerrier（Fraction版）
# ─────────────────────────────────────────

def faddeev_leverrier_frac(A_frac, n):
    """
    固有多項式の係数を Fraction で厳密計算
    returns coeffs: [c_{n-1}, c_{n-2}, ..., c_0]
    """
    I = identity(n, Fraction)
    M = identity(n, Fraction)
    coeffs = []

    for k in range(1, n + 1):
        AM = mat_mul(A_frac, M, n)
        ck = -Fraction(trace(AM, n), k)
        coeffs.append(ck)
        M = mat_add(AM, mat_scale(ck, I, n), n)

    return coeffs  # coeffs[-1] = c_0


# ─────────────────────────────────────────
# ケイリー・ハミルトン逆行列（Fraction版）
# ─────────────────────────────────────────

def ch_inverse_frac(A_frac, n):
    """A⁻¹ を Fraction の行列多項式で厳密計算"""
    coeffs = faddeev_leverrier_frac(A_frac, n)
    c0 = coeffs[-1]

    if c0 == 0:
        return None

    I = identity(n, Fraction)
    Q = [row[:] for row in I]
    for k in range(n - 1):
        Q = mat_add(mat_mul(A_frac, Q, n), mat_scale(coeffs[k], I, n), n)

    inv_c0 = Fraction(-1, 1) / c0
    return mat_scale(inv_c0, Q, n)


# ─────────────────────────────────────────
# 連立方程式を3通りで解く
# ─────────────────────────────────────────

def solve_all(A_raw, b_raw, label=""):
    n = len(b_raw)
    print(f"\n{'─'*55}")
    print(f"  {label}（{n}次）")
    print(f"{'─'*55}")

    # ── float版 CH ──
    A_f = np.array(A_raw, dtype=float)
    b_f = np.array(b_raw, dtype=float)

    coeffs_f = []
    I_f = np.eye(n)
    M_f = np.eye(n)
    for k in range(1, n + 1):
        AM = A_f @ M_f
        ck = -np.trace(AM) / k
        coeffs_f.append(ck)
        M_f = AM + ck * I_f

    c0_f = coeffs_f[-1]
    Q_f = np.eye(n)
    for k in range(n - 1):
        Q_f = A_f @ Q_f + coeffs_f[k] * I_f
    Ainv_f = (-1.0 / c0_f) * Q_f
    x_float = Ainv_f @ b_f

    # ── numpy解 ──
    x_np = np.linalg.solve(A_f, b_f)

    # ── Fraction版 CH ──
    A_frac = [[Fraction(A_raw[i][j]) for j in range(n)] for i in range(n)]
    b_frac = [Fraction(b_raw[i]) for i in range(n)]
    Ainv_frac = ch_inverse_frac(A_frac, n)
    if Ainv_frac is not None:
        x_frac = mat_vec(Ainv_frac, b_frac, n)
    else:
        x_frac = None

    # ── 表示 ──
    print(f"\n  {'変数':<6} {'float CH':>18} {'numpy':>18} {'Fraction CH':>20}")
    print(f"  {'':─<6} {'':─>18} {'':─>18} {'':─>20}")
    for i in range(n):
        fv  = f"{x_float[i]:.15f}"
        nv  = f"{x_np[i]:.15f}"
        frv = str(x_frac[i]) if x_frac else "—"
        print(f"  x{i+1:<5} {fv:>18} {nv:>18} {frv:>20}")

    # ── 残差比較 ──
    r_float = np.linalg.norm(A_f @ x_float - b_f)
    r_np    = np.linalg.norm(A_f @ x_np - b_f)
    if x_frac:
        Ax_frac = mat_vec(A_frac, x_frac, n)
        r_frac  = max(abs(Ax_frac[i] - b_frac[i]) for i in range(n))
    else:
        r_frac = None

    print(f"\n  残差ノルム:")
    print(f"    float CH   : {r_float:.2e}")
    print(f"    numpy      : {r_np:.2e}")
    print(f"    Fraction CH: {r_frac}  ← 厳密ゼロ" if r_frac == 0 else f"    Fraction CH: {r_frac}")

    # ── Fraction解を分数表記で表示 ──
    if x_frac:
        print(f"\n  厳密解（分数）:")
        for i, v in enumerate(x_frac):
            print(f"    x{i+1} = {v}", end="")
            if v.denominator != 1:
                print(f"  = {float(v):.10f}…（無限小数）")
            else:
                print(f"  （整数）")

    return x_frac


# ─────────────────────────────────────────
# デモ
# ─────────────────────────────────────────

def demo():
    print("=" * 55)
    print("  Fraction × ケイリー・ハミルトン 精度検証")
    print("=" * 55)

    # 例1: 2次（x1=9/5, x2=7/5 が既知の答え）
    solve_all(
        [[2,  1],
         [3, -1]],
        [5, 4],
        label="2次: x1=9/5, x2=7/5"
    )

    # 例2: 3次（整数解）
    solve_all(
        [[2,  1, -1],
         [-3,-1,  2],
         [-2, 1,  2]],
        [8, -11, -3],
        label="3次: x1=2, x2=3, x3=-1"
    )

    # 例3: 3次（分数解になる系）
    solve_all(
        [[3, 1,  2],
         [1, 4, -1],
         [2, 1,  3]],
        [1, 2, 3],
        label="3次: 分数解"
    )

    # 例4: 4次（分数解）
    solve_all(
        [[4, 1, 2, 0],
         [1, 3, 0, 1],
         [2, 0, 5, 1],
         [0, 1, 1, 4]],
        [1, 2, 3, 4],
        label="4次: 分数解"
    )

    print("\n" + "=" * 55)
    print("  まとめ:")
    print("    Fraction版は誤差ゼロで厳密解を保証")
    print("    ただし n が増えると分子・分母が巨大化")
    print("    → n ≤ 6 程度が実用的な上限の目安")
    print("=" * 55)


if __name__ == "__main__":
    demo()
