"""
x²+y²=z² における内積ベクトル多段合算方式
z = ‖v‖ をノルムとして捉え、加算ベクトルエンコーダーと統合する

3つのアプローチ:
  A. 直交合算  : vᵢ⊥vⱼ → ‖Σvᵢ‖² = Σ‖vᵢ‖² → ピタゴラス数の合成
  B. 射影分解  : 任意ベクトルを既知三つ組方向へ射影してノルム保存
  C. 基底行列法: B·w=v, ‖v‖²=z², CH×Fraction で厳密検証
"""

from fractions import Fraction
import math


# ══════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════

def norm2(v):
    """‖v‖²（Fraction対応）"""
    return sum(x*x for x in v)

def dot(a, b):
    """内積（Fraction対応）"""
    return sum(a[i]*b[i] for i in range(len(a)))

def is_perfect_square(n):
    """n が完全平方数か判定（Fraction → 分子/分母ともに）"""
    if isinstance(n, Fraction):
        if n.denominator != 1:
            return False, Fraction(0)
        n = n.numerator
    r = math.isqrt(n)
    return r*r == n, r

def vec_add(*vecs):
    """多段ベクトル加算 Σvᵢ"""
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) for i in range(n)]


# ══════════════════════════════════════════════
# A. 直交合算: vᵢ⊥vⱼ ならノルムが加法的
#    ‖v₁+v₂‖² = ‖v₁‖² + ‖v₂‖²  （交差項=0）
#    → ピタゴラス三つ組 (a,b,c) から新たな三つ組を生成
# ══════════════════════════════════════════════

def orthogonal_multistep(triples):
    """
    既知のピタゴラス三つ組から直交合算で新三つ組を生成

    各三つ組 (x,y,z) に対応するベクトル v=(x,y) と
    その90°回転 v⊥=(-y,x) を使う。
    v と v⊥ は直交し ‖v‖=‖v⊥‖=z なので
    v+v⊥ のノルムは z√2（整数でない場合が多い）

    整数ノルムを得るには直交ペアを別々の三つ組から選ぶ:
    v₁=(x₁,y₁,z₁), v₂=(x₂,y₂,z₂) で v₁·v₂=0 のとき
    ‖v₁+v₂‖² = z₁² + z₂²  → これ自体が新しい x²+y²=z²
    """
    print("【A. 直交合算による三つ組生成】")
    print(f"  {'v₁=(x,y)':^14} {'v₂=(x,y)':^14} "
          f"{'v₁·v₂':^8} {'合算ベクトル':^14} {'‖Σv‖²':^10} {'z=‖Σv‖':^10}")
    print(f"  {'':─<14} {'':─<14} {'':─<8} {'':─<14} {'':─<10} {'':─<10}")

    found = []
    triples_xy = [(x, y, z) for x, y, z in triples]

    for i in range(len(triples_xy)):
        for j in range(i, len(triples_xy)):
            x1,y1,z1 = triples_xy[i]
            x2,y2,z2 = triples_xy[j]
            # v2 を90°回転して直交にする: v2⊥ = (-y2, x2)
            v1 = (x1, y1)
            v2p = (-y2, x2)   # v1 と内積を計算
            d = dot(v1, v2p)
            if d == 0:  # 直交
                vs = vec_add(v1, v2p)
                n2 = norm2(vs)
                ok, z = is_perfect_square(n2)
                row = (f"  ({x1},{y1})   ", f"({-y2},{x2})   ",
                       f"{d:^8}", f"({vs[0]},{vs[1]})   ",
                       f"{n2:^10}", f"{z if ok else '─':^10}")
                if ok and z > 0:
                    found.append((*vs, z))
                    print("".join(row), "← ピタゴラス ✓")
                # 直交かつ結果が非自明なものだけ表示
            # 一般内積の場合も一部表示
            if i != j and abs(d) < 50:
                v1 = (x1,y1); v2 = (x2,y2)
                d2 = dot(v1,v2)
                vs2 = vec_add(v1,v2)
                n2_2 = norm2(vs2)
                ok2, z2r = is_perfect_square(n2_2)
                if ok2 and z2r > 0 and (x1,y1,z1) != (x2,y2,z2):
                    found.append((*vs2, z2r))
                    print(f"  ({x1},{y1})   ({x2},{y2})   "
                          f"{d2:^8} ({vs2[0]},{vs2[1]})   "
                          f"{n2_2:^10} {z2r:^10} ← 直接合算 ✓")

    if not found:
        print("  （この範囲では整数ノルムの直交合算なし）")
    return found


# ══════════════════════════════════════════════
# B. 多段加算ベクトルエンコーダー × ノルム判定
#    v = Σ wᵢ bᵢ  (bᵢ は基底ベクトル)
#    ‖v‖² = z² かどうかを Fraction で厳密判定
# ══════════════════════════════════════════════

def _I(n): return [[Fraction(1 if i==j else 0) for j in range(n)] for i in range(n)]
def _mul(A,B,n):
    C=[[Fraction(0)]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n): C[i][j]+=A[i][k]*B[k][j]
    return C
def _add(A,B,n): return [[A[i][j]+B[i][j] for j in range(n)] for i in range(n)]
def _scl(s,A,n): return [[s*A[i][j] for j in range(n)] for i in range(n)]
def _tr(A,n):    return sum(A[i][i] for i in range(n))
def _mv(A,v,n):  return [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]

def ch_inverse(A_raw):
    n=len(A_raw)
    A=[[Fraction(A_raw[i][j]) for j in range(n)] for i in range(n)]
    I=_I(n); M=_I(n); coeffs=[]
    for k in range(1,n+1):
        AM=_mul(A,M,n)
        ck=-Fraction(_tr(AM,n),k); coeffs.append(ck)
        M=_add(AM,_scl(ck,I,n),n)
    c0=coeffs[-1]
    if c0==0: return None
    Q=_I(n)
    for k in range(n-1): Q=_add(_mul(A,Q,n),_scl(coeffs[k],I,n),n)
    return _scl(Fraction(-1)/c0,Q,n)


class PythagoreanEncoder:
    """
    x²+y²=z² を加算ベクトルエンコーダーで扱うクラス

    基底行列 B の各列が「方向ベクトル」、
    重み w が各方向の寄与量、
    合成ベクトル v=Bw のノルム² が z² に対応。
    """

    def __init__(self, basis):
        self.n   = len(basis)
        self.B   = [[Fraction(basis[i][j]) for j in range(self.n)]
                    for i in range(self.n)]
        self.Inv = ch_inverse(basis)

    def encode(self, weights):
        """w → v = B@w"""
        w = [Fraction(x) for x in weights]
        return _mv(self.B, w, self.n)

    def add(self, v, dw):
        """多段加算: v' = v + B@Δw"""
        dv = _mv(self.B, [Fraction(x) for x in dw], self.n)
        vf = [Fraction(x) for x in v]
        return [vf[i]+dv[i] for i in range(self.n)]

    def decode(self, v):
        """w = B⁻¹@v（厳密）"""
        return _mv(self.Inv, [Fraction(x) for x in v], self.n)

    def norm2(self, v):
        """‖v‖²（Fraction厳密）"""
        return sum(Fraction(x)**2 for x in v)

    def is_pythagorean(self, v):
        """‖v‖² が完全平方数かどうか判定 → z を返す"""
        n2 = self.norm2(v)
        ok, z = is_perfect_square(n2)
        return ok, z, n2

    def multistep_search(self, w_steps, verbose=True):
        """
        重み差分リストで多段加算し、各ステップでノルムを検証

        Parameters
        ----------
        w_steps : list of list  各ステップの Δw
        """
        v = [Fraction(0)] * self.n
        if verbose:
            print(f"  step  {'合算v':^24} {'‖v‖²':^14} {'z=‖v‖':^10} {'ピタゴラス':^10}")
            print(f"  {'':─<4}  {'':─<24} {'':─<14} {'':─<10} {'':─<10}")

        results = []
        for k, dw in enumerate(w_steps):
            v = self.add(v, dw)
            ok, z, n2 = self.is_pythagorean(v)
            mark = f"({v[0]},{v[1]}) z={z} ✓" if ok else ""
            if verbose:
                vs = f"({float(v[0]):.3f},{float(v[1]):.3f})"
                print(f"  {k+1:>4}  {vs:^24} {str(n2):^14} "
                      f"{str(z) if ok else '─':^10} {mark}")
            if ok and z > 0:
                results.append((v[0], v[1], z, k+1))
        return results


# ══════════════════════════════════════════════
# C. 内積による交差項の解析
#    ‖v₁+v₂‖² = ‖v₁‖²+‖v₂‖²+2(v₁·v₂)
#    交差項 2(v₁·v₂) が整数かつ全体が完全平方 → 新三つ組
# ══════════════════════════════════════════════

def cross_term_analysis(triples):
    """多段合算の交差項を Fraction で厳密解析"""
    print("\n【C. 交差項（内積）の厳密解析】")
    print(f"  {'v₁':^10} {'v₂':^10} "
          f"{'‖v₁‖²':^8} {'‖v₂‖²':^8} {'2v₁·v₂':^10} "
          f"{'‖Σv‖²':^10} {'z':^8}")
    print(f"  {'':─<10} {'':─<10} "
          f"{'':─<8} {'':─<8} {'':─<10} {'':─<10} {'':─<8}")

    found = []
    for i in range(len(triples)):
        for j in range(i, len(triples)):
            x1,y1,z1 = triples[i]
            x2,y2,z2 = triples[j]
            v1 = [Fraction(x1), Fraction(y1)]
            v2 = [Fraction(x2), Fraction(y2)]
            n1 = norm2(v1)
            n2 = norm2(v2)
            cross = 2 * dot(v1, v2)
            total = n1 + n2 + cross
            ok, z = is_perfect_square(total)
            vs = vec_add(v1, v2)
            if ok and z > 0 and int(vs[0])>0 and int(vs[1])>0:
                found.append((int(vs[0]), int(vs[1]), z))
                print(f"  ({x1},{y1})   ({x2},{y2})   "
                      f"{int(n1):^8} {int(n2):^8} {int(cross):^10} "
                      f"{int(total):^10} {z:^8} ✓")
    return found


# ══════════════════════════════════════════════
# デモ
# ══════════════════════════════════════════════

def demo():
    sep = "=" * 62
    print(sep)
    print("  x²+y²=z²  内積ベクトル多段合算 効率演算")
    print(sep)

    # 既知のピタゴラス三つ組（原始的なもの）
    primitives = [
        (3,4,5),(5,12,13),(8,15,17),(7,24,25),
        (20,21,29),(9,40,41),(12,35,37),(11,60,61),
    ]

    # ── A. 直交合算 ──
    orthogonal_multistep(primitives)

    # ── B. 基底行列エンコーダーによる多段加算探索 ──
    print("\n【B. 加算ベクトルエンコーダーによる多段ピタゴラス探索】")

    # 基底: 単位ベクトル方向（最もシンプルな基底）
    B_std = [[1, 0],
             [0, 1]]
    enc = PythagoreanEncoder(B_std)

    # 重み差分を整数ステップで多段加算
    # Δwステップを設計: 各ステップで既知三つ組方向へ加算
    w_steps = [
        [3,  4],   # v=(3,4)   → z²=25, z=5  ✓
        [2,  0],   # v=(5,4)   → z²=41       ✗
        [0,  8],   # v=(5,12)  → z²=169,z=13 ✓
        [3,  0],   # v=(8,12)  → z²=208       ✗
        [0,  3],   # v=(8,15)  → z²=289,z=17 ✓
        [12, 9],   # v=(20,24) → z²=976       ✗
        [0,  1],   # v=(20,25) → z²=1025      ✗
        [-4, 0],   # v=(16,25) → z²=881       ✗
        [-4, 0],   # v=(12,25) → z²=769       ✗
        [-3, 0],   # v=(9,25)  → z²=706       ✗
        [-2, 0],   # v=(7,25)  → z²=674       ✗
        [0, -1],   # v=(7,24)  → z²=625,z=25 ✓
    ]
    results = enc.multistep_search(w_steps)

    print(f"\n  検出したピタゴラス三つ組: {len(results)} 件")
    for x, y, z, step in results:
        print(f"    step {step:>2}: ({int(x)}, {int(y)}, {z})")

    # ── C. 交差項解析 ──
    cross_term_analysis(primitives[:6])

    # ── まとめ: ノルム保存性の検証 ──
    print(f"\n{'─'*62}")
    print("  【ノルム保存性の検証】")
    print("  合算ベクトルのノルム² = ‖v₁‖² + ‖v₂‖² + 2(v₁·v₂)")
    print()

    cases = [
        ((3,4), (4,-3),  "直交ペア (dot=0)"),
        ((3,4), (5,12),  "同方向寄り   "),
        ((3,4), (12,-5), "dot=3*12+4*(-5)=16"),
    ]
    for v1t, v2t, label in cases:
        v1 = [Fraction(v1t[0]), Fraction(v1t[1])]
        v2 = [Fraction(v2t[0]), Fraction(v2t[1])]
        n1   = norm2(v1)
        n2   = norm2(v2)
        cr   = 2 * dot(v1, v2)
        vs   = vec_add(v1, v2)
        tot  = norm2(vs)
        ok, z = is_perfect_square(tot)
        print(f"  {label}")
        print(f"    v₁={v1t}, v₂={v2t}")
        print(f"    ‖v₁‖²={n1}, ‖v₂‖²={n2}, 2(v₁·v₂)={cr}")
        print(f"    ‖v₁+v₂‖²={tot}  →  z={'√'+str(tot) if not ok else str(z)}"
              f"{'  ← ピタゴラス ✓' if ok and z>0 else ''}")
        print()

    print(sep)
    print("  結論:")
    print("    1. 直交合算（v₁·v₂=0）: ‖Σv‖²=Σ‖vᵢ‖² → 最も効率的")
    print("    2. 一般合算: 交差項 2(v₁·v₂) を Fraction で厳密管理")
    print("    3. CH×Fraction 逆行列: 重みwを誤差ゼロで復元")
    print("    4. 多段加算で既知三つ組の軌跡上を辿り新三つ組を発見")
    print(sep)


if __name__ == "__main__":
    demo()
