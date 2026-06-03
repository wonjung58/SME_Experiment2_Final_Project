import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

# ── 모듈 전역: 격자 및 탐색 범위 (main에서 1회 초기화) ──────
_grid = None
_bounds = None


def _setup_grid(p_bs):
    """앵커 좌표로부터 탐색 영역과 0.5 m 격자를 생성한다."""
    global _grid, _bounds
    beta = 0.30
    wx = p_bs[0].max() - p_bs[0].min()
    wy = p_bs[1].max() - p_bs[1].min()
    _bounds = [
        (p_bs[0].min() - beta * wx, p_bs[0].max() + beta * wx),
        (p_bs[1].min() - beta * wy, p_bs[1].max() + beta * wy),
    ]
    xs = np.arange(_bounds[0][0], _bounds[0][1] + 0.5, 0.5)
    ys = np.arange(_bounds[1][0], _bounds[1][1] + 0.5, 0.5)
    gx, gy = np.meshgrid(xs, ys)
    _grid = np.stack([gx.ravel(), gy.ravel()], axis=1)


def _minimax(d_vec, p_bs):
    """
    Step 1 — Chebyshev Center (Minimax)
    NLOS 양의 편향 성질(d_hat >= d_true)을 이용하여
    18개 측정 원의 가장 깊은 내부점을 탐색한다.
    목적함수: min_q max_i (||q - b_i|| - d_i)
    log-sum-exp(alpha=10)로 smooth 근사하여 L-BFGS-B로 최적화.
    """
    diff_x = _grid[:, 0:1] - p_bs[0:1, :]
    diff_y = _grid[:, 1:2] - p_bs[1:2, :]
    pred = np.sqrt(diff_x ** 2 + diff_y ** 2)
    slack = pred - d_vec[None, :]
    q_init = _grid[np.argmin(slack.max(axis=1))]

    alpha = 10.0

    def obj(q):
        dist = np.sqrt((q[0] - p_bs[0]) ** 2 + (q[1] - p_bs[1]) ** 2)
        s = dist - d_vec
        return (1.0 / alpha) * np.log(np.exp(alpha * s).sum())

    res = minimize(obj, q_init, method="L-BFGS-B", bounds=_bounds,
                   options={"ftol": 1e-12})
    return res.x


def _lts_refine(p_init, d_vec, p_bs, k=7):
    """
    Step 2 — Least Trimmed Squares (LTS)
    18개 잔차 중 제곱값이 가장 작은 k개만의 합을 최소화하여
    NLOS 심한 앵커를 자동으로 무시하고 위치를 정밀화한다.
    """
    def obj(q):
        r = np.sqrt((q[0] - p_bs[0]) ** 2 + (q[1] - p_bs[1]) ** 2) - d_vec
        return np.sort(r ** 2)[:k].sum()

    res = minimize(obj, p_init, method="L-BFGS-B", bounds=_bounds,
                   options={"ftol": 1e-12})
    return res.x


def your_algorithm(d_u, p_bs):
    """
    Minimax–LTS Localization
    -------------------------------------------------
    Step 1: Chebyshev Center (Minimax)
            모든 측정 원 내부에서 가장 깊은 점을 찾는다.
    Step 2: LTS 정밀화
            잔차 하위 7개 앵커만 fit하여 위치를 정밀화한다.
    Step 3: Slack 기반 Fallback
            LTS 결과의 max slack > 5 m이면
            Minimax 결과로 되돌린다.
    """
    d_vec = np.asarray(d_u, dtype=float)

    # Step 1
    p_mm = _minimax(d_vec, p_bs)

    # Step 2
    p_lts = _lts_refine(p_mm, d_vec, p_bs, k=7)

    # Step 3
    dist_lts = np.sqrt((p_lts[0] - p_bs[0]) ** 2 + (p_lts[1] - p_bs[1]) ** 2)
    max_slack = (dist_lts - d_vec).max()

    if max_slack > 5.0:
        return p_mm
    return p_lts


def main():
    # 1) 입력 데이터 로드
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'], dtype=float)   # (2, 18)
    d_hat = np.asarray(data['d_hat'], dtype=float)                 # (18, num_user)
    p = np.asarray(data['p'], dtype=float)                         # (2, num_user)

    # 격자 초기화 (1회)
    _setup_grid(BS_positions)

    # 2) 본인 알고리즘
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 3) 결과 반환
    return p_hat


if __name__ == "__main__":
    main()