"""本测试模块用于验证约束相关辅助函数的逻辑正确性，确保可行性掩码、约束筛选和概率处理结果符合预期。"""

def test_acquisition_returns_legal_candidates():
    import torch

    from sws_bo.acquisition.constrained_qnehvi import recommend_candidates
    from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem
    from sws_bo.surrogate.train import train_independent_gp
    from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator

    simulator = MockDSGCSTSimulator(seed=4)
    xs = []
    ys = []
    for shift in [0.0, 0.01, -0.01, 0.015, -0.015]:
        x = DSGSWSProblem.reference_design.copy()
        x[0] += shift
        x[3] += 0.2 * shift
        xs.append(x)
        result = simulator.run(x)
        ys.append([result.Kc_mean, result.vp_std, result.ohmic_loss_mean, result.S11_max])
    train_X = torch.tensor(DSGSWSProblem.normalize(xs), dtype=torch.double)
    train_Y = torch.tensor(ys, dtype=torch.double)
    model = train_independent_gp(train_X, train_Y, ard=True)
    rec = recommend_candidates(model, train_X, train_Y, q=2, pf_min=0.1)
    assert rec.normalized.shape == (2, 5)
    assert ((rec.normalized >= 0.0) & (rec.normalized <= 1.0)).all()
