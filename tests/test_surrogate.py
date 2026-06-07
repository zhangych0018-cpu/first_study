"""本测试模块用于验证代理模型在小样本数据上能够完成训练和预测，并返回形状正确的结果。"""

def test_surrogate_fit_and_predict():
    import torch

    from sws_bo.problems.dsg_bwo_problem import DSGSWSProblem
    from sws_bo.surrogate.independent_gp import build_independent_model, fit_gp_model, predict
    from sws_bo.utils.mock_dsg_cst import MockDSGCSTSimulator

    simulator = MockDSGCSTSimulator(seed=2)
    xs = [DSGSWSProblem.reference_design + 0.01 * i for i in range(6)]
    ys = []
    for x in xs:
        result = simulator.run(x)
        ys.append([result.Kc_mean, result.vp_std, result.ohmic_loss_mean, result.S11_max])
    train_X = torch.tensor(DSGSWSProblem.normalize(xs), dtype=torch.double)
    train_Y = torch.tensor(ys, dtype=torch.double)
    model = fit_gp_model(build_independent_model(train_X, train_Y))
    mean, std = predict(model, train_X[:2])
    assert mean.shape == (2, 4)
    assert std.shape == (2, 4)
