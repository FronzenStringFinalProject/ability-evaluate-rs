mod evaluate_resolve;

use anyhow::{anyhow, Ok};
use argmin::{
    core::Executor,
    solver::{gradientdescent::SteepestDescent, linesearch::MoreThuenteLineSearch},
};
use evaluate_resolve::TARGET_COST;
pub use evaluate_resolve::{AnsweredQuiz, EvaluateProblem, LEST_QUIZ_NUMBER};

/// evaluate 通过一定的做题数量进行学生水平评估
///
/// 使用最大似然估计法进行水平评估
///     - 使用 More Thuente Line Search 进行步长迭代
///     - 使用 Steepest Descent 进行梯度下降
///
/// - ans_record: 学生的做题记录，推荐记录数量大于25。过少记录准确度将偏低
pub fn evaluate(ans_record: &[AnsweredQuiz]) -> anyhow::Result<f64> {
    let problem = EvaluateProblem::new(ans_record);

    let line_search = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(line_search);

    let executor = Executor::new(problem, solver)
        .configure(|state| state.param(0.0).max_iters(5).target_cost(TARGET_COST));

    let result = executor.run()?;
    if let Some(param) = result.state().best_param {
        Ok(param)
    } else {
        Err(anyhow!("未找到最佳的水平值"))
    }
}
