mod evaluate_resolve;

use anyhow::{anyhow, bail, Ok};
use argmin::{
    core::Executor,
    solver::{gradientdescent::SteepestDescent, linesearch::MoreThuenteLineSearch},
};
pub use evaluate_resolve::{AnsweredQuiz, EvaluateProblem, LEST_QUIZ_NUMBER};
use evaluate_resolve::{MIN_QUIZ_NUMBER, TARGET_COST};

/// evaluate
pub fn evaluate(ans_record: &[AnsweredQuiz]) -> anyhow::Result<f64> {
    if ans_record.len() < MIN_QUIZ_NUMBER {
        bail!(
            "做题记录大小[{}]小于最低记录[5],推荐记录>20",
            ans_record.len()
        );
    }
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
