use argmin::core::CostFunction;

use crate::evaluate_resolve::{defines::irt_3pl_mle_ln, AnsweredQuiz};

use super::EvaluateProblem;

impl CostFunction for EvaluateProblem<'_> {
    type Param = f64;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let cost = -self
            .records
            .iter()
            .map(
                |ob @ &AnsweredQuiz {
                     diff,
                     disc,
                     lambdas,
                     ..
                 }| irt_3pl_mle_ln(*param, diff, disc, lambdas, ob.pf()),
            )
            .sum::<f64>();
        Ok(cost)
    }
}
