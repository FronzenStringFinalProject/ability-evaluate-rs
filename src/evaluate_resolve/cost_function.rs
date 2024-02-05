use std::time::Duration;

use argmin::core::CostFunction;

use crate::evaluate_resolve::{defines::irt_3pl_mle_ln, AnsweredQuiz};

use super::EvaluateProblem;

impl CostFunction for EvaluateProblem {
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
        println!("Cost: {cost} param: {param}");
        // std::thread::sleep(Duration::from_secs(4));
        Ok(cost)
    }
}
