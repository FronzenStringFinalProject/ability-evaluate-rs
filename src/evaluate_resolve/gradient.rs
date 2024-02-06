use super::{defines::irt_3pl_mle_ln_diff, AnsweredQuiz, EvaluateProblem};
use argmin::core::Gradient;
impl Gradient for EvaluateProblem<'_> {
    type Param = f64;

    type Gradient = f64;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let v = -self
            .records
            .iter()
            .map(
                |ob @ &AnsweredQuiz {
                     diff,
                     disc,
                     lambdas,
                     ..
                 }| irt_3pl_mle_ln_diff(*param, diff, disc, lambdas, ob.pf()),
            )
            .sum::<f64>();
        Ok(v)
    }
}
