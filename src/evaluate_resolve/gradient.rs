use super::EvaluateProblem;
use argmin::core::Gradient;
impl Gradient for EvaluateProblem {
    type Param = f64;

    type Gradient = f64;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        Ok(-self
            .records
            .iter()
            .map(|ob| Self::ln_probability_derivative(*ob, *param))
            .sum::<f64>())
    }
}
