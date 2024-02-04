use argmin::core::CostFunction;

use super::EvaluateProblem;

impl CostFunction for EvaluateProblem {
    type Param = f64;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let cost = -self
            .records
            .iter()
            .map(|ob| Self::ln_probability(*ob, *param))
            .sum::<f64>();
        println!("Cost: {cost} param: {param}");
        Ok(cost)
    }
}
