use anyhow::Error;
use argmin::core::{IterState, KV, Problem, Solver};

pub struct ExpectationMaximizationResolver<ML, EL> {
    m_step: ML,
    e_step: EL,
}

impl<ML, EL, O, P, G, J, H, F> Solver<O, IterState<P, G, J, H, F>> for ExpectationMaximizationResolver<ML, EL>
    where ML: {
    const NAME: &'static str = "Expectation Maximization";

    fn next_iter(&mut self, problem: &mut Problem<O>, state: IterState<P, G, J, H, F>) -> Result<(IterState<P, G, J, H, F>, Option<KV>), Error> {
        todo!()
    }
}