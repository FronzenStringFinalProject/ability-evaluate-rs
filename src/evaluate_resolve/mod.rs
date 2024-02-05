mod defines;
use serde::Deserialize;
use serde::Serialize;
mod cost_function;
mod gradient;
pub struct EvaluateProblem {
    records: Vec<AnsweredQuiz>,
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct AnsweredQuiz {
    diff: f64,
    disc: f64,
    lambdas: f64,
    correct: bool,
}

impl AnsweredQuiz {
    fn pf(&self) -> f64 {
        if self.correct {
            1.0
        } else {
            0.0
        }
    }
    fn pi(&self) -> i32 {
        if self.correct {
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod test {
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::{core::Executor, solver::gradientdescent::SteepestDescent};
    use presistence::{
        sea_orm::{ConnectOptions, Database},
        service::quiz_record::{ChildQuizAns, QuizRecord},
    };
    use std::f64::INFINITY;

    use super::{AnsweredQuiz, EvaluateProblem};
    #[tokio::test]
    async fn test() {
        let db = Database::connect(ConnectOptions::new(
            "postgres://JACKY:wyq020222@localhost/mydb",
        ))
        .await
        .expect("cannot connect  to db");

        let set = QuizRecord::get_ans_quiz_by_child_id(&db, 2)
            .await
            .expect("cannot get child ans")
            .into_iter()
            .map(
                |ChildQuizAns {
                     diff,
                     disc,
                     lambdas,
                     correct,
                     ..
                 }| {
                    AnsweredQuiz {
                        diff,
                        disc,
                        lambdas,
                        correct,
                    }
                },
            )
            .collect::<Vec<_>>();
        let len = set.len() as f64;
        let tc = 2.0f64.ln() * len;
        println!("{tc}");
        let problem = EvaluateProblem { records: set };
        let linesearch = MoreThuenteLineSearch::new();
        let solver = SteepestDescent::new(linesearch);

        let exec = Executor::new(problem, solver);
        let res = exec
            .configure(|state| state.param(1.0).max_iters(20).target_cost(0.0))
            .run()
            .expect("Err");

        println!("{}", res);

        // Extract results from state

        // // Best parameter vector

        // // Cost function value associated with best parameter vector
        // let best_cost = res.state().get_best_cost();

        // // Check the execution status
        // let termination_status = res.state().get_termination_status();

        // // Optionally, check why the optimizer terminated (if status is terminated)
        // let termination_reason = res.state().get_termination_reason();

        // // Time needed for optimization
        // let time_needed = res.state().get_time().unwrap();

        // // Total number of iterations needed
        // let num_iterations = res.state().get_iter();

        // // Iteration number where the last best parameter vector was found
        // let num_iterations_best = res.state().get_last_best_iter();

        // // Number of evaluation counts per method (Cost, Gradient)
        // let function_evaluation_counts = res.state().get_func_counts();
    }
}
