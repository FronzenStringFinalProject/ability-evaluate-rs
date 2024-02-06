mod defines;
use presistence::service::quiz_record::ChildQuizAns;
use serde::Deserialize;
use serde::Serialize;
mod cost_function;
mod gradient;
pub struct EvaluateProblem<'r> {
    records: &'r [AnsweredQuiz],
}

impl<'r> EvaluateProblem<'r> {
    pub fn new(records: &'r [AnsweredQuiz]) -> Self {
        Self { records }
    }
}

pub const LEST_QUIZ_NUMBER: usize = 25;
pub const TARGET_COST: f64 = 500.0;
#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct AnsweredQuiz {
    diff: f64,
    disc: f64,
    lambdas: f64,
    correct: bool,
}

impl From<ChildQuizAns> for AnsweredQuiz {
    fn from(
        ChildQuizAns {
            diff,
            disc,
            lambdas,
            correct,
            ..
        }: ChildQuizAns,
    ) -> Self {
        AnsweredQuiz {
            diff,
            disc,
            lambdas,
            correct,
        }
    }
}

impl AnsweredQuiz {
    fn pf(&self) -> f64 {
        if self.correct {
            1.0
        } else {
            0.0
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

    use crate::evaluate_resolve::defines::irt_3pl;
    use crate::evaluate_resolve::LEST_QUIZ_NUMBER;

    use super::{AnsweredQuiz, EvaluateProblem};
    #[tokio::test]
    async fn test() {
        let cid = 345;
        let db = Database::connect(ConnectOptions::new(
            "postgres://JACKY:wyq020222@localhost/mydb",
        ))
        .await
        .expect("cannot connect  to db");

        let set = QuizRecord::get_ans_quiz_by_child_id(&db, cid)
            .await
            .expect("cannot get child ans")
            .into_iter()
            .take(LEST_QUIZ_NUMBER)
            .map(AnsweredQuiz::from)
            .collect::<Vec<_>>();
        let problem = EvaluateProblem { records: &set };

        let linesearch = MoreThuenteLineSearch::new();
        let solver = SteepestDescent::new(linesearch);

        let exec = Executor::new(problem, solver);
        let res = exec
            .configure(|state| state.param(1.0).max_iters(20).target_cost(500.0))
            .run()
            .expect("Err");

        println!("{}", res);
        let ab = res.state().best_param.unwrap();
        println!("abi= {ab}");
        let set = QuizRecord::get_ans_quiz_by_child_id(&db, cid)
            .await
            .expect("cannot get child ans");
        for ChildQuizAns {
            diff,
            quiz,
            ans,
            disc,
            lambdas,
            correct,
            ability,
            pred,
        } in set
        {
            let irt = irt_3pl(ab, diff, disc, lambdas);
            println!(
                "quiz:{quiz}={ans}, pred={pred}, expect={}, diff={}, abb= {ability}, correct={correct}, exp_abb={ab}",
                irt,pred-irt
            )
        }
    }
}
