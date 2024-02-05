const D: f64 = 1.702;

fn exp_pow(x: f64, diff: f64, disc: f64) -> f64 {
    -D * disc * (x - diff)
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_diff(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn irt_3pl(x: f64, diff: f64, disc: f64, lambda: f64) -> f64 {
    lambda + (1.0 - lambda) * sigmoid(-exp_pow(x, diff, disc))
}

pub fn irt_3pl_diff(x: f64, diff: f64, disc: f64, lambda: f64) -> f64 {
    D * disc
        * (1.0 - lambda)
        * exp_pow(x, diff, disc).exp()
        * sigmoid(-exp_pow(x, diff, disc)).powi(2)
}

pub fn irt_3pl_mle(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    // println!("irt 3pl :{} ,pf:{pf}", irt_3pl(x, diff, disc, lambda));
    irt_3pl(x, diff, disc, lambda).powf(pf) * (1.0 - irt_3pl(x, diff, disc, lambda)).powf(1.0 - pf)
        + 1e-10
}

pub fn irt_3pl_mle_diff(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    D * disc
        * (1.0 - lambda)
        * exp_pow(x, diff, disc).exp()
        * sigmoid(-exp_pow(x, diff, disc)).powi(2)
        * irt_3pl_mle(x, diff, disc, lambda, pf)
        * (pf / irt_3pl(x, diff, disc, lambda)
            - (1.0 - pf) / (1.0 - irt_3pl(x, diff, disc, lambda)))
}

pub fn irt_3pl_mle_ln(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    // println!("irt 3pl mle {}", irt_3pl_mle(x, diff, disc, lambda, pf));
    irt_3pl_mle(x, diff, disc, lambda, pf).ln()
}

pub fn irt_3pl_mle_ln_diff(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    irt_3pl_mle_diff(x, diff, disc, lambda, pf) * irt_3pl_mle(x, diff, disc, lambda, pf).powi(-1)
}
