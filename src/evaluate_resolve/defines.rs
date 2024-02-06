const D: f64 = 1.702;

fn exp_pow(x: f64, diff: f64, disc: f64) -> f64 {
    D * disc * (x - diff)
}

fn exp(x: f64) -> f64 {
    x.exp()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn irt_3pl(x: f64, diff: f64, disc: f64, lambda: f64) -> f64 {
    lambda + (1.0 - lambda) * sigmoid(exp_pow(x, diff, disc))
}

pub fn irt_3pl_diff(x: f64, diff: f64, disc: f64, lambda: f64) -> f64 {
    D * disc
        * (1.0 - lambda)
        * (-exp_pow(x, diff, disc)).exp()
        * sigmoid(exp_pow(x, diff, disc)).powi(2)
}

pub fn irt_3pl_mle(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    irt_3pl(x, diff, disc, lambda).powf(pf) * (1.0 - irt_3pl(x, diff, disc, lambda)).powf(1.0 - pf)
        + 1e-10
}

pub fn irt_3pl_mle_diff(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    let v1 = D
        * disc
        * (-lambda + pf + pf * exp(-D * diff * disc) * exp(D * disc * x)
            - exp(-D * diff * disc) * exp(D * disc * x))
        * exp(-D * diff * disc)
        * exp(D * disc * x)
        / ((1.0 + exp(-D * diff * disc) * exp(D * disc * x))
            * (lambda + exp(-D * diff * disc) * exp(D * disc * x)));

    assert!(
        !v1.is_nan() || v1.is_finite(),
        "v={v1}, x={x}, diff={diff}, disc={disc}, lambda={lambda},p={pf}, irt_3pl= {},irt_3pl_mle={}, irt_3pl_diff={}",
        irt_3pl(x, diff, disc, lambda),irt_3pl_mle(x, diff, disc, lambda, pf),irt_3pl_diff(x, diff, disc, lambda)
    );
    v1
}

pub fn irt_3pl_mle_ln(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    let irt_3pl_v = irt_3pl_mle(x, diff, disc, lambda, pf);
    debug_assert!(
        irt_3pl_v > 0.0,
        "irt 3pl <= 0 {irt_3pl_v} , x={x}, diff={diff}, disc={disc}, lambda={lambda},p={pf}"
    );
    irt_3pl_mle(x, diff, disc, lambda, pf).ln()
}

pub fn irt_3pl_mle_ln_diff(x: f64, diff: f64, disc: f64, lambda: f64, pf: f64) -> f64 {
    irt_3pl_mle_diff(x, diff, disc, lambda, pf) * irt_3pl_mle(x, diff, disc, lambda, pf).powi(-1)
}
