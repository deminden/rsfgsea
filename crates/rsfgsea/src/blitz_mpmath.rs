use anyhow::{Result, bail};
use astro_float::{BigFloat, Consts, Radix, RoundingMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TailBranch {
    Positive,
    Negative,
}

#[derive(Debug, Clone)]
pub(crate) struct GammaCdf {
    pub cdf: f64,
    pub survival: f64,
    lower: BigFloat,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct TailProbability {
    pub gamma_prob: f64,
    pub survival_prob: f64,
    pub prob_two_tailed: f64,
    pub p_value: f64,
}

struct Hp {
    precision: usize,
    exact_precision: bool,
    cc: Consts,
}

impl Hp {
    fn new(dps: usize) -> Result<Self> {
        let working_dps = dps
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("blitz mpmath fallback dps is too large"))?;
        Self::with_dps(working_dps, false)
    }

    fn final_context(dps: usize) -> Result<Self> {
        Self::with_dps(dps, true)
    }

    fn with_dps(dps: usize, exact_precision: bool) -> Result<Self> {
        Ok(Self {
            precision: dps_to_prec(dps),
            exact_precision,
            cc: Consts::new()?,
        })
    }

    fn zero(&self) -> BigFloat {
        BigFloat::new(self.precision)
    }

    fn one(&self) -> BigFloat {
        BigFloat::from_u32(1, self.precision)
    }

    fn half(&self) -> BigFloat {
        BigFloat::from_f64(0.5, self.precision)
    }

    fn bf_f64(&self, value: f64) -> BigFloat {
        BigFloat::from_f64(value, self.precision)
    }

    fn bf_u64(&self, value: u64) -> BigFloat {
        BigFloat::from_u64(value, self.precision)
    }

    fn add(&self, lhs: &BigFloat, rhs: &BigFloat) -> Result<BigFloat> {
        self.checked(lhs.add(rhs, self.precision, RoundingMode::ToEven))
    }

    fn sub(&self, lhs: &BigFloat, rhs: &BigFloat) -> Result<BigFloat> {
        self.checked(lhs.sub(rhs, self.precision, RoundingMode::ToEven))
    }

    fn mul(&self, lhs: &BigFloat, rhs: &BigFloat) -> Result<BigFloat> {
        self.checked(lhs.mul(rhs, self.precision, RoundingMode::ToEven))
    }

    fn div(&self, lhs: &BigFloat, rhs: &BigFloat) -> Result<BigFloat> {
        self.checked(lhs.div(rhs, self.precision, RoundingMode::ToEven))
    }

    fn neg(&self, value: &BigFloat) -> BigFloat {
        -value
    }

    fn abs(&self, value: &BigFloat) -> Result<BigFloat> {
        self.checked(value.abs())
    }

    fn ln(&mut self, value: &BigFloat) -> Result<BigFloat> {
        let result = value.ln(self.precision, RoundingMode::ToEven, &mut self.cc);
        self.checked(result)
    }

    fn exp(&mut self, value: &BigFloat) -> Result<BigFloat> {
        let result = value.exp(self.precision, RoundingMode::ToEven, &mut self.cc);
        self.checked(result)
    }

    fn pi(&mut self) -> Result<BigFloat> {
        let value = self.cc.pi(self.precision, RoundingMode::ToEven);
        self.checked(value)
    }

    fn render_f64(&mut self, value: &BigFloat) -> Result<f64> {
        let rendered = value.format(Radix::Dec, RoundingMode::ToEven, &mut self.cc)?;
        Ok(rendered.parse::<f64>()?)
    }

    fn checked(&self, mut value: BigFloat) -> Result<BigFloat> {
        if value.is_nan() {
            bail!("blitz mpmath fallback produced NaN");
        }
        if self.exact_precision {
            // astro-float evaluates at a whole-word mantissa even when a
            // smaller precision is requested. mpmath rounds after every
            // final-context operation at the exact bit precision.
            value.set_precision(self.precision, RoundingMode::ToEven)?;
        }
        Ok(value)
    }

    fn round_to_context(&self, mut value: BigFloat) -> Result<BigFloat> {
        value.set_precision(self.precision, RoundingMode::ToEven)?;
        self.checked(value)
    }
}

pub(crate) fn gammacdf(x: f64, alpha: f64, beta: f64, dps: usize) -> Result<GammaCdf> {
    if !x.is_finite() || !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        bail!("blitz mpmath fallback requires finite positive gamma parameters");
    }
    if x < 0.0 {
        let hp = Hp::new(dps)?;
        let lower = hp.zero();
        return Ok(GammaCdf {
            cdf: 0.0,
            survival: 1.0,
            lower,
        });
    }

    let mut hp = Hp::new(dps)?;
    // blitzgsea promotes both operands before dividing inside its extradps
    // context. Dividing as f64 first can move extreme tails across a final
    // context rounding boundary.
    let z = hp.div(&hp.bf_f64(x), &hp.bf_f64(beta))?;
    if z.is_zero() {
        let lower = hp.zero();
        return Ok(GammaCdf {
            cdf: 0.0,
            survival: 1.0,
            lower,
        });
    }
    let a = hp.bf_f64(alpha);
    let one = hp.one();
    let switch = hp.add(&a, &one)?;
    let use_lower_series = z < switch;
    let (lower, upper) = if use_lower_series {
        let lower = lower_gamma_series_regularized(&mut hp, &a, &z, alpha)?;
        let upper = hp.sub(&one, &lower)?;
        (lower, upper)
    } else {
        let upper = upper_gamma_cf_regularized(&mut hp, &a, &z, alpha)?;
        let lower = hp.sub(&one, &upper)?;
        (lower, upper)
    };
    // The extradps result returned by mpmath retains exactly the 2*dps
    // context. Keep guard-word arithmetic inside the gamma solver, then trim
    // its two public results once before final-context tail arithmetic.
    let lower = hp.round_to_context(lower)?;
    let upper = hp.round_to_context(upper)?;
    let cdf = hp.render_f64(&lower)?;
    let survival = hp.render_f64(&upper)?;
    Ok(GammaCdf {
        cdf,
        survival,
        lower,
    })
}

pub(crate) fn tail_probability(
    branch: TailBranch,
    x: f64,
    alpha: f64,
    beta: f64,
    pos_ratio: f64,
    dps: usize,
) -> Result<TailProbability> {
    let gamma = gammacdf(x, alpha, beta, dps)?;
    let mut py_hp = Hp::final_context(dps)?;
    let ratio = py_hp.bf_f64(pos_ratio);
    let one = py_hp.one();
    let half = py_hp.half();

    let combined = match branch {
        TailBranch::Positive => {
            let weighted = py_hp.mul(&gamma.lower, &ratio)?;
            // Preserve Python's left-associated `lower * ratio + 1 - ratio`.
            py_hp.sub(&py_hp.add(&weighted, &one)?, &ratio)?
        }
        TailBranch::Negative => {
            let weighted = py_hp.mul(&gamma.lower, &ratio)?;
            py_hp.add(&py_hp.sub(&gamma.lower, &weighted)?, &ratio)?
        }
    };
    let combined = if combined < one {
        combined
    } else {
        one.clone()
    };
    let raw_prob_two = py_hp.sub(&one, &combined)?;
    let mut prob_two = if raw_prob_two < half {
        raw_prob_two
    } else {
        half.clone()
    };
    if branch == TailBranch::Negative && prob_two == half {
        prob_two = py_hp.sub(&prob_two, &gamma.lower)?;
    }
    let two = py_hp.bf_u64(2);
    let mut p_value = py_hp.mul(&two, &prob_two)?;
    if p_value > one {
        p_value = one.clone();
    }
    if p_value.is_zero() {
        let survival_prob = py_hp.sub(&one, &gamma.lower)?;
        return Ok(TailProbability {
            gamma_prob: py_hp.render_f64(&gamma.lower)?,
            survival_prob: py_hp.render_f64(&survival_prob)?,
            prob_two_tailed: 0.0,
            p_value: 0.0,
        });
    }
    Ok(TailProbability {
        gamma_prob: gamma.cdf,
        survival_prob: gamma.survival,
        prob_two_tailed: py_hp.render_f64(&prob_two)?,
        p_value: py_hp.render_f64(&p_value)?,
    })
}

fn lower_gamma_series_regularized(
    hp: &mut Hp,
    a: &BigFloat,
    z: &BigFloat,
    alpha_f64: f64,
) -> Result<BigFloat> {
    let one = hp.one();
    let mut ap = a.clone();
    let mut term = hp.div(&one, a)?;
    let mut sum = term.clone();
    let eps = hp.bf_f64(2f64.powi(-((hp.precision.saturating_sub(24)).min(900) as i32)));
    for _ in 0..20_000 {
        ap = hp.add(&ap, &one)?;
        term = hp.div(&hp.mul(&term, z)?, &ap)?;
        sum = hp.add(&sum, &term)?;
        let threshold = hp.mul(&hp.abs(&sum)?, &eps)?;
        if hp.abs(&term)? < threshold {
            let prefactor = gamma_prefactor(hp, a, z, alpha_f64)?;
            return hp.mul(&prefactor, &sum);
        }
    }
    bail!("blitz mpmath fallback lower gamma series did not converge")
}

fn upper_gamma_cf_regularized(
    hp: &mut Hp,
    a: &BigFloat,
    z: &BigFloat,
    alpha_f64: f64,
) -> Result<BigFloat> {
    let one = hp.one();
    let two = hp.bf_u64(2);
    let tiny = hp.bf_f64(1.0e-300);
    let eps = hp.bf_f64(2f64.powi(-((hp.precision.saturating_sub(24)).min(900) as i32)));

    let mut b = hp.add(&hp.sub(z, a)?, &one)?;
    let mut c = hp.div(&one, &tiny)?;
    let mut d = reciprocal_nonzero(hp, &b, &tiny)?;
    let mut h = d.clone();
    for i in 1..20_000u64 {
        let i_bf = hp.bf_u64(i);
        let a_minus_i = hp.sub(a, &i_bf)?;
        let an = hp.mul(&i_bf, &a_minus_i)?;
        b = hp.add(&b, &two)?;

        d = hp.add(&hp.mul(&an, &d)?, &b)?;
        if hp.abs(&d)? < tiny {
            d = tiny.clone();
        }
        c = hp.add(&b, &hp.div(&an, &c)?)?;
        if hp.abs(&c)? < tiny {
            c = tiny.clone();
        }
        d = hp.div(&one, &d)?;
        let delta = hp.mul(&d, &c)?;
        h = hp.mul(&h, &delta)?;
        let delta_minus_one = hp.abs(&hp.sub(&delta, &one)?)?;
        if delta_minus_one < eps {
            let prefactor = gamma_prefactor(hp, a, z, alpha_f64)?;
            return hp.mul(&prefactor, &h);
        }
    }
    bail!("blitz mpmath fallback upper gamma continued fraction did not converge")
}

fn reciprocal_nonzero(hp: &Hp, value: &BigFloat, tiny: &BigFloat) -> Result<BigFloat> {
    let one = hp.one();
    if hp.abs(value)? < *tiny {
        hp.div(&one, tiny)
    } else {
        hp.div(&one, value)
    }
}

fn gamma_prefactor(hp: &mut Hp, a: &BigFloat, z: &BigFloat, alpha_f64: f64) -> Result<BigFloat> {
    let ln_z = hp.ln(z)?;
    let a_ln_z = hp.mul(a, &ln_z)?;
    let neg_z = hp.neg(z);
    let ln_gamma = ln_gamma_positive(hp, alpha_f64)?;
    let exponent = hp.sub(&hp.add(&neg_z, &a_ln_z)?, &ln_gamma)?;
    hp.exp(&exponent)
}

fn ln_gamma_positive(hp: &mut Hp, alpha: f64) -> Result<BigFloat> {
    if let Some(n) = positive_integer(alpha) {
        return ln_factorial(hp, n - 1);
    }
    if let Some(m) = positive_half_integer_offset(alpha) {
        let ln_two_m_fact = ln_factorial(hp, 2 * m)?;
        let ln_m_fact = ln_factorial(hp, m)?;
        let ln_four = hp.ln(&hp.bf_u64(4))?;
        let m_ln_four = hp.mul(&hp.bf_u64(m), &ln_four)?;
        let half = hp.half();
        let pi = hp.pi()?;
        let ln_pi = hp.ln(&pi)?;
        let half_ln_pi = hp.mul(&half, &ln_pi)?;
        return hp.add(
            &hp.sub(&hp.sub(&ln_two_m_fact, &m_ln_four)?, &ln_m_fact)?,
            &half_ln_pi,
        );
    }
    spouge_ln_gamma(hp, alpha, 80)
}

fn spouge_ln_gamma(hp: &mut Hp, alpha: f64, a: u64) -> Result<BigFloat> {
    let z_minus_one = hp.sub(&hp.bf_f64(alpha), &hp.one())?;
    let pi = hp.pi()?;
    let two_pi = hp.mul(&hp.bf_u64(2), &pi)?;
    let c0 = two_pi.sqrt(hp.precision, RoundingMode::ToEven);
    let mut sum = hp.checked(c0)?;
    for k in 1..a {
        let a_minus_k = hp.bf_u64(a - k);
        let exponent = hp.sub(&hp.bf_u64(k), &hp.half())?;
        let ln_a_minus_k = hp.ln(&a_minus_k)?;
        let power_log = hp.mul(&exponent, &ln_a_minus_k)?;
        let coeff_numerator_log = hp.add(&power_log, &a_minus_k)?;
        let ln_k_minus_one_fact = ln_factorial(hp, k - 1)?;
        let coeff_log = hp.sub(&coeff_numerator_log, &ln_k_minus_one_fact)?;
        let mut coeff = hp.exp(&coeff_log)?;
        if k % 2 == 0 {
            coeff = hp.neg(&coeff);
        }
        let denominator = hp.add(&z_minus_one, &hp.bf_u64(k))?;
        let term = hp.div(&coeff, &denominator)?;
        sum = hp.add(&sum, &term)?;
    }

    let base = hp.add(&z_minus_one, &hp.bf_u64(a))?;
    let exponent = hp.add(&z_minus_one, &hp.half())?;
    let ln_base = hp.ln(&base)?;
    let leading = hp.mul(&exponent, &ln_base)?;
    let log_sum = hp.ln(&sum)?;
    hp.add(&hp.sub(&leading, &base)?, &log_sum)
}

fn ln_factorial(hp: &mut Hp, n: u64) -> Result<BigFloat> {
    let mut total = hp.zero();
    for value in 2..=n {
        let ln_value = hp.ln(&hp.bf_u64(value))?;
        total = hp.add(&total, &ln_value)?;
    }
    Ok(total)
}

fn positive_integer(value: f64) -> Option<u64> {
    if value >= 1.0 && value <= u64::MAX as f64 && value.fract() == 0.0 {
        Some(value as u64)
    } else {
        None
    }
}

fn positive_half_integer_offset(value: f64) -> Option<u64> {
    let offset = value - 0.5;
    if offset >= 0.0 && offset <= u64::MAX as f64 && offset.fract() == 0.0 {
        Some(offset as u64)
    } else {
        None
    }
}

fn dps_to_prec(dps: usize) -> usize {
    const BLOG2_10: f64 = std::f64::consts::LOG2_10;
    (dps.saturating_add(1) as f64 * BLOG2_10).round().max(1.0) as usize
}
