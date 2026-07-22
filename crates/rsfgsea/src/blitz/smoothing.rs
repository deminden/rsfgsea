use super::compat::numpy_pairwise_sum_f64;
use core::range::Range;

// LOWESS smooths noisy anchor fits into stable per-pathway-size model parameters.
const LOWESS_ITERS: usize = 3;

#[derive(Clone)]
pub(super) struct LinearInterp {
    pub(super) x: Vec<f64>,
    pub(super) y: Vec<f64>,
}

impl LinearInterp {
    pub(super) fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self { x, y }
    }

    pub(super) fn at(&self, xq: f64) -> f64 {
        let n = self.x.len();
        if n == 0 {
            return f64::NAN;
        }
        if n == 1 {
            return self.y[0];
        }

        // Match scipy.interpolate.interp1d's linear call path: search for the
        // upper neighbor, clip it to [1, n - 1], compute the slope first, and
        // then multiply by the query offset. The operation order matters by a
        // few ulps and can be amplified substantially by the inverse-normal
        // transform in extreme Blitz tails.
        let upper = match self
            .x
            .binary_search_by(|probe| probe.partial_cmp(&xq).unwrap())
        {
            Ok(i) | Err(i) => i.clamp(1, n - 1),
        };
        let lower = upper - 1;
        let x0 = self.x[lower];
        let x1 = self.x[upper];
        let y0 = self.y[lower];
        let y1 = self.y[upper];
        if x1 == x0 {
            y0
        } else {
            let slope = (y1 - y0) / (x1 - x0);
            slope * (xq - x0) + y0
        }
    }
}

pub(super) fn lowess_interpolation(x: &[f64], y: &[f64], frac: f64) -> LinearInterp {
    LinearInterp::new(x.to_vec(), lowess(y, x, frac))
}

pub(super) fn lowess(y: &[f64], x: &[f64], frac: f64) -> Vec<f64> {
    let n = x.len();
    if n <= 2 {
        return y.to_vec();
    }
    let k = ((frac * n as f64 + 1e-10) as usize).clamp(2, n);
    let mut residual_weights = vec![1.0; n];
    let mut fitted = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let mut residuals = Vec::with_capacity(n);

    for iter in 0..=LOWESS_ITERS {
        fitted.fill(0.0);
        let mut left_end = 0usize;
        let mut right_end = k;

        for i in 0..n {
            let xval = x[i];
            while right_end < n && xval > (x[left_end] + x[right_end]) / 2.0 {
                left_end += 1;
                right_end += 1;
            }
            let radius = (xval - x[left_end]).max(x[right_end - 1] - xval);
            let window = Range {
                start: left_end,
                end: right_end,
            };

            let mut nonzero_weights = 0usize;
            for j in window {
                let dist = ((x[j] - xval).abs() / radius).clamp(0.0, 1.0);
                let dist3 = dist * dist * dist;
                let tricube = 1.0 - dist3;
                let w = (tricube * tricube * tricube) * residual_weights[j];
                weights[j] = w;
                if w > 1e-12 {
                    nonzero_weights += 1;
                }
            }
            let sum_weights = numpy_pairwise_sum_f64(&weights[window]);

            if nonzero_weights < 2 || sum_weights <= 0.0 {
                fitted[i] = y[i];
                continue;
            }

            for weight in &mut weights[window] {
                *weight /= sum_weights;
            }
            let sum_weighted_x = window.into_iter().map(|j| weights[j] * x[j]).sum::<f64>();
            let weighted_sqdev_x = window
                .into_iter()
                .map(|j| weights[j] * (x[j] - sum_weighted_x).powf(2.0))
                .sum::<f64>()
                .max(1e-12);
            fitted[i] = window
                .into_iter()
                .map(|j| {
                    let projection = weights[j]
                        * (1.0
                            + (xval - sum_weighted_x) * (x[j] - sum_weighted_x) / weighted_sqdev_x);
                    projection * y[j]
                })
                .sum();
        }

        if iter == LOWESS_ITERS {
            break;
        }

        residuals.clear();
        residuals.extend(y.iter().zip(&fitted).map(|(yi, fi)| (yi - fi).abs()));
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n.is_multiple_of(2) {
            0.5 * (residuals[n / 2 - 1] + residuals[n / 2])
        } else {
            residuals[n / 2]
        };
        if median == 0.0 {
            for i in 0..n {
                residual_weights[i] = if (y[i] - fitted[i]).abs() > 0.0 {
                    0.0
                } else {
                    1.0
                };
            }
        } else {
            let scale = 6.0 * median;
            for i in 0..n {
                let u = ((y[i] - fitted[i]).abs() / scale).min(1.0);
                let bisquare = 1.0 - u * u;
                residual_weights[i] = bisquare * bisquare;
            }
        }
    }
    fitted
}

#[cfg(test)]
mod tests {
    use super::LinearInterp;

    #[test]
    fn linear_interp_matches_scipy_operation_order_exact_bits() {
        let interp = LinearInterp::new(vec![1.0, 3.0, 7.0], vec![0.1, 1.7, -0.2]);
        let scipy_reference = [
            (-2.0, 13_835_733_595_226_269_286),
            (1.0, 4_591_870_180_066_957_722),
            (2.0, 4_606_281_698_874_543_308),
            (3.0, 4_610_334_938_539_176_755),
            (5.0, 4_604_930_618_986_332_160),
            (7.0, 13_819_745_816_549_104_024),
            (9.0, 13_831_229_995_598_898_789),
        ];
        for (query, expected_bits) in scipy_reference {
            assert_eq!(interp.at(query).to_bits(), expected_bits, "query={query}");
        }
    }
}
