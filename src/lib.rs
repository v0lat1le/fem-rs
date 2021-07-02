pub mod cell;
pub mod io;

pub fn xi(order: usize, index: usize) -> f64 {
    return -1.0 + 2.0*index as f64/order as f64;
}

pub fn lagrange(order: usize, index: usize, x: f64) -> f64 {
    let mut result = 1.0;
    for m in 0..=order {
        if m == index { continue; }
        result *= (x - xi(order, m))/(xi(order, index) - xi(order, m));
    }
    return result;
}

pub fn lagrange_gradient(order: usize, index: usize, x: f64) -> f64 {
    let mut result = 0.0;
    for i in 0..=order {
        if i == index { continue; }
        let mut inner = 1.0;
        for m in 0..=order {
            if m == index ||  m == i { continue; }
            inner *= (x - xi(order, m))/(xi(order, index) - xi(order, m));
        }
        result += inner / (xi(order, index) - xi(order, i));
    }
    return result;
}

pub fn integrate<F>(f: F) -> f64
    where F: Fn(f64)->f64 {
    let xs = [
         (3./7. - 2.*(6./5. as f64).sqrt()/7.).sqrt(),
        -(3./7. - 2.*(6./5. as f64).sqrt()/7.).sqrt(),
         (3./7. + 2.*(6./5. as f64).sqrt()/7.).sqrt(),
        -(3./7. + 2.*(6./5. as f64).sqrt()/7.).sqrt()];
    let ws = [
        (18. + 30_f64.sqrt())/36.,
        (18. + 30_f64.sqrt())/36.,
        (18. - 30_f64.sqrt())/36.,
        (18. - 30_f64.sqrt())/36.];
    return f(xs[0])*ws[0] + f(xs[1])*ws[1] + f(xs[2])*ws[2] + f(xs[3])*ws[3];
}

#[cfg(test)]
mod tests {
    use super::*;
    
    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr, $eps:expr) => {{
            let (a, b) = (&$a, &$b);
            let eps = $eps;
            assert!(
                (*a - *b).abs() < eps,
                "assertion failed: `(left !== right)` \
                (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
                *a,
                *b,
                eps,
                (*a - *b).abs()
            );
        }};
    }

    #[test]
    fn test_lagrange() {
        assert_eq!(lagrange(1, 0, -1.0), 1.0);
        assert_eq!(lagrange(1, 0,  0.0), 0.5);
        assert_eq!(lagrange(1, 0,  1.0), 0.0);
        assert_eq!(lagrange(1, 1, -1.0), 0.0);
        assert_eq!(lagrange(1, 1,  0.0), 0.5);
        assert_eq!(lagrange(1, 1,  1.0), 1.0);

        assert_eq!(lagrange(2, 0, -1.0), 1.0);
        assert_eq!(lagrange(2, 0,  0.0), 0.0);
        assert_eq!(lagrange(2, 0,  1.0), 0.0);
        assert_eq!(lagrange(2, 1, -1.0), 0.0);
        assert_eq!(lagrange(2, 1,  0.0), 1.0);
        assert_eq!(lagrange(2, 1,  1.0), 0.0);
        assert_eq!(lagrange(2, 2, -1.0), 0.0);
        assert_eq!(lagrange(2, 2,  0.0), 0.0);
        assert_eq!(lagrange(2, 2,  1.0), 1.0);

    }

    #[test]
    fn test_lagrange_gradient() {
        assert_eq!(lagrange_gradient(1, 0, -1.0), -0.5);
        assert_eq!(lagrange_gradient(1, 0,  0.0), -0.5);
        assert_eq!(lagrange_gradient(1, 0,  1.0), -0.5);
        assert_eq!(lagrange_gradient(1, 1, -1.0),  0.5);
        assert_eq!(lagrange_gradient(1, 1,  0.0),  0.5);
        assert_eq!(lagrange_gradient(1, 1,  1.0),  0.5);

        assert_eq!(lagrange_gradient(2, 0, -1.0), -1.5);
        assert_eq!(lagrange_gradient(2, 0,  0.0), -0.5);
        assert_eq!(lagrange_gradient(2, 0,  1.0),  0.5);
        assert_eq!(lagrange_gradient(2, 1, -1.0),  2.0);
        assert_eq!(lagrange_gradient(2, 1,  0.0),  0.0);
        assert_eq!(lagrange_gradient(2, 1,  1.0), -2.0);
        assert_eq!(lagrange_gradient(2, 2, -1.0), -0.5);
        assert_eq!(lagrange_gradient(2, 2,  0.0),  0.5);
        assert_eq!(lagrange_gradient(2, 2,  1.0),  1.5);
    }

    #[test]
    fn test_itegrate() {
        assert_eq!(integrate(|_x| 1.0), 2.0);

        assert_eq!(integrate(|x| lagrange(1, 0, x)), 1.0);
        assert_eq!(integrate(|x| lagrange(1, 1, x)), 1.0);

        assert_eq!(integrate(|x| lagrange(2, 0, x)), 1.0/3.0);
        assert_approx_eq!(integrate(|x| lagrange(2, 1, x)), 4.0/3.0, 1e-15);
        assert_eq!(integrate(|x| lagrange(2, 2, x)), 1.0/3.0);

        assert_eq!(integrate(|x| lagrange_gradient(2, 0, x)), -1.0);
        assert_eq!(integrate(|x| lagrange_gradient(2, 1, x)),  0.0);
        assert_eq!(integrate(|x| lagrange_gradient(2, 2, x)),  1.0);
    }
}
