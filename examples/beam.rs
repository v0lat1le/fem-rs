use fem_rs::integrate;
use ndarray::{Array, Ix1, Ix2};
use ndarray_linalg::Solve;

fn to_global(order: usize, element: usize, dof: usize) -> usize {
    return element * order + dof;
}

fn set_boundary_condition(index: usize, value: f64, K: &mut Array<f64, Ix2>, F: &mut Array<f64, Ix1>) {
    K.row_mut(index).fill(0.0);
    K[[index, index]] = 1.0;
    F[index] = value;
}

fn main() {
    let g1 = 0.0;
    let g2 = 0.001;
    let E = 1e11;
    let A = 1e-4;
    let f = |x: f64| x*1e11;
    let L = 0.1;
    let h = 1e6;

    let Nel = 10;
    let order = 3;

    let he = L/Nel as f64;

    let mut K = Array::<f64, Ix2>::zeros((Nel*order+1, Nel*order+1));
    let mut F = Array::<f64, Ix1>::zeros(Nel*order+1);

    for element in 0..Nel {
        for localA in 0..=order {
            let globalA = to_global(order, element, localA);
            F[globalA] += 0.5*A*he*integrate(|x| fem_rs::lagrange(order, localA, x) * f(he*(element as f64+0.5*(x+1.0))));
            
            for localB in 0..=order {
                let globalB = to_global(order, element, localB);
                K[[globalA, globalB]] += 2.0*E*A/he*integrate(|x| fem_rs::lagrange_gradient(order, localA, x)*fem_rs::lagrange_gradient(order, localB, x));
            }
        }
    }

    set_boundary_condition(0, g1, &mut K, &mut F);
    set_boundary_condition(Nel*order, g2, &mut K, &mut F);
    let X = K.solve(&F).unwrap();

    println!("{:?}", K);
    println!("{:?}", F);
    println!("{:?}", X);
}
