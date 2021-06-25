use fem_rs::integrate;
use fem_rs::cell::Cell;
use ndarray::{Array, Ix1, Ix2};
use ndarray_linalg::Solve;


fn set_boundary_condition(index: usize, value: f64, K: &mut Array<f64, Ix2>, F: &mut Array<f64, Ix1>) {
    K.row_mut(index).fill(0.0);
    K[[index, index]] = 1.0;
    F[index] = value;
}

struct Cell1D {
    pub order: usize,
    pub dof0: usize,
    pub x0: f64,
    pub w: f64,
}

impl Cell for Cell1D {
    type Coord = f64;

    fn basis(&self, dof: usize, x: Self::Coord) -> f64 {
        return fem_rs::lagrange(self.order, dof, x);
    }

    fn basis_gradient(&self, dof: usize, x: Self::Coord) -> Self::Coord {
        return fem_rs::lagrange_gradient(self.order, dof, x);
    }

    fn global_coord(&self, x: Self::Coord) -> Self::Coord {
        return self.x0 + 0.5*self.w*(x+1.0);
    }

    fn global_dof(&self, dof: usize) -> usize {
        return self.dof0 + dof;
    }

    fn local_dofs(&self) -> usize {
        return self.order+1;
    }
}

impl Cell1D {
    fn jacob(&self, _x: f64) -> f64 {
        return 0.5*self.w;
    }

    fn jacob_det(&self, _x: f64) -> f64 {
        return 0.5*self.w;
    }

    fn jacob_inv(&self, _x: f64) -> f64 {
        return 2.0/self.w;
    }
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

    let mut K = Array::<f64, Ix2>::zeros((Nel*order+1, Nel*order+1));
    let mut F = Array::<f64, Ix1>::zeros(Nel*order+1);

    for element in 0..Nel {
        let cell = Cell1D {
            order: order,
            dof0: element * order,
            x0: L/Nel as f64 * element as f64,
            w: L/Nel as f64
        };
        for localA in 0..cell.local_dofs() {
            let globalA = cell.global_dof(localA);
            F[globalA] += A*integrate(|x| cell.basis(localA, x) * f(cell.global_coord(x)) * cell.jacob_det(x));
            
            for localB in  0..cell.local_dofs() {
                let globalB = cell.global_dof(localB);
                K[[globalA, globalB]] += E*A*integrate(|x| cell.basis_gradient(localA, x) * cell.jacob_inv(x) * cell.basis_gradient(localB, x) * cell.jacob_inv(x) * cell.jacob_det(x));
            }
        }
    }

    set_boundary_condition(0, g1, &mut K, &mut F);
    set_boundary_condition(Nel*order, g2, &mut K, &mut F);
    let D = K.solve(&F).unwrap();

    println!("{:?}", K);
    println!("{:?}", F);
    println!("{:?}", D);
}
