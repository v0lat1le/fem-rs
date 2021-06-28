use fem_rs::integrate;
use fem_rs::cell::{Node, Cell};
use ndarray::{Array, Ix1, Ix2, aview2, aview_mut2};
use ndarray_linalg::{Determinant, Inverse, Solve};


fn set_boundary_condition(index: usize, value: f64, K: &mut Array<f64, Ix2>, F: &mut Array<f64, Ix1>) {
    K.row_mut(index).fill(0.0);
    K[[index, index]] = 1.0;
    F[index] = value;
}

struct Node1D {
    pub order: usize,
    pub index: usize,
    pub global: usize,
    pub coord: f64,
}

impl Node for Node1D {
    type Coord = f64;
    fn index(&self) -> usize { self.global }
    fn coord(&self) -> Self::Coord { self.coord }
    fn basis(&self, x: &Self::Coord) -> f64 { fem_rs::lagrange(self.order, self.index, *x) }
    fn basis_grad(&self, x: &Self::Coord) -> Self::Coord  { fem_rs::lagrange_gradient(self.order, self.index, *x) }
}

struct Cell1D {
    pub order: usize,
    pub dof0: usize,
    pub x0: f64,
    pub w: f64,
}

impl Cell for Cell1D {
    type Coord = f64;
    type Jacob = [[f64; 1]; 1];

    fn interp(&self, x: &Self::Coord) -> Self::Coord {
        return self.nodes().map(|n| n.basis(x)*n.coord()).sum();
    }

    fn jacob(&self, x: &Self::Coord) -> Self::Jacob {
        return [[self.nodes().map(|n| n.basis_grad(x)*n.coord()).sum()]];
    }
}

impl Cell1D {
    fn nodes(&self) -> impl Iterator<Item=Node1D> + '_ {
        return (0..=self.order).map(move |index| Node1D {
            order: self.order,
            index: index,
            global: self.dof0 + index,
            coord: self.x0 + self.w*index as f64/self.order as f64
        });
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
        for nodeA in cell.nodes() {
            F[nodeA.index()] += A*integrate(|x| {
                let jacob_data = cell.jacob(&x);
                let jacob = aview2(&jacob_data);
                return nodeA.basis(&x) * f(cell.interp(&x)) * jacob.det().unwrap();
            });
            
            for nodeB in cell.nodes() {
                K[[nodeA.index(), nodeB.index()]] += E*A*integrate(|x| {
                    let mut jacob_data = cell.jacob(&x);
                    let mut jacob = aview_mut2(&mut jacob_data);
                    let jacob_det = jacob.det().unwrap();
                    let jacob_inv = jacob.inv().unwrap();
                    return nodeA.basis_grad(&x) * jacob_inv[[0,0]] * nodeB.basis_grad(&x) * jacob_inv[[0,0]] * jacob_det;
                });
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
