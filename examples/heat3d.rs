use fem_rs::cell::{Node, Cell};
use fem_rs::{integrate};
use ndarray::{Array, Ix1, Ix2, aview1, aview_mut2};
use ndarray_linalg::{Determinant, Inverse, Solve};
use std::fs::File;

fn set_boundary_condition(index: usize, value: f64, K: &mut Array<f64, Ix2>,  F: &mut Array<f64, Ix1>) {
    K.row_mut(index).fill(0.0);
    K[[index, index]] = 1.0;
    F[index] = value;
}

struct Node3D {
    pub order: usize,
    pub index: usize,
    pub global: usize,
    pub coord: <Node3D as Node>::Coord,
}

impl Node for Node3D {
    type Coord = [f64; 3];

    fn index(&self) -> usize { self.global }
    fn coord(&self) -> Self::Coord { self.coord }

    fn basis(&self, x: &Self::Coord) -> f64 {
        let nnodes = self.order+1;
        let dofx = self.index % nnodes;
        let dofy = (self.index / nnodes) % nnodes;
        let dofz = (self.index / nnodes) / nnodes;
        return fem_rs::lagrange(self.order, dofx, x[0])
            * fem_rs::lagrange(self.order, dofy, x[1])
            * fem_rs::lagrange(self.order, dofz, x[2]);
    }

    fn basis_grad(&self, x: &Self::Coord) -> Self::Coord {
        let nnodes = self.order+1;
        let dofx = self.index % nnodes;
        let dofy = (self.index / nnodes) % nnodes;
        let dofz = (self.index / nnodes) / nnodes;
        return [
            fem_rs::lagrange_gradient(self.order, dofx, x[0])
                * fem_rs::lagrange(self.order, dofy, x[1])
                * fem_rs::lagrange(self.order, dofz, x[2]),
            fem_rs::lagrange(self.order, dofx, x[0])
                * fem_rs::lagrange_gradient(self.order, dofy, x[1])
                * fem_rs::lagrange(self.order, dofz, x[2]),
            fem_rs::lagrange(self.order, dofx, x[0])
                * fem_rs::lagrange(self.order, dofy, x[1])
                * fem_rs::lagrange_gradient(self.order, dofz, x[2]),
        ];
    }
}

struct Cell3D {
    pub order: usize,
    pub nodes: [Node3D; 8],
}

impl Cell for Cell3D {
    type Coord = [f64; 3];
    type Jacob = [Self::Coord; 3];

    fn interp(&self, x: &Self::Coord) -> Self::Coord {
        let mut result = [0.0, 0.0, 0.0];
        for node in self.nodes() {
            let coord = node.coord();
            let weight = node.basis(x);
            for i in 0..3 {
                result[i] += weight*coord[i];
            }
        }
        return result; 
    }

    fn jacob(&self, x: &Self::Coord) -> Self::Jacob {
        let mut result = [[0.0; 3]; 3];

        for node in self.nodes() {        
            let grad = node.basis_grad(x);
            let coord = node.coord();
            for i in 0..3 {
                for j in 0..3 {
                    result[i][j] += grad[j]*coord[i];
                }
            }
        }
        
        return result;
    }
}

impl Cell3D {
    fn nodes(&self) -> &[Node3D; 8] {
        &self.nodes
    }
}

fn make_node(index: usize, point: usize, dataset: &fem_rs::io::Dataset) -> Node3D {
    Node3D {
        order: 1,
        index: index,
        global: point,
        coord: [dataset.points[point].x, dataset.points[point].y, dataset.points[point].z]
    }
}

fn make_cell(index: usize, dataset: &fem_rs::io::Dataset) -> Cell3D {
    let cell = &dataset.cells[index];
    let nodes = [
        make_node(0, cell.points[0], dataset),
        make_node(1, cell.points[1], dataset),
        make_node(2, cell.points[3], dataset),
        make_node(3, cell.points[2], dataset),
        make_node(4, cell.points[4], dataset),
        make_node(5, cell.points[5], dataset),
        make_node(6, cell.points[7], dataset),
        make_node(7, cell.points[6], dataset),
    ];
    return Cell3D {
        order: 1,
        nodes: nodes
    };
}

fn main() {
    let mut dataset = fem_rs::io::read_dataset(
        &mut std::io::BufReader::new(File::open("data/heat3d.vtk").unwrap())).unwrap();
    let n_dof = dataset.points.len();
    let kappa = Array::<f64, Ix2>::eye(3)*385.0f64;
    
    let mut K = Array::<f64, Ix2>::zeros((n_dof, n_dof));
    let mut F = Array::<f64, Ix1>::zeros(n_dof);
    for cell in 0..dataset.cells.len() {
        let cell = make_cell(cell, &dataset);

        for node_a in cell.nodes() {
            // F[nodeA.index()] += A*integrate(|x| {
            //     let jacob_data = cell.jacob(&x);
            //     let jacob = aview2(&jacob_data);
            //     return nodeA.basis(&x) * f(cell.interp(&x)) * jacob.det().unwrap();
            // });
            
            for node_b in cell.nodes() {
                K[[node_a.index(), node_b.index()]] += integrate(|x0| integrate(|x1| integrate(|x2| {
                    let x = [x0, x1, x2];
                    let mut jacob_data = cell.jacob(&x);
                    let jacob = aview_mut2(&mut jacob_data);
                    let jacob_det = jacob.det().unwrap();
                    let jacob_inv = jacob.inv().unwrap();
                    return aview1(&node_a.basis_grad(&x)).dot(&jacob_inv).dot(&kappa).dot(&aview1(&node_b.basis_grad(&x)).dot(&jacob_inv)) * jacob_det;
                })));
            }
        }
    }

    for (index, point) in dataset.points.iter().enumerate() {
        if point.x == 0.0 {
            set_boundary_condition(index, 300.0*(1.0 + (point.y+point.z)/3.0), &mut K, &mut F);
        } else if point.x == 0.04 {
            set_boundary_condition(index, 310.0*(1.0 + (point.y+point.z)/3.0), &mut K, &mut F);
        }
    }

    let D = K.solve(&F).unwrap();

    println!("{:?}", K);
    println!("{:?}", F);
    println!("{:?}", D);

    dataset.point_data.insert("D".to_owned(), fem_rs::io::Attributes::Scalar(D.into_raw_vec()));
    let mut file = File::create("solution.vtk").unwrap();
    fem_rs::io::write_dataset(&mut file, &dataset).unwrap();
}
