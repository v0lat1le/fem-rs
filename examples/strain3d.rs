use fem_rs::cell::{Node, Cell};
use fem_rs::{integrate};
use ndarray::{Array, Ix1, Ix2, aview1, aview2};
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

fn kappa(j: usize, l: usize) -> Array<f64, Ix2> {
    let E= 2.0e11;
    let nu= 0.3;
    let lambda= (E*nu)/((1.+nu)*(1.-2.*nu));
    let mu= E/(2.*(1.+nu));
    let mut r = Array::<f64, Ix2>::zeros((3, 3));
    for i in 0..3 {
        for k in 0..3 {
            r[[i, k]] = lambda*(i==j && k==l) as usize as f64 + mu*((i==k && j==l) as usize + (i==l && j==k) as usize) as f64;
        }
    }
    return r;
}

fn main() {
    let mut dataset = fem_rs::io::read_dataset(&mut std::io::BufReader::new(File::open("data/strain3d.vtk").unwrap())).unwrap();
    let n_dof = dataset.points.len()*3;

    let mut K = Array::<f64, Ix2>::zeros((n_dof, n_dof));
    let mut F = Array::<f64, Ix1>::zeros(n_dof);

    for cell in 0..dataset.cells.len() {
        let cell = make_cell(cell, &dataset);

        for node_a in cell.nodes() {
            for i in 0..3 {
                if i==2 && node_a.coord()[2] == 1.0 {
                    F[node_a.index()*3+i] += integrate(|x0| integrate(|x1| { 
                        let x = [x0, x1, 1.0];
                        let mut jacob_data = cell.jacob(&x);
                        jacob_data[0][2] = 0.0;  // replacing dT/dx3 part of J with normal
                        jacob_data[1][2] = 0.0;  // to the cell face in global coords
                        jacob_data[2][2] = 1.0;  // HACK: assuming axis aligned mapping here
                        let face_jacob_det = aview2(&jacob_data).det().unwrap();
                        return node_a.basis(&x) * 1.0e9*cell.interp(&x)[0] * face_jacob_det;
                    }));
                }
                
                for node_b in cell.nodes() {
                    for j in 0..3 {
                        K[[node_a.index()*3+i, node_b.index()*3+j]] += integrate(|x0| integrate(|x1| integrate(|x2| {
                            let x = [x0, x1, x2];
                            let jacob_data = cell.jacob(&x);
                            let jacob = aview2(&jacob_data);
                            let jacob_det = jacob.det().unwrap();
                            let jacob_inv = jacob.inv().unwrap();
                            return aview1(&node_a.basis_grad(&x)).dot(&jacob_inv).dot(&kappa(i,j)).dot(&aview1(&node_b.basis_grad(&x)).dot(&jacob_inv)) * jacob_det;
                        })));
                    }
                }
            }
        }
    }

    for (index, point) in dataset.points.iter().enumerate() {
        if point.z == 0.0 {
            set_boundary_condition(index*3+0, 0.0, &mut K, &mut F);
            set_boundary_condition(index*3+1, 0.0, &mut K, &mut F);
            set_boundary_condition(index*3+2, 0.0, &mut K, &mut F);
        }
    }

    let D = K.solve(&F).unwrap();

    println!("{:?}", K);
    println!("{:?}", F);
    println!("{:?}", D);

    let d_len = D.len();
    let u = D.into_shape((d_len/3, 3)).unwrap().outer_iter().map(|r| fem_rs::io::Point{x: r[0], y: r[1], z: r[2]}).collect();
    dataset.point_data.insert("u".to_owned(), fem_rs::io::Attributes::Vector(u));
    let mut file = File::create("solution.vtk").unwrap();
    fem_rs::io::write_dataset(&mut file, &dataset).unwrap();
}
