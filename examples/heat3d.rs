use fem_rs::cell::{Node, Cell};
use fem_rs::{integrate};
use ndarray::{Array, Ix1, Ix2, aview1, aview_mut2};
use ndarray_linalg::{Determinant, Inverse, Solve};
use std::fs::File;
use std::io::Write;

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

fn main() {
    let order = 1;
    let x_cells = 4;
    let y_cells = 8;
    let z_cells = 2;
    let n_nodes = (x_cells*order+1)*(y_cells*order+1)*(z_cells*order+1);
    let kappa = Array::<f64, Ix2>::eye(3)*385.0f64;
    
    let mut K = Array::<f64, Ix2>::zeros((n_nodes, n_nodes));
    let mut F = Array::<f64, Ix1>::zeros(n_nodes);

    for cellx in 0..x_cells {
        for celly in 0..y_cells {
            for cellz in 0..z_cells {
                let nodes = [
                    Node3D{order: 1, index: 0, global: cellx + (x_cells+1)*(celly + cellz*(y_cells+1)), coord: [cellx as f64*0.01, celly as f64*0.01, cellz as f64*0.01]},
                    Node3D{order: 1, index: 1, global: cellx+1 + (x_cells+1)*(celly + cellz*(y_cells+1)), coord: [(cellx+1) as f64*0.01, celly as f64*0.01, cellz as f64*0.01]},
                    Node3D{order: 1, index: 2, global: cellx + (x_cells+1)*(celly+1 + cellz*(y_cells+1)), coord: [cellx as f64*0.01, (celly+1) as f64*0.01, cellz as f64*0.01]},
                    Node3D{order: 1, index: 3, global: cellx+1 + (x_cells+1)*(celly+1 + cellz*(y_cells+1)), coord: [(cellx+1) as f64*0.01, (celly+1) as f64*0.01, cellz as f64*0.01]},
                    Node3D{order: 1, index: 4, global: cellx + (x_cells+1)*(celly + (cellz+1)*(y_cells+1)), coord: [cellx as f64*0.01, celly as f64*0.01, (cellz+1) as f64*0.01]},
                    Node3D{order: 1, index: 5, global: cellx+1 + (x_cells+1)*(celly + (cellz+1)*(y_cells+1)), coord: [(cellx+1) as f64*0.01, celly as f64*0.01, (cellz+1) as f64*0.01]},
                    Node3D{order: 1, index: 6, global: cellx + (x_cells+1)*(celly+1 + (cellz+1)*(y_cells+1)), coord: [cellx as f64*0.01, (celly+1) as f64*0.01, (cellz+1) as f64*0.01]},
                    Node3D{order: 1, index: 7, global: cellx+1 + (x_cells+1)*(celly+1 + (cellz+1)*(y_cells+1)), coord: [(cellx+1) as f64*0.01, (celly+1) as f64*0.01, (cellz+1) as f64*0.01]},
                ];
                let cell = Cell3D {
                    order: 1,
                    nodes: nodes
                };

                for nodeA in cell.nodes() {
                    // F[nodeA.index()] += A*integrate(|x| {
                    //     let jacob_data = cell.jacob(&x);
                    //     let jacob = aview2(&jacob_data);
                    //     return nodeA.basis(&x) * f(cell.interp(&x)) * jacob.det().unwrap();
                    // });
                    
                    for nodeB in cell.nodes() {
                        K[[nodeA.index(), nodeB.index()]] += integrate(|x0| integrate(|x1| integrate(|x2| {
                            let x = [x0, x1, x2];
                            let mut jacob_data = cell.jacob(&x);
                            let jacob = aview_mut2(&mut jacob_data);
                            let jacob_det = jacob.det().unwrap();
                            let jacob_inv = jacob.inv().unwrap();
                            return aview1(&nodeA.basis_grad(&x)).dot(&jacob_inv).dot(&kappa).dot(&aview1(&nodeB.basis_grad(&x)).dot(&jacob_inv)) * jacob_det;
                        })));
                    }
                }
            }
        }
    }

    
    for nodey in 0..=y_cells {
        let y = nodey as f64*0.01;
        for nodez in 0..=z_cells {
            let z = nodez as f64*0.01;
            let left = (x_cells+1)*(nodey + nodez*(y_cells+1));
            let right = x_cells + (x_cells+1)*(nodey + nodez*(y_cells+1));
            set_boundary_condition(left, 300.0*(1.0 + (y+z)/3.0), &mut K, &mut F);
            set_boundary_condition(right, 310.0*(1.0 + (y+z)/3.0), &mut K, &mut F);
        }
    }

    let D = K.solve(&F).unwrap();

    let mut file = File::create("solution.vtk").unwrap();
    writeln!(file, "# vtk DataFile Version 3.0");
    writeln!(file, "# This file was generated by the fem-rs library");
    writeln!(file, "ASCII");
    writeln!(file, "DATASET UNSTRUCTURED_GRID");
    writeln!(file, "POINTS {} double", n_nodes);
    for nodez in 0..=z_cells {
        for nodey in 0..=y_cells {
            for nodex in 0..=x_cells {
                writeln!(file, "{} {} {}", nodex as f64*0.01, nodey as f64*0.01, nodez as f64*0.01);
            }
        }
    }
    writeln!(file, "CELLS {} {}", x_cells*y_cells*z_cells, x_cells*y_cells*z_cells*9);
    for cellz in 0..z_cells {
        for celly in 0..y_cells {
            for cellx in 0..x_cells {
                writeln!(file, "8 {} {} {} {} {} {} {} {}", 
                cellx + (x_cells+1)*(celly + cellz*(y_cells+1)),
                cellx+1 + (x_cells+1)*(celly + cellz*(y_cells+1)),
                cellx+1 + (x_cells+1)*(celly+1 + cellz*(y_cells+1)),
                cellx + (x_cells+1)*(celly+1 + cellz*(y_cells+1)),
                cellx + (x_cells+1)*(celly + (cellz+1)*(y_cells+1)),
                cellx+1 + (x_cells+1)*(celly + (cellz+1)*(y_cells+1)),
                cellx+1 + (x_cells+1)*(celly+1 + (cellz+1)*(y_cells+1)),
                cellx + (x_cells+1)*(celly+1 + (cellz+1)*(y_cells+1)));
            }
        }
    }
    writeln!(file, "CELL_TYPES {}", x_cells*y_cells*z_cells);
    writeln!(file, "{}", "12 ".repeat(x_cells*y_cells*z_cells));

    writeln!(file, "POINT_DATA {}", n_nodes);
    writeln!(file, "SCALARS D double 1");
    writeln!(file, "LOOKUP_TABLE default");
    for v in D.iter() {
        write!(file, "{} ", v);
    }
    writeln!(file, "");

    println!("{:?}", K);
    println!("{:?}", F);
    println!("{:?}", D);
}
