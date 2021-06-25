pub trait Cell {
    type Coord;

    fn basis(&self, dof: usize, x: Self::Coord) -> f64;
    fn basis_gradient(&self, dof: usize, x: Self::Coord) -> Self::Coord;
    fn global_coord(&self, x: Self::Coord) -> Self::Coord;
    fn global_dof(&self, dof: usize) -> usize;
    fn local_dofs(&self) -> usize;
}
