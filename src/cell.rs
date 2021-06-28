pub trait Node {
    type Coord;
    
    fn index(&self) -> usize;
    fn coord(&self) -> Self::Coord;
    fn basis(&self, x: &Self::Coord) -> f64;
    fn basis_grad(&self, x: &Self::Coord) -> Self::Coord;
}

pub trait Cell {
    type Coord;
    type Jacob;

    fn interp(&self, x: &Self::Coord) -> Self::Coord;
    fn jacob(&self, x: &Self::Coord) -> Self::Jacob;
}
