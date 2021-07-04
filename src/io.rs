use std::collections::HashMap;
use std::error::Error;
use std::io::Write;
use std::io::BufRead;
use std::str::FromStr;

#[derive(Copy, Clone)]
enum Format {
    ASCII,
}

#[derive(Copy, Clone)]
pub enum DatsetType {
    UNSTRUCTURED_GRID,
}

#[derive(Copy, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Copy, Clone)]
pub enum CellType {
    HEXAHEDRON = 12
}

impl FromStr for CellType {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self> {
        let val = usize::from_str(s)?;
        if val != CellType::HEXAHEDRON as usize {
            return Err(error("Unsupported cell type"));
        }
        return Ok(CellType::HEXAHEDRON);
    }
}

pub struct Cell {
    pub type_: CellType,
    pub points: Vec<usize>,
}

pub enum Attributes {
    Scalar(Vec<f64>),
    Vector(Vec<Point>),
}

pub struct Dataset {
    pub type_: DatsetType,
    pub points: Vec<Point>,
    pub cells: Vec<Cell>,
    pub point_data: HashMap<String, Attributes>,
    pub cell_data: HashMap<String, Attributes>,
}

#[derive(Debug, Clone, PartialEq, Eq)]pub struct ParseError {
    msg: &'static str
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.msg.fmt(f)
    }
}

impl Error for ParseError {
}

impl From<std::io::Error> for ParseError {
    fn from(_: std::io::Error) -> Self {
        ParseError{msg: "io error"}
    }
}

impl From<std::num::ParseIntError> for ParseError {
    fn from(_: std::num::ParseIntError) -> Self {
        ParseError{msg: "int parse error"}
    }
}

impl From<std::num::ParseFloatError> for ParseError {
    fn from(_: std::num::ParseFloatError) -> Self {
        ParseError{msg: "float parse error"}
    }
}

type Result<T> = std::result::Result<T, ParseError>;

fn error(msg: &'static str) -> ParseError {
    return ParseError{msg};
}

pub fn read_dataset(reader: &mut dyn BufRead) -> Result<Dataset> {
    let _format = read_header(reader)?;

    let mut tokens= reader.lines().flat_map(|r| match r {
        Err(e) => vec![Err(ParseError::from(e))],
        Ok(s) => s.split_whitespace().map(|s| Ok(s.to_owned())).collect(),
    });
    
    let type_ = read_type(&mut tokens)?;
    let points = read_points(&mut tokens)?;
    let cells = read_cells(&mut tokens)?;

    return Ok(Dataset{type_, points, cells, point_data: HashMap::new(), cell_data: HashMap::new()});
}
fn read_header(reader: &mut dyn BufRead) -> Result<Format> {
    let mut lines = reader.lines();
    match lines.next() {
        None => return Err(error("Missing version and identifier")),
        Some(Err(e)) => return Err(ParseError::from(e)),
        Some(Ok(l)) => if !l.starts_with("# vtk DataFile Version") { return Err(error("Not a valid Legacy VTK file")); }
    }
    lines.next(); // skip header
    match lines.next() {
        None => return Err(error("Missing format specification")),
        Some(Err(e)) => return Err(ParseError::from(e)),
        Some(Ok(l)) => if !l.starts_with("ASCII") { return Err(error("Invalid format specification")); }
    }
    return Ok(Format::ASCII);
}

fn read_type(tokens: &mut dyn Iterator<Item=Result<String>>) -> Result<DatsetType> {
    match tokens.next() {
        Some(Ok(w)) if !w.starts_with("DATASET") =>  { return Err(error("Expected DATASET keyword")) },
        Some(Err(e)) => return Err(e),
        None => return Err(error("Expected DATASET keyword")),
        _ => ()
    }

    match tokens.next() {
        Some(Ok(w)) if !w.starts_with("UNSTRUCTURED_GRID") => return Err(error("Unsupported dataset type")),
        Some(Err(e)) => return Err(e),
        None => return Err(error("Expected dataset type specification")),
        _ => return Ok(DatsetType::UNSTRUCTURED_GRID)
    }
}

fn read_points(tokens: &mut dyn Iterator<Item=Result<String>>) -> Result<Vec<Point>> {
    match tokens.next() {
        Some(Ok(w)) if !w.starts_with("POINTS") => return Err(error("Expected POINTS keyword")),
        Some(Err(e)) => return Err(e),
        None => return Err(error("Expected POINTS keyword")),
        _ => ()
    }
    let n_points: usize = read_value(tokens)?;
    tokens.next(); // skip datatype, assume float

    let mut points = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(read_point(tokens)?);
    }
    return Ok(points);
}

fn read_cells(tokens: &mut dyn Iterator<Item=Result<String>>) -> Result<Vec<Cell>> {
    match tokens.next() {
        Some(Ok(w)) if !w.starts_with("CELLS") => return Err(error("Expected CELLS keyword")),
        Some(Err(e)) => return Err(e),
        None => return Err(error("Expected CELLS keyword")),
        _ => ()
    }
    let n_cells: usize = read_value(tokens)?;
    let _n_values: usize = read_value(tokens)?;

    let mut cells = Vec::with_capacity(n_cells);
    for _ in 0..n_cells {
        cells.push(read_cell(tokens)?);
    }

    match tokens.next() {
        Some(Ok(w)) if !w.starts_with("CELL_TYPES") => return Err(error("Expected CELL_TYPES keyword")),
        Some(Err(e)) => return Err(e),
        None => return Err(error("Expected CELL_TYPES keyword")),
        _ => ()
    }
    let n_cell_typess: usize = read_value(tokens)?;
    for i in 0..n_cell_typess {
        cells[i].type_ = read_value(tokens)?;
    }

    return Ok(cells);
}

fn read_point(tokens: &mut dyn Iterator<Item=Result<String>>) -> Result<Point> {
    return Ok(Point{
        x: read_value(tokens)?,
        y: read_value(tokens)?,
        z: read_value(tokens)?});
}

fn read_cell(tokens: &mut dyn Iterator<Item=Result<String>>) -> Result<Cell> {
    let type_ = CellType::HEXAHEDRON;
    let n_points: usize = read_value(tokens)?;
    let mut points = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(read_value(tokens)?);
    }
    return Ok(Cell{type_, points});
}

fn read_value<T: FromStr>(tokens: &mut dyn Iterator<Item=Result<String>>) -> Result<T> 
where ParseError: From<T::Err> {
    match tokens.next() {
        Some(Ok(w)) => return Ok(T::from_str(&w)?),
        Some(Err(e)) => return Err(e),
        None => return Err(error("Expected a value")),
    }
}

pub fn write_dataset(file: &mut dyn Write, dataset: &Dataset) -> Result<()> {
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "# This file was generated by the fem-rs library")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(file, "POINTS {} double", dataset.points.len())?;
    for point in &dataset.points {
        writeln!(file, "{} {} {}", point.x, point.y, point.z)?;
    }
    writeln!(file, "CELLS {} {}", dataset.cells.len(), 9*dataset.cells.len())?;
    for cell in &dataset.cells {
        write!(file, "{} ", cell.points.len())?;
        for point in &cell.points {
            write!(file, "{} ", point)?;
        }
        writeln!(file)?;
    }
    writeln!(file, "CELL_TYPES {}", dataset.cells.len())?;
    for cell in &dataset.cells {
        write!(file, "{} ", cell.type_ as usize)?;
    }
    writeln!(file)?;

    if !dataset.point_data.is_empty() {
        writeln!(file, "POINT_DATA {}", dataset.points.len())?;
        for (name, attributes) in &dataset.point_data {
            match attributes {
                Attributes::Scalar(data) => {
                    writeln!(file, "SCALARS {} double 1", name)?;
                    writeln!(file, "LOOKUP_TABLE default")?;
                    for v in data {
                        write!(file, "{} ", v)?;
                    }
                    writeln!(file)?;
                },
                Attributes::Vector(data) => {
                    writeln!(file, "VECTORS {} double", name)?;
                    for point in data {
                        writeln!(file, "{} {} {}", point.x, point.y, point.z)?;
                    }
                }
            }
        }
    }
    return Ok(());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header() {
        let mut input = b"# vtk DataFile Version 3.0
test dataset
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 8 double
0 0 0
0.01 0 0
0 0.01 0
0.01 0.01 0
0 0 0.01
0.01 0 0.01
0 0.01 0.01
0.01 0.01 0.01
CELLS 1 9
8 0 1 3 2 4 5 7 6
CELL_TYPES 1
12" as &[u8];
        let dataset = read_dataset(&mut input).unwrap();
        assert_eq!(dataset.points.len(), 8);
        assert_eq!(dataset.points[3].x, 0.01);
        assert_eq!(dataset.cells.len(), 1);
        assert_eq!(dataset.cells[0].points.len(), 8);
        assert_eq!(dataset.cells[0].points[3], 2);
    }
}
