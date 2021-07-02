use std::fmt;
use std::error::Error;
use std::io::BufRead;
use std::str::FromStr;

enum Format {
    ASCII,
}

pub enum DatsetType {
    UNSTRUCTURED_GRID,
}

pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

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

pub struct Dataset {
    pub type_: DatsetType,
    pub points: Vec<Point>,
    pub cells: Vec<Cell>,
}

#[derive(Debug, Clone, PartialEq, Eq)]pub struct ParseError {
    msg: &'static str
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

pub fn parse(reader: &mut dyn BufRead) -> Result<Dataset> {
    let _format = read_header(reader)?;
    return read_dataset(reader);
}

fn error(msg: &'static str) -> ParseError {
    return ParseError{msg};
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

fn read_dataset(reader: &mut dyn BufRead) -> Result<Dataset> {
    let mut tokens= reader.lines().flat_map(|r| match r {
        Err(e) => vec![Err(ParseError::from(e))],
        Ok(s) => s.split_whitespace().map(|s| Ok(s.to_owned())).collect(),
    });
    
    let type_ = read_type(&mut tokens)?;
    let points = read_points(&mut tokens)?;
    let cells = read_cells(&mut tokens)?;

    return Ok(Dataset{type_, points, cells});
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
        let dataset = parse(&mut input).unwrap();
        assert_eq!(dataset.points.len(), 8);
        assert_eq!(dataset.points[3].x, 0.01);
        assert_eq!(dataset.cells.len(), 1);
        assert_eq!(dataset.cells[0].points.len(), 8);
        assert_eq!(dataset.cells[0].points[3], 2);
    }
}
