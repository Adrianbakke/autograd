use std::ops::Mul;
use std::iter::Sum;
use std::cmp::Ordering::Equal;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub elements: Vec<T>,    
    pub M: usize,
    pub N: usize,
}
                                                              
impl<T> Matrix<T> 
    where T: Clone + Copy + PartialEq + PartialOrd + Mul<Output = T> + Sum<T> {

    pub fn new(elem: Vec<T>, M: usize, N: usize) -> Self {
        let l = elem.len();
        Matrix {
            elements: elem,
            M: M,
            N: N,
        }
    }
    
    //creates a square matrix
    pub fn new_sqr(elem: Vec<T>) -> Matrix<T> {
        let dim = (elem.len() as f64).sqrt();
        Matrix {
            elements: elem,
            M: dim as usize,
            N: dim as usize,
        }
    }
    
    //TODO: write tests for these operations
    //      Make it work on slices
    pub fn get_column(&self, col: usize) -> Vec<T> {
        let elem = self.elements.to_vec();
        let mut column = Vec::new();
        let M = self.M as usize;
        let N = self.N as usize;
        for n in 0..M {
            //push elements at column pos to column array
            column.push(elem[n*N+col])
        }
        column
    }
    
    // r is the choosen row
    pub fn get_row(&self, r: usize) -> Vec<T> {
        let elem = self.elements.to_vec();
        let mut row = Vec::new();
        let N = self.N as usize;
        for n in 0..N {
            //push elements at column pos to column array
            row.push(elem[r*N+n])
        }
        row
    }
    
    //insert column in matrix, changes the matrix (not copying)
    pub fn insert_column_(&mut self, elem: Vec<T>, col: usize) {
        let elem = elem.to_vec();
        let N = self.N as usize;
        let col = col as usize;
        for (c,e) in elem.iter().enumerate() {
            self.elements[c*N+col] = *e
        }
    }
    
    pub fn insert_row_(&mut self, elem: Vec<T>, r: usize,) {
        let elem = elem.to_vec();
        let N = self.N as usize;
        for (c,e) in elem.iter().enumerate() {
            self.elements[r*N+c] = *e
        }
    }     
    
    //implements the possibilty to add values as slices into rows/columns
    //should have some test to check that indexes like start stop is within reach
    pub fn insert_column_slice_
        (&mut self, elem: Vec<T>, col: usize, start: usize) {
        let elem = elem.to_vec();
        let N = self.N as usize;
        let stop = start+elem.len()-1;
        let mut ind = 0;
        for c in 0..self.elements.len() {
            if c >= start && c <= stop {
                self.elements[c*N+col] = elem[ind];
                ind += 1;
                if c == stop { break }
            }
        }
    }
    
    pub fn insert_row_slice_
        (&mut self, elem: Vec<T>, r: usize, start: usize) {
        let elem = elem.to_vec();
        let N = self.N as usize;
        let stop = start+elem.len()-1;
        let mut ind = 0;
        for c in 0..self.elements.len() {
            if c >= start && c <= stop {
                self.elements[r*N+c] = elem[ind];
                ind += 1;
                if c == stop { break }
            }
        }
    }

    pub fn transpose(&self) -> Self {
        let mut transpose = Vec::new();
        for x in 0..self.M {
            let mut col = self.get_column(x as usize);
            transpose.append(&mut col);            
        }
        Self::new(transpose, self.N, self.M)
    }

    // rotate 90degrees. Takes the transpose and reverses the row/columns,
    // depending on whether the rotation should be clockwise and counterclockwise.
    // p in the function name stands for plus as in posetive direction of rotation.
    // m is for minus and is the opposite
    pub fn rotate_90p(&self) -> Matrix<T> {
        let mut res = self.transpose();
        let tempM = res.clone();
        for x in 0..self.M {
            let col = tempM.get_column(x)
                .into_iter().rev().collect::<Vec<T>>();
            res.insert_column_(col, x);
        }
        res
    }
    
    pub fn rotate_90m(&self) -> Matrix<T> {
        let mut res = self.transpose();
        let tempM = res.clone();
        for x in 0..self.N {
            let row = tempM.get_row(x).into_iter().rev().collect::<Vec<T>>();
            res.insert_row_(row, x);
        }
        res
    }

    pub fn unique(&self) -> u32 {
        let el = self.elements.to_vec();
        let mut unique = Vec::new();
        for e in el.iter() {                                                                                              
            let exists = unique.iter().any(|x| x==e); 
            if !exists { unique.push(*e) } 
        }
        unique.len() as u32
    }

    pub fn unique_count(&self) -> Vec<u8> {
        let mut el = self.elements.to_vec();
        let mut hh = Vec::new();
        let mut cc = 1;
        el.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));
        for n in 0..el.len()-1 {
           if el[n] == el[n+1] { cc += 1 } 
           else {
               hh.push(cc);
               cc = 1;
           }
        }
        hh.push(cc);
        hh
    }

    pub fn mul(&self, other: Self) -> Self {
        assert!(self.N == other.M,
                "wrong dim: M first = {}, N sec = {}",
                self.M, self.N);
        let mut res = Vec::new();
        for n in 0..self.M {
            for m in 0..other.N {
                let row = self.get_row(n);
                let col = other.get_column(m);
                let c = Self::dot(row, col);
                res.push(c);
            }
        }
        Self::new(res, self.M, other.N)
    }

    pub fn dot(row: Vec<T>, col: Vec<T>) -> T {
        assert!(col.len() == row.len(),
                "not of same length: col = {}, row = {}",
                col.len(), row.len());
        let mut res = Vec::new();
        for i in 0..row.len() {
            let c = row[i]*col[i];
            res.push(c);
        }
        res.into_iter().sum()
    }
 }
