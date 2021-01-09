use std::ops;

pub type Dim = (usize, usize);

#[derive(Clone,Debug)]
pub struct Matrix {
    pub elem:         Vec<f32>,
    pub dim:          (usize, usize),
    pub is_transpose: bool,
}

impl Matrix {
    pub fn new(elem: Vec<f32>, dim: Dim) -> Self {
        Self { elem, dim, is_transpose: false }
    }

    pub fn get_col(&self, col_num: usize) -> Vec<f32> {
        let mut col = Vec::new();
        let (M, N) = self.dim;
        for n in 0..M {
            // push elements at column pos to column array
            col.push(self.elem[n * N + col_num])
        }
        col
    }

    pub fn get_row(&self, row_num: usize) -> Vec<f32> {
        let mut row = Vec::new();
        let N = self.dim.1;
        for n in 0..N {
            // push elements at column pos to column array
            row.push(self.elem[row_num * N + n])
        }
        row
    }

    //insert column in matrix, changes the matrix (not copying)
    pub fn insert_column_(&mut self, elem: Vec<f32>, col: usize) {
        //let elem = elem.to_vec();
        let N = self.dim.1 as usize;
        for (c,e) in elem.iter().enumerate() {
            self.elem[c * N + col] = *e
        }
    }
    
    pub fn insert_row_(&mut self, elem: Vec<f32>, r: usize,) {
        //let elem = elem.to_vec();
        let N = self.dim.1 as usize;
        for (c,e) in elem.iter().enumerate() {
            self.elem[r * N + c] = *e
        }
    }

    pub fn append_col_(&mut self, elem: Vec<f32>, col: usize) {
        //let elem = elem.to_vec();
        let N = self.dim.1 as usize;
        for i in 0..elem.len()-1 {
            self.elem.insert(i * N + col, elem[i]);
        }
        self.elem.push(elem[elem.len()-1]);
    }

    pub fn append_row_(&mut self, elem: Vec<f32>) {
        //let elem = elem.to_vec();
        for e in elem.iter() {
            self.elem.push(*e)
        }
    }

    pub fn dot(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
        assert!(vec1.len() == vec2.len(), "lengths must be equal");
        let mut res = 0_f32;
        for i in 0..vec1.len() {
            res = res + (vec1[i] * vec2[i]);
        }
        res
    }

    pub fn add(&self, other: &Self) -> Self {
        let (M, N) = self.dim;
        let (otherM, otherN) = other.dim;
        
        let mut lhs = self.clone();
        let mut rhs = other.clone();
        
        let mut res_dim = lhs.dim;

        if M == 1 && N == 1 {
            lhs = Matrix::new(vec![lhs.elem[0]; otherM*otherN], rhs.dim);
            res_dim = rhs.dim;
        }

        if otherM == 1 && otherN == 1 {
            rhs = Matrix::new(vec![rhs.elem[0]; M*N], lhs.dim);
            res_dim = lhs.dim;
        }

        let (M, N) = res_dim;

        //assert!(N == otherN && M == otherM,
        //     "wrong dim - must be equal}");
            
        let mut res = Vec::new();
        for m in 0..M {
            let row1 = lhs.get_row(m);
            let row2 = rhs.get_row(m);
            for i in 0..row1.len() {
                res.push(row1[i] + row2[i]);
            }
        }

        Self::new(res, res_dim)
    }

    pub fn sub(&self, other: &Self) -> Self {
        let (M, N) = self.dim;
        let (otherM, otherN) = other.dim;

        assert!(N == otherN && M == otherM,
             "wrong dim must be equal}");
            
        let mut res = Vec::new();
        for m in 0..M {
            let row1 = self.get_row(m);
            let row2 = other.get_row(m);
            for i in 0..row1.len() {
                res.push(row1[i] - row2[i]);
            }
        }

        Self::new(res, self.dim)
    }

    pub fn mul(&self, other: &Self) -> Self {
        /*
        assert!(self.dim.1 == other.dim.0,
                "wrong dim: N first = {}, M sec = {}",
                self.dim.1, other.dim.0);
        */
        
        let mut res = Vec::new();
        for m in 0..self.dim.0 {
            for n in 0..other.dim.1 {
                let mut row = self.get_row(m as usize);
                let mut col = other.get_col(n as usize);
                let c = Self::dot(&row, &col);
                res.push(c);
            }
        }

        /*
        assert!(res.len() == (self.dim.0 * other.dim.1) as usize,
                "len should be {}, but is {}",
                res.len(), (self.dim.0 * other.dim.1));
        */

        Self::new(res, (self.dim.0, other.dim.1))
    }

    pub fn transpose(mut self) -> Self {
        let mut vec = Vec::new();
        for i in 0..self.dim.1 {
            vec.extend(self.get_col(i))
        }
        self.elem = vec;
        self.dim = (self.dim.1, self.dim.0);
        self.is_transpose = true;

        self
    }

    pub fn eye(&mut self, dim: Dim) {
        let val = self.elem[0];
        let mut elem = vec![0_f32; dim.0 * dim.1];
        for i in 0..dim.0 {
            elem[dim.1 * i + i] = val;
        }
        self.elem = elem;
        self.dim = dim;
    }

    pub fn transpose_inplace(&mut self) {
        let mut vec = Vec::new();
        for i in 0..self.dim.1 {
            vec.extend(self.get_col(i))
        }
        self.elem = vec;
        self.dim = (self.dim.1, self.dim.0);
        self.is_transpose = true;
    }



    /// elementwise multiplication
    pub fn hadamard(&self, other: &Self) -> Self {
        /*
        assert!(self.dim.0 == other.dim.1 && self.dim.0 == other.dim.1,
             "wrong dim must be equal}");
        */
        
        let mut lhs = self.clone();
        let mut rhs = other.clone();

        let mut res_dim = lhs.dim;

        if lhs.dim == (1,1) {
            lhs = Self::new(vec![lhs.elem[0]; rhs.dim.0 * rhs.dim.1], rhs.dim);
            res_dim = rhs.dim;
        }
        if rhs.dim == (1,1) {
            rhs = Self::new(vec![rhs.elem[0]; lhs.dim.0 * lhs.dim.1], lhs.dim);
            res_dim = lhs.dim;
        }

        let mut res = Vec::new();
        for m in 0..lhs.dim.0 {
            let row1 = lhs.get_row(m);
            let row2 = rhs.get_row(m);
            for i in 0..row1.len() {
                res.push(&row1[i] * &row2[i]);
            }
        }

        Self::new(res, res_dim)
    }

    pub fn sum(self) -> f32 {
        self.elem.iter().sum()
    }

    pub fn form(&self, dim: Dim) -> Self {
        if self.dim == dim { return self.clone() }

        if self.dim == (1,1) {
            return Self::new(vec![self.elem[0]; dim.0*dim.1], dim)
        }

        let req = self.dim.1 == dim.0;

        if self.dim.0 == dim.1 && req { return self.clone().transpose() }

        // 1 down 0 left
        // 2x3 3x4 -> 2x4 | 3x2 -> 3x4 || 8x3 
        // 3x1 -> 3x3 and 3x3 -> 1x3
        let (tmp_dim, contract_n, extend_n, tmp) = if req {
            ((1, dim.0), self.dim.0, dim.1, self.get_row(0))
        } else {
            ((dim.1, 1), self.dim.1, dim.0, self.get_col(0))
        };

        let mut res = Self::new(tmp, tmp_dim);
        for i in 1..contract_n {          
            let elem = if req { self.get_row(i) } else { self.get_col(i) };
            let mat = Self::new(elem, tmp_dim);
            res = res.add(&mat);
        }
        let tmp = res.elem.clone();
        if tmp.len() > 1 {
            res.transpose_inplace();
        }
        for i in 1..extend_n {
            let elem = tmp.clone();
            if req {
                res.append_col_(elem, i)
            } else {
                res.append_row_(elem)
            }
        }
        res.dim = dim;
        res
    }
}

impl<'a> ops::Mul<&'a Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Matrix {
        if self.dim == (1, 1) || rhs.dim == (1, 1) {
            self.hadamard(rhs)
        } else {
            self.mul(rhs)
        }
    }
}

