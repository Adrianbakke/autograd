
pub type Dim = (usize, usize);

#[derive(Clone,Debug)]
pub struct Matrix {
    pub elem: Vec<f32>,
    pub dim:  (usize, usize),
}

impl Matrix {
    pub fn new(elem: Vec<f32>, dim: Dim) -> Self {
        Self { elem, dim }
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

        if M == 1 && N == 1 {
            lhs = Matrix::new(vec![lhs.elem[0]; otherM*otherN], rhs.dim);
        }

        if otherM == 1 && otherN == 1 {
            rhs = Matrix::new(vec![rhs.elem[0]; M*N], lhs.dim);
        }

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

        Self::new(res, self.dim)
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

        self
    }

    pub fn transpose_inplace(&mut self) {
        let mut vec = Vec::new();
        for i in 0..self.dim.1 {
            vec.extend(self.get_col(i))
        }
        self.elem = vec;
        self.dim = (self.dim.1, self.dim.0);
    }

    /// elementwise multiplication
    pub fn hadamard(&self, other: &Self) -> Self {
        /*
        assert!(self.dim.0 == other.dim.1 && self.dim.0 == other.dim.1,
             "wrong dim must be equal}");
        */        let (M, N) = self.dim;
        let (otherM, otherN) = other.dim;
        let (M, N) = self.dim;
        let (otherM, otherN) = other.dim;

        let mut lhs = self.clone();
        let mut rhs = other.clone();

        if M == 1 && N == 1 {
            lhs = Matrix::new(vec![lhs.elem[0]; otherM*otherN], rhs.dim);
        }

        if otherM == 1 && otherN == 1 {
            rhs = Matrix::new(vec![rhs.elem[0]; M*N], lhs.dim);
        }
            
        let mut res = Vec::new();
        for m in 0..lhs.dim.0 {
            let row1 = lhs.get_row(m);
            let row2 = rhs.get_row(m);
            for i in 0..row1.len() {
                res.push(&row1[i] * &row2[i]);
            }        let (M, N) = self.dim;
        let (otherM, otherN) = other.dim;
        }

        Self::new(res, lhs.dim)
    }

     pub fn sum(self) -> f32 {
        self.elem.iter().sum()
    }
/*

    pub fn grad(&self) -> Self {                                                
        // run backward before this                                             
        let mut grads = Vec::new();                                             
        for e in self.elem.iter() {                                             
            grads.push(e.get().grad().unwrap());                                
        }                                                                       
                                                                              
        Self::new(grads, self.M, self.N) 
    }

    pub fn sigmoid(&self) -> Self {
        let rg = self.require_grad;

        let mut res = Vec::new();

        for e in self.elem.iter() {
            res.push(sigmoid(e));
        }

        res        let (M, N) = self.dim;
        let (otherM, otherN) = other.dim;
    }

     pub fn sum(mut self) -> Container {
        self.elem.iter().sum()
    }

    // write a loss function
    pub fn loss(&self, other: &Self) -> Container {
        let x1 = &self.elem;
        let x2 = &other.elem;
        let mut res = (&(&x1[0] - &x2[0]) * &(&x1[0] - &x2[0]));
        for i in 1..x1.len() {
            res = &res + &(&(&x2[i] - &x1[i]) * &(&x2[i] - &x1[i]))
        }

        res
    }
    */
}
