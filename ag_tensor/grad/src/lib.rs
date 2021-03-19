use std::ops;
use std::cell::{Ref, RefMut, RefCell};
use std::rc::Rc;
use std::fmt;

extern crate rand;

mod matrix; 

pub use crate::matrix::{Matrix, Dim};

use rand::prelude::*;

#[derive(Debug, Clone)]
pub enum Order {
    Left,
    Right,
    Pass,
}

#[derive(Debug, Clone)]
pub struct Container {
    matrix:        Matrix,
    children:      Vec<Child>,
    grad_values:   Option<Matrix>,
    requires_grad: bool,
}

#[derive(Debug, Clone)]
pub struct Child {
    weights:   Matrix,
    grads:     Tensor,
    order:     Order,
    transpose: bool,
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<Container>>);

impl Child {
    pub fn new(weights: Matrix, grads: Tensor, order: Order) -> Self {
        Self { weights, grads, order, transpose: false }
    }

    pub fn set_transpose(&mut self, b: bool) {
        self.transpose = b;
    }

    pub fn create_childs_mul
        (lhs: &Tensor, rhs: &Tensor, z: &Tensor) -> (Self, Self) {
        let grads_lhs = rhs.get_matrix();
        let grads_rhs = lhs.get_matrix();

        let lhs_t = grads_lhs.is_transpose;
        let rhs_t = grads_rhs.is_transpose;

        let mut lhs_child =
            Self::new(grads_lhs, z.from_tensor(), Order::Left);

        let mut rhs_child =
            Self::new(grads_rhs, z.from_tensor(), Order::Right);

        if lhs_t { rhs_child.set_transpose(true) }
        if rhs_t { lhs_child.set_transpose(true) }

        (lhs_child, rhs_child)
    }

    pub fn create_childs_add
        (lhs: &Tensor, rhs: &Tensor, z: &Tensor) -> (Self, Self) {
        let lhs_t = rhs.get().matrix.is_transpose;
        let rhs_t = lhs.get().matrix.is_transpose;

        let grads_lhs = Matrix::new(
            vec![1_f32; rhs.get_dim().0 * rhs.get_dim().1],
                 rhs.get_dim()); 

        let grads_rhs = Matrix::new(
            vec![1_f32; lhs.get_dim().0 * lhs.get_dim().1],
                 lhs.get_dim()); 

        let mut lhs_child =
            Self::new(grads_lhs, z.from_tensor(), Order::Pass);

        let mut rhs_child =
            Self::new(grads_rhs, z.from_tensor(), Order::Pass);

        if lhs_t { rhs_child.set_transpose(true) }
        if rhs_t { lhs_child.set_transpose(true) }

        (lhs_child, rhs_child)
    }

    pub fn create_childs_sub
        (lhs: &Tensor, rhs: &Tensor, z: &Tensor) -> (Self, Self) {
        let lhs_t = rhs.get().matrix.is_transpose;
        let rhs_t = lhs.get().matrix.is_transpose;

        let grads_lhs = Matrix::new(
            vec![1_f32; rhs.get_dim().0 * rhs.get_dim().1],
                 rhs.get_dim()); 

        let grads_rhs = Matrix::new(
            vec![-1_f32; lhs.get_dim().0 * lhs.get_dim().1],
                 lhs.get_dim()); 

        let mut lhs_child =
            Self::new(grads_lhs, z.from_tensor(), Order::Pass);

        let mut rhs_child =
            Self::new(grads_rhs, z.from_tensor(), Order::Pass);

        if lhs_t { rhs_child.set_transpose(true) }
        if rhs_t { lhs_child.set_transpose(true) }

        (lhs_child, rhs_child)
    }

    pub fn create_childs_hadamard
        (lhs: &Tensor, rhs: &Tensor, z: &Tensor)-> (Self, Self) {
        let grads_lhs = rhs.get_matrix(); 
        let grads_rhs = lhs.get_matrix();

        let lhs_t = grads_lhs.is_transpose;
        let rhs_t = grads_rhs.is_transpose;

        let mut lhs_child =
            Self::new(grads_lhs, z.from_tensor(), Order::Pass);

        let mut rhs_child =
            Self::new(grads_rhs, z.from_tensor(), Order::Pass);

        if lhs_t { rhs_child.set_transpose(true) }
        if rhs_t { lhs_child.set_transpose(true) }

        (lhs_child, rhs_child)
    }
}

impl Tensor {
    pub fn new(elem: Vec<f32>, dim: Dim) -> Self {
        let matrix = Matrix::new(elem, dim);

        let cont = Container {
            matrix,
            children: vec![],
            grad_values: None,
            requires_grad: false, 
        };

        let container = Rc::new(RefCell::new(cont));

        Self(container)
    }

    pub fn new_rand(dim: Dim) -> Self {
        let elem = rand_vec(dim.0 * dim.1, dim.1);

        let matrix = Matrix::new(elem, dim);

        let cont = Container {
            matrix,
            children: vec![],
            grad_values: None,
            requires_grad: false, 
        };

        let container = Rc::new(RefCell::new(cont));

        Self(container)
    }

    pub fn from_mat(matrix: Matrix) -> Self {
        let cont = Container {
            matrix,
            children: vec![],
            grad_values: None,
            requires_grad: false, 
        };

        let container = Rc::new(RefCell::new(cont));

        Self(container)
    }

    pub fn from_tensor(&self) -> Self {
        let container = Rc::clone(&self.0);

        Self(container)
    }

    pub fn push(&self, child: Child) {
        self.get_mut().children.push(child);
    }

    pub fn get_matrix(&self) -> Matrix {
        self.get().matrix.clone()
    }

    pub fn get_dim(&self) -> Dim {
        self.get().matrix.dim
    }

    pub fn get(&self) -> Ref<Container> {
        self.0.borrow()
    }

    pub fn get_mut(&self) -> RefMut<Container> {
        self.0.borrow_mut()
    }

    pub fn get_grad(&mut self) -> Self {
        Tensor::from_mat(self.grad().unwrap())
    }

    pub fn grad(&self) -> Option<Matrix> {
        let is_none = match self.get().grad_values {
            None => true,
            _    => false,
        };

        if is_none {
            let mut grad = Matrix::new(vec![0_f32], (1,1));
            for child in self.get().children.iter() {
                let w = child.weights.clone();
                let g = child.grads.grad().unwrap();
                let mut grad_tmp = match &child.order {
                    Order::Right => &w.transpose() * &g,
                    Order::Left  => &g * &w.transpose(),
                    Order::Pass  => g.hadamard(&w),
                };
                if child.transpose { grad_tmp.transpose_inplace() };
                grad = grad.add(&grad_tmp);
            }
            self.set_grad_values(grad.clone());
            Some(grad)
        } else {
            self.get().grad_values.clone()
        }
    }

    pub fn set_grad_values(&self, grad: Matrix) {
        self.get_mut().grad_values = Some(grad);
    }

    pub fn activate_grad(&mut self) {
        self.get_mut().requires_grad = true;
    }

    pub fn deactivate_grad(&mut self) {
        self.get_mut().requires_grad = false;
    }

    pub fn requires_grad(&self) -> bool {
        self.get().requires_grad
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mat = self.get_matrix().add(&rhs.get_matrix());
        Tensor::from_mat(mat)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mat = self.get_matrix().mul(&rhs.get_matrix());
        Tensor::from_mat(mat)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let mat = self.get_matrix().sub(&rhs.get_matrix());
        Tensor::from_mat(mat)
    }

    pub fn hadamard(&self, rhs: &Self) -> Self {
        let mat = self.get_matrix().hadamard(&rhs.get_matrix());
        Tensor::from_mat(mat)
    }

    pub fn pow(&self) -> Self {
        hadamard(self, self)
    }

    pub fn sum(&self) -> Self {
        let sum = self.get_matrix().sum();
        let mut z = Self::new(vec![sum], (1,1));

        if self.requires_grad() {
            let grads = Matrix::new(
                vec![1_f32; self.get_dim().0 * self.get_dim().1],
                self.get_dim());

            self.push(Child::new(grads, z.from_tensor(), Order::Pass));

            z.activate_grad();
        }

        z
    }

    pub fn sigmoid(&self) -> Self {
        let mut res = vec![];
        for e in self.get().matrix.elem.iter() {
            res.push((1.0 / (1.0 + (-e as f64).exp())) as f32)
        }

        let mut z = Self::new(res.clone(), self.get_dim());

        if self.requires_grad() {
            let mut grads_vec = vec![];
            for e in res.iter() {
                grads_vec.push(e * (1.0 - e))
            }

            let grads = Matrix::new(grads_vec, self.get_dim());
            let child = Child::new(grads, z.from_tensor(), Order::Pass);

            self.push(child);

            z.activate_grad();
        }

        z
    }

    pub fn backward(&mut self) {
        assert!(
            self.get().requires_grad,
            "activate requires_grad to run backward");
        self.get_mut().grad_values = Some(Matrix::new(vec![1.0], (1,1)));
    }

    pub fn transpose_inplace(&self) {
        self.get_mut().matrix.transpose_inplace();
    }

    pub fn transpose(self) -> Self {
        self.get_mut().matrix.transpose_inplace();
        self 
    }
}

impl<'a> ops::Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Tensor {
        let mut z = self.add(rhs);

        if self.requires_grad() || rhs.requires_grad() {
            let (lhs_child, rhs_child) =
                Child::create_childs_add(&self, &rhs, &z);

            self.push(lhs_child);
            rhs.push(rhs_child);

            z.activate_grad();
        }

        z        
    }
}

impl<'a> ops::Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Tensor {
        if self.get_dim() == (1, 1) || rhs.get_dim() == (1, 1) {
            hadamard(self, rhs)
        } else {
            mul(self, rhs)
        }
    }
}

impl<'a> ops::Sub<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Tensor {
        let mut z = self.sub(rhs);

        if self.requires_grad() || rhs.requires_grad() {
            let (lhs_child, rhs_child) =
                Child::create_childs_sub(&self, &rhs, &z);

            self.push(lhs_child);
            rhs.push(rhs_child);

            z.activate_grad();
        }

        z        
    }
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let mut z = lhs.mul(rhs);
    
    if lhs.requires_grad() || rhs.requires_grad(){
        let (lhs_child, rhs_child) = Child::create_childs_mul(&lhs, &rhs, &z);

        lhs.push(lhs_child);
        rhs.push(rhs_child);

        z.activate_grad();
    }

    z        
}

pub fn hadamard(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let mut z = lhs.hadamard(&rhs);

    if lhs.requires_grad() || rhs.requires_grad() {
        let (lhs_child, rhs_child) =
            Child::create_childs_hadamard(&lhs, &rhs, &z);

        lhs.push(lhs_child);
        rhs.push(rhs_child);

        z.activate_grad();
    }

    z        
}

pub fn mse(x: &Tensor, y: &Tensor) -> Tensor {
    let z = (x - y).pow();
    z.sum()
}

pub fn rand_vec(n: usize, N: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut numbers: Vec<f32> = (0..n).map(|_| {
        rng.gen_range(-100, 100) as f32
    }).collect();

    for i in 0..(numbers.len() / N) {
        let mut m = Vec::new();
        for t in 0..N {
            m.push(numbers[i * N + t]);
        }

        let s: f32 = m.iter().sum();

        for t in 0..N {
            numbers[i * N + t] = numbers[i * N + t] / (s+1_f32);
        }
        
    }

    numbers
}


impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = String::new();
        let matrix = self.get_matrix();
        for m in 0..matrix.dim.0 {
            if matrix.dim.0 < 7 || m < 3 || m > matrix.dim.0-3 {
                res.push_str("|");
                let row = matrix.get_row(m);
                for c in 0..matrix.dim.1 { 
                    if matrix.dim.1 > 6 && c > 2 && c < matrix.dim.1-2 {
                        if c == 3 { res.push_str(" .... ") }
                        continue
                    }
                    if row[c] < 0.0 {
                        res.push_str(
                            &format!(" {:.4} ", row[c]));
                    } else {
                        res.push_str(
                            &format!("  {:.4} ", row[c]));
                    }
                }
                res.push_str("|");
                if m < matrix.dim.0-1 { res.push_str("\n") }
            } else if m == 3 {
                let sl: usize;
                if matrix.dim.1 > 6 {
                    sl = 53; 
                } else {
                    sl = 9 * matrix.dim.1 + 2;
                }
                for _ in 0..2 {
                    for _ in 0..sl {
                        res.push_str(".")
                    }
                    res.push_str("\n");
                }
            }
        } 
        write!(f, "{}", res)
    }
}

pub fn test() {
    let mut a = Tensor::new(vec![0.2405, 0.0464, 0.3038, 0.4051, 0.4147,
        0.2212, 0.1244, 0.2350], (2,4));
    let mut b = Tensor::new(vec![0.9857, 0.9792, 0.9828, 0.9859], (4,1));
    let mut x1 = Tensor::new(vec![0.0, 0.0,
                                  0.0, 1.0,
                                  1.0, 0.0,
                                  1.0, 1.0], (4, 2));
    let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], (4, 1));
    let mut test = Tensor::new(vec![5.0, 3.0, 3.0, 9.0], (4, 1));
    a.activate_grad();
    b.activate_grad();
    test.activate_grad();


    let mut n = (&(&x1 * &a).sigmoid() * &b).sigmoid();

    let mut z = mse(&n, &y);

    
    z.backward();
    println!("{:?}\n", x1.get_grad());
}

