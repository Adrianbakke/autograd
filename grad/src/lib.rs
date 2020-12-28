use std::ops;
use std::cell::{Ref, RefMut, RefCell};
use std::rc::Rc;
use std::iter::Sum;
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
    children:     Vec<Child>,
    grad_values:  Option<Matrix>,
    require_grad: bool,
}

#[derive(Debug, Clone)]
pub struct Child {
    weights:   Matrix,
    grads:     Tensor,
    order:     Order,
    transpose: bool,
}

/*
#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<Container>>);
*/

#[derive(Debug, Clone)]
pub struct Tensor {
    matrix: Matrix,
    container: Rc<RefCell<Container>>,
}

impl Child {
    pub fn new(weights: Matrix, grads: Tensor, order: Order) -> Self {
        Self { weights, grads, order, transpose: false }
    }

    pub fn set_transpose(&mut self, b: bool) {
        self.transpose = b;
    }

    pub fn create_childs_mul(lhs: &Tensor, rhs: &Tensor, z: &Tensor) -> (Self, Self) {
        let mut grads_lhs = rhs.get_matrix();

        let mut grads_rhs = lhs.get_matrix();

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

    pub fn create_childs_add(lhs: &Tensor, rhs: &Tensor, z: &Tensor) -> (Self, Self) {
        let lhs_t = rhs.get_matrix().is_transpose;
        let rhs_t = lhs.get_matrix().is_transpose;

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

    pub fn create_childs_sub(lhs: &Tensor, rhs: &Tensor, z: &Tensor) -> (Self, Self) {
        let lhs_t = rhs.get_matrix().is_transpose;
        let rhs_t = lhs.get_matrix().is_transpose;

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

    pub fn create_childs_hadamard(lhs: &Tensor, rhs: &Tensor, z: &Tensor) -> (Self, Self) {
        let lhs_t = rhs.get_matrix().is_transpose;
        let rhs_t = lhs.get_matrix().is_transpose;

        let grads_lhs = rhs.get_matrix(); 
        let grads_rhs = lhs.get_matrix();

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
            children: vec![],
            grad_values: None,
            require_grad: false,
        };
        Self { matrix, container: Rc::new(RefCell::new(cont)) }
    }

    pub fn from_mat(matrix: Matrix) -> Self {
        let cont = Container {
            children: vec![],
            grad_values: None,
            require_grad: false,
        };
        Self { matrix, container: Rc::new(RefCell::new(cont)) }
    }

    pub fn from_tensor(&self) -> Self {
        let matrix = self.matrix.clone();
        let cont = Rc::clone(&self.container);
        Self { matrix, container: cont }
    }

    pub fn push(&self, child: Child) {
        self.container.borrow_mut().children.push(child);
    }

    pub fn get_matrix(&self) -> Matrix {
        self.matrix.clone()
    }

    pub fn get_dim(&self) -> Dim {
        self.matrix.dim
    }

    pub fn get(&self) -> Ref<Container> {
        self.container.borrow()
    }

    pub fn get_mut(&self) -> RefMut<Container> {
        self.container.borrow_mut()
    }

    pub fn grad(self) -> Option<Matrix> {
        match self.get().grad_values {
            None => {
                let mut grad = Matrix::new(vec![], (1,1));
                let mut t = 0;
                for child in self.get().children.iter() {
                    t += 1;
                    let mut w = child.weights.clone();
                    let mut g = child.grads.clone().grad().unwrap();
                    if t > 1 {
                        let is_transpose = w.is_transpose;
                        let mut grad_tmp = match &child.order {
                            Order::Right => &(w.transpose()) * &g,
                            Order::Left  => &g * &(w.transpose()),
                            Order::Pass  => &g * &w,
                        };
                        if child.transpose { grad_tmp = grad_tmp.transpose() };
                        grad = grad.add(&grad_tmp);
                    } else {
                        let is_transpose = w.is_transpose;
                        println!("w {:?}, g {:?}", w, g);
                        grad = match &child.order {
                            Order::Right => &w.transpose() * &g,
                            Order::Left  => &g * &w.transpose(),
                            Order::Pass  => &g * &w,
                        };
                        if child.transpose { grad = grad.transpose() };
                    }                    
                }
                Some(grad)
            }
            _    => self.get().grad_values.clone()
        }
    }

    pub fn activate_grad(self) -> Self {
        self.get_mut().require_grad = true;
        self
    }

    pub fn deactivate_grad(self) -> Self {
        self.get_mut().require_grad = false;
        self
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

    pub fn sum(&self) -> Self {
        let sum = self.get_matrix().sum();
        let z = Self::new(vec![sum], (1,1));

        let grads = Matrix::new(
            vec![1_f32; self.get_dim().0 * self.get_dim().1], self.get_dim()); 

        self.push(Child::new(grads, z.from_tensor(), Order::Pass));

        z
    }

    pub fn sigmoid(&self) -> Self {
        let mut res = vec![];
        for e in self.matrix.elem.iter() {
            res.push((1.0 / (1.0 + (-e as f64).exp())) as f32)
        }

        let z = Self::new(res.clone(), self.matrix.dim);

        let mut grads_vec = vec![];
        for e in res.iter() {
            grads_vec.push(e * (1.0 - e))
        }

        let grads = Matrix::new(grads_vec, self.matrix.dim);
        let child = Child::new(grads, z.from_tensor(), Order::Pass);

        self.push(child);

        z
    }

    pub fn backward(&self) {
        assert!(
            self.get().require_grad, "activate require_grad to run backward");
        self.get_mut().grad_values = Some(Matrix::new(vec![1.0], (1,1)));
    }

    pub fn transpose_inplace(&mut self) {
        self.matrix.transpose_inplace();
    }

    pub fn transpose(&self) -> Self {
        let mut res = Self::from_mat(self.get_matrix().transpose());
        res.container = Rc::clone(&self.container);

        res
    }
}

impl<'a> ops::Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Tensor {
        let z = self.add(rhs);

        let (lhs_child, rhs_child) = Child::create_childs_add(&self, &rhs, &z);

        self.push(lhs_child);
        rhs.push(rhs_child);

        z        
    }
}

impl<'a> ops::Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Tensor {
        if self.get_dim() == (1, 1) || rhs.get_dim() == (1, 1) || self.get_dim() == rhs.get_dim() {
            hadamard(self, rhs)
        } else {
            mul(self, rhs)
        }
    }
}

impl<'a> ops::Sub<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Tensor {
        let z = self.sub(rhs);

        let (lhs_child, rhs_child) = Child::create_childs_sub(&self, &rhs, &z);

        self.push(lhs_child);
        rhs.push(rhs_child);

        z        
    }
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let z = lhs.mul(rhs);
    
    let (lhs_child, rhs_child) = Child::create_childs_mul(&lhs, &rhs, &z);

    lhs.push(lhs_child);
    rhs.push(rhs_child);

    z        
}


pub fn hadamard(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let z = lhs.hadamard(&rhs);

    let (lhs_child, rhs_child) = Child::create_childs_hadamard(&lhs, &rhs, &z);

    lhs.push(lhs_child);
    rhs.push(rhs_child);

    z        
}

pub fn mse(x: &Tensor, y: &Tensor) -> Tensor {
    let sub = x - y;
    let z = hadamard(&sub, &sub);

    z.sum()
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = String::new();
        let matrix = self.matrix.clone();
        let l = matrix.elem.len();
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
                for n in 0..2 {
                    for x in 0..sl {
                        res.push_str(".")
                    }
                    res.push_str("\n");
                }
            }
        } 
        write!(f, "{}", res)
    }
}

/*
pub fn it_works() {
    let a = Tensor::new(vec![1.0,2.0,3.0, 1.0,2.0,3.0, 1.0,2.0,3.0], (3,3)).activate_grad();
    let b = Tensor::new(vec![1.0,2.0,3.0], (1,3)).activate_grad();
    let r = Tensor::new(vec![6.0,12.0,18.0], (1,3)).activate_grad();
    let c = Tensor::new(vec![1.0,2.0,3.0], (3,1)).activate_grad();

    let mut n = (&b * &a).activate_grad();
    //let n = (&b.transpose() * &k).activate_grad();
    
    //let mut n = (&c.transpose() * &k).activate_grad();
    
    let z = mse(&n.sigmoid(), &r).activate_grad();

    //let n = (&b * &a).activate_grad();

    //let u = &b.transpose() * &n;
    println!("z: {}", z);
    //let z = n.sum().activate_grad();

    
    
    z.backward();
    //println!("{:?}", a);
    println!("{:?}", a.grad());
    //println!("{:?}", b.grad());
}
/*

