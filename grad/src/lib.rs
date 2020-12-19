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
pub struct Container {
    matrix:       Matrix,
    children:     Vec<Child>,
    grad_values:        Option<Matrix>,
    require_grad: bool,
}

#[derive(Debug, Clone)]
pub struct Child {
    weights: Matrix,
    grads:   Tensor,
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<Container>>);

impl Child {
    pub fn new(weights: Matrix, grads: Tensor) -> Self {
        Self { weights, grads }
    }
}

impl Tensor {
    pub fn new(elem: Vec<f32>, dim: Dim) -> Self {
        let matrix = Matrix::new(elem, dim);
        let cont = Container {
            matrix,
            children: vec![],
            grad_values: None,
            require_grad: false,
        };
        Self(Rc::new(RefCell::new(cont)))
    }

    pub fn from_mat(matrix: Matrix) -> Self {
        let cont = Container {
            matrix,
            children: vec![],
            grad_values: None,
            require_grad: false,
        };
        Self(Rc::new(RefCell::new(cont)))
    }

    pub fn push(&self, child: Child) {
        self.0.borrow_mut().children.push(child);
    }

    pub fn get_matrix(&self) -> Matrix {
        self.0.borrow().matrix.clone()
    }

    pub fn get_dim(&self) -> Dim {
        self.0.borrow().matrix.dim
    }

    pub fn get(&self) -> Ref<Container> {
        self.0.borrow()
    }

    pub fn get_mut(&self) -> RefMut<Container> {
        self.0.borrow_mut()
    }

    pub fn grad(self) -> Option<Matrix> {
        match self.get().grad_values {
            None => {
                let mut grad = Matrix::new(
                    vec![0_f32; self.get_dim().0 * self.get_dim().1],
                    self.get_dim());
                for child in self.get().children.iter() {
                    let w = child.weights.clone();
                    let g = child.grads.clone().grad();
                    println!("test:");
                    println!("\n{:?}\n", w);
                    println!("\n{:?}\n", g);

                    grad = grad.add(&w.hadamard(&g.unwrap()));
                    println!("\n{:?}\n", grad);
                }

                Some(grad)
            }
            _    => self.get().grad_values.clone()
        }
    }

    /*

    pub fn sigmoid(&self) -> Self {
        let rg = self.require_grad;
        let mut res = Vec::new();
        for e in self.elem.iter() {
            res.push(sigmoid(e));
        }
        res
    }
    */

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
        let z = Tensor::new(vec![sum], (1,1));

        let grads = Matrix::new(
            vec![1_f32; self.get_dim().0 * self.get_dim().1], self.get_dim()); 

        self.push(Child::new(grads, Tensor(Rc::clone(&z.0))));

        z
    }

    pub fn backward(&self) {
        assert!(
            self.get().require_grad, "activate require_grad to run backward");
        self.get_mut().grad_values = Some(Matrix::new(vec![1.0], (1,1)));
    }
}

impl<'a> ops::Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Tensor {
        let z = self.add(rhs);

        let grads_lhs = Matrix::new(
            vec![1_f32; rhs.get_dim().0 * rhs.get_dim().1],
                 rhs.get_dim()); 

        let grads_rhs = Matrix::new(
            vec![1_f32; self.get_dim().0 * self.get_dim().1],
                 self.get_dim()); 

        self.push(
            Child::new(grads_rhs, Tensor(Rc::clone(&z.0))));

        rhs.push(
            Child::new(grads_lhs, Tensor(Rc::clone(&z.0))));

        z        
    }
}

impl<'a> ops::Mul<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Tensor {
        let z = self.mul(rhs);

        let grads_lhs = rhs.get_matrix().clone().transpose(); 
        let grads_rhs = self.get_matrix().clone().transpose();

        self.push(
            Child::new(grads_rhs, Tensor(Rc::clone(&z.0))));

        rhs.push(
            Child::new(grads_lhs, Tensor(Rc::clone(&z.0))));

        z        
    }
}

impl<'a> ops::Sub<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Tensor {
        let z = self.sub(rhs);

        let grads_lhs = Matrix::new(
            vec![1_f32; rhs.get_dim().0 * rhs.get_dim().1],
                 rhs.get_dim()); 

        let grads_rhs = Matrix::new(
            vec![1_f32; self.get_dim().0 * self.get_dim().1],
                 self.get_dim()); 

        self.push(
            Child::new(grads_rhs, Tensor(Rc::clone(&z.0))));

        rhs.push(
            Child::new(grads_lhs, Tensor(Rc::clone(&z.0))));

        z        
    }
}

pub fn hadamard(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let z = lhs.hadamard(&rhs);

    let grads_lhs = rhs.get_matrix(); 
    let grads_rhs = lhs.get_matrix();

    lhs.push(
        Child::new(grads_rhs, Tensor(Rc::clone(&z.0))));

    rhs.push(
        Child::new(grads_lhs, Tensor(Rc::clone(&z.0))));

    z        
}

pub fn mse(x: &Tensor, y: &Tensor) -> Tensor {
    let sub = x - y;
    let z = hadamard(&sub, &sub);

    z.sum()
}
        
/*
pub fn sum(tensor: &mut Tensor) -> Container { 
    let mut res = 0_f32;
    for i in 0..tensor.matrix.elem.len() {
        res += tensor.matrix.elem[i]
    }
    
    let ret = Tensor::new(vec![res], (1,1));

    let parent = Rc::new(Tensor::new(vec![res], (1,1))
    let grads = Matrix::new(
        vec![1_f32; tensor.matrix.dim.0 * tensor.matrix.dim.1],
        tensor.matrix.dim);

    tensor.children.push(Child::new(grads, Container(Rc::clone(&z.0))));

    z 
}
*/

/*
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = String::new();
        let l = self.elem.len();
        for m in 0..self.M {
            if self.M < 7 || m < 3 || m > self.M-3 {
                res.push_str("|");
                let row = Tensor::get_row(self, m as usize);
                for c in 0..self.N { 
                    if self.N > 6 && c > 2 && c < self.N-2 {
                        if c == 3 { res.push_str(" .... ") }
                        continue
                    }
                    if row[c].get().value < 0.0 {
                        res.push_str(
                            &format!(" {:.4} ", row[c].get().value));
                    } else {
                        res.push_str(
                            &format!("  {:.4} ", row[c].get().value));
                    }
                }
                res.push_str("|");
                if m < self.M-1 { res.push_str("\n") }
            } else if m == 3 {
                let sl: usize;
                if self.N > 6 {
                    sl = 53; 
                } else {
                    sl = 9 * self.N + 2;
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
*/

pub fn it_works() {

    let m = Tensor::new(vec![1.0,2.0,3.0], (3,1)).activate_grad();
    let n = Tensor::new(vec![1.0,2.0,3.0], (1,3)).activate_grad();
    let l = Tensor::new(vec![1.0], (1,1)).activate_grad();


    //let z = mse(&(&n * &m), &l).activate_grad();
    let z = (&m * &n).sum().activate_grad();

    z.backward();

    println!("{:?}", n);

    println!("{:?}", n.grad());
}

