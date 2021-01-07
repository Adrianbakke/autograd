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
pub struct Child {
    weights:   Matrix,
    grad:      Tensor
    order:     Order,
    transpose: bool,
}

#[derive(Debug, Clone)]
pub struct Container {
    matrix:        Matrix,
    grad_values:   Option<Matrix>,
    requires_grad: bool,
    children:      Vec<Child>,
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<Container>>);

impl Child {
    pub fn new(weights: Matrix, grad: Tensor, order: Order) -> Self {
        Self { weights, order, transpose: false }
    }

    pub fn set_transpose(&mut self, b: bool) {
        self.transpose = b;
    }

    pub fn create(vec: Vec<&Tensor>, grad: Vec<Matrix>, order: Vec<Order>) -> Vec<Self> {
        let mut res = vec![];
        for i in 0..vec.len() {
            let v_t = vec[i].matrix.is_transpose;

            let mut child = Self::new(grad[i], order[i]);

            if v_t { child.set_transpose(true) }

            res.push(child);
        }

        res
    }
 }

impl Tensor {
    pub fn new(elem: Vec<f32>, dim: Dim) -> Self {
        let matrix = Matrix::new(elem, dim);
        let container = Contianer { matrix, grad_values: None, requires_grad: false, children: vec![] };

        Self(Rc::new(RefCell::new(container)))
    }

    /*
    pub fn new_rand(dim: Dim) -> Self {
        let elem = rand_vec(dim.0 * dim.1, dim.1);

        let matrix = Matrix::new(elem, dim);
         
        let container = Container { matrix, grad_values: None, requires_grad: false, children: vec![] }
        
        Self(Rc::new(container)
    }


    pub fn from_mat(matrix: Matrix) -> Self {
        let container = Container { matrix, grad_values: None, requires_grad: false, children: vec![] }
        
        Self(Rc::new(container)
    }

    pub fn from_tensor(&self) -> Self {
        let matrix = self.matrix.clone();
        let children = Rc::clone(&self.children);

        Self { matrix, grad_values: None, requires_grad: false, children }
    }
    */

    pub fn push(&self, child: Child) {
        self.children.borrow_mut().push(child);
    }

    pub fn clone_matrix(&self) -> Matrix {
        self.matrix.clone()
    }

    pub fn get_dim(&self) -> Dim {
        self.matrix.dim
    }

    /*
    pub fn get_grad(&self) -> Self {
        self.traverse();
        Tensor::from_mat(self.grad_values.clone().unwrap())
    }

    pub fn calc_grad(&mut self) -> Option<Matrix> { 
        let mut grad = Matrix::new(vec![0_f32], (1,1));
        for child in self.children.iter() {
            let w = child.weights.clone();
            let g = child.grads.traverse().unwrap();
            let mut grad_tmp = match &child.order {
                Order::Right => &w.transpose() * &g,
                Order::Left  => &g * &w.transpose(),
                Order::Pass  => g.hadamard(&w),
            };
            if child.transpose { grad_tmp = grad_tmp.transpose() };
            grad = grad.add(&grad_tmp);
        }
        Some(grad)
    }

    pub fn traverse(&self) -> Option<Matrix> {
        match self.get().grad_values {
            Some(_) => self.get().grad_values.clone(),
            None    => self.calc_grad(),
        }
    }

    pub fn grad_n(&self) {
        let mut current = self.clone();
        let mut comps = vec![];

        let is_None = |opt| {
            match opt {
                Some(_) => false,
                None    => true,
            }
        };

        while is_None(current.get().grad_values.clone()) {
            comps.push(current.clone());
            current = current.clone().get().children[0].grads.clone();
        };

        let mut g = Matrix::new(vec![1_f32], (1,1));
        for comp in comps.iter().rev() {
            println!("{:?}", comp);
            let mut grad = Matrix::new(vec![0_f32], (1,1));
            for child in comp.get().children.iter() {
                let w = child.weights.clone();
                let grad_tmp = match &child.order {
                    Order::Right => &w.transpose() * &g,
                    Order::Left  => &g * &w.transpose(),
                    Order::Pass  => g.hadamard(&w),
                };
                if child.transpose { grad = grad.transpose() };
                grad = grad.add(&grad_tmp);
            }
            comp.get_mut().grad_values = Some(grad.clone());
            g = grad;
        }
    }

    pub fn grad_u(&self) -> Option<Matrix> {
        match self.get().grad_values {
            None => {
                let mut grad = Matrix::new(vec![0_f32], (1,1));
                for child in self.get().children.iter() {
                    let w = child.weights.clone();
                    let g = child.grads.clone().grad_u().unwrap();
                    let mut grad_tmp = match &child.order {
                        Order::Right => &w.transpose() * &g,
                        Order::Left  => &g * &w.transpose(),
                        Order::Pass  => g.hadamard(&w),
                    };
                    if child.transpose { grad_tmp = grad_tmp.transpose() };
                    grad = grad.add(&grad_tmp);
                }
                Some(grad)
            }
            _    => self.get().grad_values.clone()
        }
    }
    pub fn activate_grad(&mut self) {
        self.requires_grad = true;
    }

    pub fn deactivate_grad(&mut self) {
        self.requires_grad = false;
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mat = self.matrix.add(&rhs.matrix);
        Tensor::from_mat(mat)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mat = self.matrix.mul(&rhs.matrix);
        Tensor::from_mat(mat)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let mat = self.matrix.sub(&rhs.matrix);
        Tensor::from_mat(mat)
    }

    pub fn hadamard(&self, rhs: &Self) -> Self {
        let mat = self.matrix.hadamard(&rhs.matrix);
        Tensor::from_mat(mat)
    }

    pub fn pow(&self) -> Self {
        hadamard(self, self)
    }

    pub fn sum(&self) -> Self {
        let sum = self.matrix.sum();
        let mut z = Rc::new(Self::new(vec![sum], (1,1)));

        if self.requires_grad() {
            let (M, N) = self.get_dim();
            let grads = Matrix::new(vec![1_f32; M * N], (M, N));

            self.push(Child::new(grads, Rc::clone(&z), Order::Pass));

            z.activate_grad();
        }

        *z
    }
    pub fn sigmoid(&self) -> Self {
        let mut res = vec![];
        for e in self.matrix.elem.iter() {
            res.push((1.0 / (1.0 + (-e as f64).exp())) as f32)
        }

        let z = Self::new(res.clone(), self.matrix.dim);

        if self.requires_grad() {
            let mut grads_vec = vec![];
            for e in res.iter() {
                grads_vec.push(e * (1.0 - e))
            }

            let grads = Matrix::new(grads_vec, self.matrix.dim);
            let child = Child::new(grads, z.from_tensor(), Order::Pass);

            self.push(child);

            z.activate_grad();
        }

        z
    }

    pub fn backward(&self) {
        assert!(
            self.get().requires_grad, "activate requires_grad to run backward");
        self.get_mut().grad_values = Some(Matrix::new(vec![1.0], (1,1)));
    }

    pub fn transpose_inplace(&mut self) {
        self.matrix.transpose_inplace();
    }

    pub fn transpose(&self) -> Self {
        let mut res = Self::from_mat(self.matrix.transpose());
        res.container = Rc::clone(&self.container);

        res
    }
    */
}

/*
impl<'a> ops::Add<&'a Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Tensor {
        let z = self.add(rhs);

        if self.requires_grad() || rhs.requires_grad() {
            let (M,N) = self.get_dim();
            let grads = vec![&Matrix::new(vec![1_f32; M * N], (M, N)); 2]; 
            let childs = Child::create_child(vec![&self, &rhs], grads, &z);

            self.push(childs[0]);
            rhs.push(childs[1]);

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
        let z = self.sub(rhs);

        if self.requires_grad() || rhs.requires_grad() {
            let (M,N) = self.get_dim();
            let grad_lhs = Matrix::new(vec![1_f32; M * N], (M, N)); 
            let grad_rhs = Matrix::new(vec![-1_f32; M * N], (M, N)); 

            let childs = Child::create(
                vec![&self, &rhs], vec![&grad_lhs, &grad_rhs], &z);

            self.push(childs[0]);
            rhs.push(childs[1]);

            z.activate_grad();
        }

        z        
    }
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let z = lhs.mul(rhs);
    
    if lhs.requires_grad() || rhs.requires_grad(){
        let grads = vec![&lhs.matrix, &rhs.matrix];
        let childs = Child::create(vec![&lhs, &rhs], grads, &z);

        lhs.push(childs[0]);
        rhs.push(childs[1]);

        z.activate_grad();
    }

    z        
}


pub fn hadamard(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let z = lhs.hadamard(&rhs);

    if lhs.requires_grad() || rhs.requires_grad() {
        let grads = vec![&lhs.matrix, &rhs.matrix];
        let childs = Child::create(vec![&lhs, &rhs], grads, &z);

        lhs.push(childs[0]);
        rhs.push(childs[1]);

        z.activate_grad();
    }

    z        
}

pub fn mse(x: &Tensor, y: &Tensor) -> Tensor {
    let z = (x - y).pow();
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


pub fn rand_vec(n: usize, N: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut numbers: Vec<f32> = (0..n).map(|_| {
        rng.gen_range(0, 100) as f32
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

*/
pub fn it_works() {
    /*
    let mut a = Tensor::new(vec![0.2405, 0.0464, 0.3038, 0.4051, 0.4147, 0.2212, 0.1244, 0.2350], (2,4));
    let mut b = Tensor::new(vec![0.9857, 0.9792, 0.9828, 0.9859], (4,1));
    let mut x1 = Tensor::new(vec![0.0, 0.0,
                                  0.0, 1.0,
                                  1.0, 0.0,
                                  1.0, 1.0], (4, 2));
    let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], (4, 1));
    */
    let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], (4, 1));
    y.push(Child::new(Matrix::new(vec![1_f32], (1,1)), Order::Pass));
    let mut z = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], (4, 1));

    z.children = Rc::clone(&y.children);
    z.push(Child::new(Matrix::new(vec![2_f32], (1,1)), Order::Pass));
    println!("{:?}", y);
    //a.activate_grad();
    //b.activate_grad();
    //let n = (&b * &a).activate_grad();

    //let u = &b.transpose() * &n;
    //let z = n.sum().activate_grad();

    
    
    //z.backward();
    //println!("{:?}", a);
    //println!("{:?}\n", a.get_grad());
    //println!("{:?}\n", b.get_grad());
    //println!("{:?}", b.grad());
}

