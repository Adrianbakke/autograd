use std::ops;
use std::cell::{Ref, RefMut, RefCell};
use std::rc::Rc;
use std::iter::Sum;
use std::fmt;

extern crate rand;

use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Var{
    pub value:        f32,
    pub children:     Vec<Child>,
    pub grad_value:   Option<f32>,
    pub require_grad: bool,
}

#[derive(Debug, Clone)]
pub struct Child {
    pub weight: f32,
    pub grad:   Container, 
}

#[derive(Debug, Clone)]
pub struct Container(Rc<RefCell<Var>>);

impl Container {
    pub fn new(value: f32) -> Self {
        let var = Var::new(value); 
        Self(Rc::new(RefCell::new(var)))
    }

    pub fn get(&self) -> Ref<Var> {
        self.0.borrow()
    }

    pub fn get_mut(&self) -> RefMut<Var> {
        self.0.borrow_mut()
    }

    pub fn backward(&self) {
        assert!(
            self.get().require_grad, "activate require_grad to run backward");
        self.get_mut().grad_value = Some(1.0);
    }
}

impl Var {
    pub fn new(value: f32) -> Self {
        Self {
            value,
            children: Vec::new(),
            grad_value: None,
            require_grad: false,
        }
    }

    pub fn grad(&self) -> Option<f32> {
        match self.grad_value {
            None => {
                let mut grad = 0.0;
                for child in self.children.iter() {
                    let w = child.weight;
                    let g = child.grad.get().grad();

                    grad += w * g.unwrap();
                }

                Some(grad as f32)
            }
            _    => self.grad_value
        }
    }
}

impl Child {
    pub fn new(weight: f32, grad: Container) -> Self {
        Self { weight, grad }
    }
}

impl<'a> ops::Add<&'a Container> for &'a Container {
    type Output = Container;

    fn add(self, rhs: Self) -> Container {
        let z = Container::new(self.get().value + rhs.get().value);

        if self.get().require_grad || rhs.get().require_grad {
            z.get_mut().require_grad = true;

            self.get_mut().children.push(
                Child::new(1.0, Container(Rc::clone(&z.0))));

            rhs.get_mut().children.push(
                Child::new(1.0, Container(Rc::clone(&z.0))));
        }

        z
    }
}

impl<'a> ops::Sub<&'a Container> for &'a Container {
    type Output = Container;

    fn sub(self, rhs: Self) -> Container {
        let z = Container::new(self.get().value - rhs.get().value);

        if self.get().require_grad || rhs.get().require_grad {
            z.get_mut().require_grad = true;

            self.get_mut().children.push(
                Child::new(1.0, Container(Rc::clone(&z.0))));

            rhs.get_mut().children.push(
                Child::new(-1.0, Container(Rc::clone(&z.0))));
        }

        z
    }
}

impl<'a> ops::Mul<&'a Container> for &'a Container {
    type Output = Container;

    fn mul(self, rhs: Self) -> Container {
        let z = Container::new(self.get().value * rhs.get().value);

        if self.get().require_grad || rhs.get().require_grad {
            z.get_mut().require_grad = true;

            self.get_mut().children.push(
                Child::new(rhs.get().value, Container(Rc::clone(&z.0))));

            rhs.get_mut().children.push(
                Child::new(self.get().value, Container(Rc::clone(&z.0))));
        }

        z
    }
}

impl<'a> Sum<&'a Container> for Container {
    fn sum<I: Iterator<Item=&'a Container>>(mut iter: I) -> Self {
        let first = iter.next().unwrap().clone();
        iter.fold(first, |x1, x2| &x1 + x2)
    }
}

#[derive(Clone,Debug)]
pub struct Matrix {
    elem: Vec<Container>,
    M: usize,
    N: usize,
    pub require_grad: bool,
}

impl Matrix {
    pub fn new(elem: Vec<f32>, M: usize, N: usize) -> Self {
        let elem = elem.iter().map(
            |x| Container::new(*x)).collect::<Vec<_>>();

        Self {
            elem,
            M,
            N,
            require_grad: false,
        }
    }

    pub fn new_rand(M: usize, N: usize) -> Self {
        let elem = rand_vec(M * N, N).iter().map(
            |x| Container::new(*x)).collect::<Vec<_>>();

        Self {
            elem,
            M,
            N,
            require_grad: false,
        }
    }

    /// create Matrix where all elements require grad
    pub fn new_wgrad(
        elem: Vec<f32>, M: usize, N: usize) -> Self {
        let elem = elem.iter().map( |x| {
            let c = Container::new(*x);
            c.get_mut().require_grad = true;
            c
        }).collect::<Vec<_>>();

        Self {
            elem,
            M,
            N,
            require_grad: true,
        }
    }

    pub fn from_containers(elem: Vec<Container>, M: usize, N: usize) -> Self {
        Self {
            elem,
            M,
            N,
            require_grad: false,
        }
    }
    
    pub fn get_column(&self, col_num: usize) -> Vec<Container> {
        let elem = self.elem.to_vec();
        let mut column = Vec::new();
        let M = self.M as usize;
        let N = self.N as usize;

        for n in 0..M {
            // push elements at column pos to column array
            column.push(elem[n * N + col_num].clone())
        }

        column
    }

    pub fn get_row(&self, row_num: usize) -> Vec<Container> {
        let elem = self.elem.to_vec();
        let mut row = Vec::new();
        let N = self.N as usize;

        for n in 0..N {
            // push elements at column pos to column array
            row.push(elem[row_num * N + n].clone())
        }

        row
    }

    pub fn dot(vec1: &Vec<Container>, vec2: &Vec<Container>) -> Container {
        let mut iter1 = vec1.iter();
        let mut iter2 = vec2.iter();

        let mut res =
            &iter1.next().unwrap().clone() * &iter2.next().unwrap().clone();

        for (v1,v2) in iter1.zip(iter2) {
            res = &res + &(v1 * v2);
        }

        res
    }

    pub fn add(&self, other: &Self) -> Self {
        assert!(self.N == other.N && self.M == other.M,
             "wrong dim must be equal}");
            
        let rg = self.require_grad || other.require_grad;

        let mut res = Vec::new();
        for m in 0..self.M {
            let row1 = self.get_row(m);
            let row2 = other.get_row(m);
            for i in 0..row1.len() {
                row1[i].get_mut().require_grad = rg;
                row2[i].get_mut().require_grad = rg;

                res.push(&row1[i] + &row2[i]);
            }
        }
        let mut z = Self::from_containers(res, self.M, self.N);

        if rg { z.require_grad = true }

        z
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert!(self.N == other.N && self.M == other.M,
             "wrong dim must be equal}");
            
        let rg = self.require_grad || other.require_grad;

        let mut res = Vec::new();
        for m in 0..self.M {
            let row1 = self.get_row(m);
            let row2 = other.get_row(m);
            for i in 0..row1.len() {
                row1[i].get_mut().require_grad = rg;
                row2[i].get_mut().require_grad = rg;

                res.push(&row1[i] - &row2[i]);
            }
        }
        let mut z = Self::from_containers(res, self.M, self.N);

        if rg { z.require_grad = true }

        z
    }
 
    pub fn mul(&self, other: &Self) -> Self {
        assert!(self.N == other.M,
                "wrong dim: N first = {}, M sec = {}",
                self.N, other.M);
        
        let rg = self.require_grad || other.require_grad;

        let mut res = Vec::new();

        for m in 0..self.M {
            for n in 0..other.N {
                let row = self.get_row(m as usize);
                let col = other.get_column(n as usize);
                for i in 0..row.len() {
                    row[i].get_mut().require_grad = rg;
                    col[i].get_mut().require_grad = rg;
                }
                let c = Self::dot(&row, &col);
                res.push(c);
            }
        }

        assert!(res.len() == (self.M * other.N) as usize,
                "len should be {}, but is {}",
                res.len(), (self.M * other.N));

        let mut z = Self::from_containers(res, self.M, other.N);

        if rg { z.require_grad = true }

        z
    }

    pub fn transpose(&mut self) -> Self {
        let mut vec = Vec::new();
        for i in 0..self.N {
            vec.extend(self.get_column(i))
        }
        let mut res = self.clone();
        res.elem = vec;
        res.M = self.N;
        res.N = self.M;
        res
    }
    /// elementwise multiplication
    pub fn multiply(&self, other: &Self) -> Self {
        assert!(self.N == other.N && self.M == other.M,
             "wrong dim must be equal}");
            
        let rg = self.require_grad || other.require_grad;

        let mut res = Vec::new();
        for m in 0..self.M {
            let row1 = self.get_row(m);
            let row2 = other.get_row(m);
            for i in 0..row1.len() {
                row1[i].get_mut().require_grad = rg;
                row2[i].get_mut().require_grad = rg;
                res.push(&row1[i] * &row2[i]);
            }
        }
        let mut z = Self::from_containers(res, self.M, self.N);

        if rg { z.require_grad = true }

        z
    }

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
            e.get_mut().require_grad = rg;
            res.push(sigmoid(e));
        }

        let mut z = Self::from_containers(res, self.M, self.N); 
        z.require_grad = true;

        z
    }

    pub fn activate_grad(&mut self) {
        self.require_grad = true;
    }

    pub fn deactivate_grad(&mut self) {
        self.require_grad = false;
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
}

impl<'a> ops::Add<&'a Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Matrix {
        self.add(rhs)
    }
}

impl<'a> ops::Sub<&'a Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Matrix {
        self.sub(rhs)
    }
}

impl<'a> ops::Mul<&'a Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Matrix {
        self.mul(rhs)
    }
}

impl<'a> ops::Mul<&'a Matrix> for f32 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        let rg = rhs.require_grad;

        let mut res = Vec::new();

        let scalar = Container::new(self);

        for el in rhs.elem.iter() {
            el.get_mut().require_grad = rg;
            res.push(&scalar * &el)
        }

        let mut z = Matrix::from_containers(res, rhs.M, rhs.N);

        if rhs.require_grad { z.require_grad = true }

        z
    }
}

impl<'a> ops::Add<&'a Matrix> for f32 {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        let rg = rhs.require_grad;

        let mut res = Vec::new();

        let scalar = Container::new(self);

        for el in rhs.elem.iter() {
            el.get_mut().require_grad = rg;
            res.push(&scalar + &el)
        }

        let mut z = Matrix::from_containers(res, rhs.M, rhs.N);

        if rg { z.require_grad = true }

        z
    }
}

impl<'a> ops::Sub<&'a Matrix> for f32 {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
        let rg = rhs.require_grad;

        let mut res = Vec::new();

        let scalar = Container::new(self);
        for el in rhs.elem.iter() {
            el.get_mut().require_grad = rg;
            res.push(&scalar - &el)
        }

        let mut z = Matrix::from_containers(res, rhs.M, rhs.N);

        if rg { z.require_grad = true }

        z
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

pub fn sigmoid(c: &Container) -> Container {
    let x    = -1.0 * c.get().value as f64;
    let sigm = (1.0 / (1.0 + x.exp())) as f32;
    let z    = Container::new(sigm); 

    if c.get().require_grad {
        z.get_mut().require_grad = true;

        let grad = sigm * (1.0 - sigm);

        c.get_mut().children.push(
                Child::new(grad, Container(Rc::clone(&z.0))));
    }

    z
}

impl fmt::Display for Container {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.get().value)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut res = String::new();
        let l = self.elem.len();
        for m in 0..self.M {
            if self.M < 7 || m < 3 || m > self.M-3 {
                res.push_str("|");
                let row = Matrix::get_row(self, m as usize);
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

pub fn it_works() {
    
    //let mut v = rand_vec((s * s) as usize);
    //let mut u = rand_vec((s * s) as usize);
    //let mut l = rand_vec(s as usize);
    //let mut n = rand_vec(s as usize);


    let mut a = Matrix::new(vec![1.0,2.0,3.0, 1.0,2.0,3.0, 1.0,2.0,3.0], 3, 3);
    a.activate_grad();

    let mut b = Matrix::new(vec![1.0,2.0,3.0], 3, 1);
    b.activate_grad();

    let y = &a * &b;

    let n = (&b.transpose() * &y).sum();

    n.backward();

    println!("{}\n", a.grad());
    println!("{}", b.grad());

/*
    let loss = m3.loss(&y);
    loss.backward();


    println!("{}", &a1 - &(0.1 * &a1.grad()));


    //let m4 = (&m3 * &m2).sigmoid();


    //let t = loss.sum().backward();
    //println!("{}", m1.grad());

    //let m3 = m1.mul(&m2);
    //let m4 = Matrix::new(l, s, 1);
    //let m5 = (m1.mul(&m2)).sigmoid();

    let m6 = Matrix::new(vec![13.0,2.0], 2, 1);
    let m7 = Matrix::new(vec![2.0,1.0], 1, 2);
    let m8 = (m6.mul(&m7)).sigmoid();
    //let t: Container = m5.elem.iter().sum();

    //let x = (&m1*&m2).sigmoid();
     //println!("{:?}\n", &m1+&m2);
 
    //println!("{:?}", m1.grad());
    //println!("{:?}", m4.grad());
    // println!("{:?}", m3);
    //

    //println!("{:?}", sigmoid(&Container::new(2.0)));
    //
    //println!("{}", m2.grad());
*/
}

