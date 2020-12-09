extern crate grad;

use grad::Matrix;

fn main() {
    let X = Matrix::new(vec![0.0, 0.0,
                             0.0, 1.0,
                             1.0, 0.0,
                             1.0, 1.0], 4, 2);

    let y = Matrix::new(vec![0.0, 1.0, 1.0, 0.0], 4, 1);

    let mut nn = vec![Matrix::new_rand(2,4), Matrix::new_rand(4,4), Matrix::new_rand(4,1)];

    // FORWARD
    let epochs = 10000;
    let mut res = y.clone();

    for u in 0..epochs {
        let mut x1 = X.clone().activate_grad();

        for n in nn.iter() {
            x1 = (&x1 * n).sigmoid();
        }
        
        let t = x1.loss(&y);

        res = x1.clone();

        t.backward();

        for i in 0..nn.len() {
            nn[i].deactivate_grad();
            let grad = nn[i].grad();
            
            nn[i] = &nn[i] - &(1 as f32 * &grad);
        }
    }

    println!("{}\n", res);

    //let m2 = Matrix::new_rand(100,100).activate_grad();
    
    //grad::it_works();

    /*
    let z = (&m1 * &m2).sum();
    
    z.backward();

    println!("{}\n", m1);


    println!("{}\n", 0.01_f32 * &m1.grad());

    let step = 0.01_f32 * &m1.grad().deactivate_grad();
    m1 = &m1.deactivate_grad() - &step;

    println!("{:?}", m1);
    */
}
