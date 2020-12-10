extern crate grad;

use grad::Matrix;

fn main() {

    let y = Matrix::new(vec![0.0, 1.0, 1.0, 0.0], 4, 1);

    let mut nn = vec![Matrix::new_rand(2,16), Matrix::new_rand(16,4), Matrix::new_rand(4,1)];

    let epochs = 100000;
    let mut res = y.clone();

    let lr = 5 as f32;

    for u in 0..epochs {

        let mut x1 = Matrix::new(vec![0.0, 0.0,
                                      0.0, 1.0,
                                      1.0, 0.0,
                                      1.0, 1.0], 4, 2);

        // do the forward
        x1.activate_grad();
        for n in nn.iter() {
            x1 = (&x1 * n).sigmoid();
        }
        
        // compute loss
        let t = x1.loss(&y);

        res = x1.clone();

        // do the backward pass
        t.backward();
        for i in 0..nn.len() {
            nn[i].deactivate_grad();

            // calculate gradient of weight
            let grad = nn[i].grad();

            // update weight
            nn[i] = &nn[i] - &(lr * &grad);
        }
        if u % 100 as usize == 0 || u == epochs-1 {
            print!("\rprogress: {}%",
                ((u as f32/(epochs-1) as f32) * 100.0) as usize) 
        }
    }

    println!("\n{}", res);
}
