extern crate grad;

use std::time::SystemTime;

use grad::{Tensor, mse};

fn main() {
    //grad::it_works();
    let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], (4, 1));

    let mut nn = vec![Tensor::new_rand((2,120)), Tensor::new_rand((120,120)), Tensor::new_rand((120,4)), Tensor::new_rand((4,1))];

    let epochs = 50_000;
    let mut res = y.clone();

    let lr = Tensor::new(vec![5_f32], (1,1));

    println!("{}", nn[0]);
    println!("{}", nn[1]);


    let time = SystemTime::now();
    for u in 0..epochs {
        let mut x1 = Tensor::new(vec![0.0, 0.0,
                                      0.0, 1.0,
                                      1.0, 0.0,
                                      1.0, 1.0], (4, 2));

        // do the forward
        x1.activate_grad();
        for n in nn.iter() {
            x1 = (&x1 * n).sigmoid();
        }
        
        // compute loss
        let mut t = mse(&x1, &y);

        res = x1.clone();

        // do the backward pass
        t.backward();
        let len = nn.len();
        for i in 1..nn.len()-1 {
            nn[len-i].deactivate_grad();

            // calculate gradient of weight
            let grad = nn[len-i].get_grad();

            // update weight
            nn[len-i] = &nn[len-i] - &(&lr * &grad);
        }
       
        if u % 1000 == 0 {
            print!("loss: {}\n", t); 
        }
    }

    println!("\n{}", res);
    println!("\ntime used: {:?}", time.elapsed().unwrap());
}

