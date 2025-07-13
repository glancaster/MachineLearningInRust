use ndarray::{Array1, Zip};
use rand::prelude::*;
use plotters::prelude::*;

fn main() {
    println!("Hello, world!");

    // Simple Dataset 
    let x = Array1::from(vec![1., 2., 3., 4., 5.]);
    let y = Array1::from(vec![5., 7., 9., 11., 13.]); //y = 2x + 3

    // Large Random Dataset
    let mut x_1 = Vec::new();
    let mut y_1 = Vec::new();
    let mut rng = rand::rng();

    for i in 0..100 {
        x_1.push(i as f64);
        let y_rand:f64 = ((2*i) + 3) as f64 + rng.random_range(-1.0..1.0);
        y_1.push(y_rand);
    }

    let x_v = Array1::from(x_1);
    let y_v = Array1::from(y_1);
    
    //println!("{x:?}");
    //println!("{y:?}");
    
    //let w = 0.0;
    //let b = 0.0;
    //let pred = predict(&x, w, b);
    //
    //println!("{pred:?}");
    //
    //let mse = mean_squared_error(&y, &pred);
    //
    //println!("{mse}");

    let (final_w, final_b, final_mse) = train(&x_v, &y_v, 1000, 0.0001);
    println!("Trained Model: y = {:.4}x + {:.4}", final_w, final_b);


    // Let's plot the original data, expected line, predicted line, and mse
    // using plotters to give static images after completion

    let root_drawing_area = BitMapBackend::new("images/0.1.png", (1024, 768))
        .into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .build_cartesian_2d(-2.0..x_v.len() as f64, -2.0..(2.0*y_v.len() as f64 +3.0) + 10.0)
        .unwrap();

    let org_data = Zip::from(&x_v).and(&y_v).map_collect(|&x,&y| (x,y));

    // Dataset
    chart.draw_series(
        org_data.iter().map(|point| Circle::new(*point, 2, &BLUE)),
    )
    .unwrap();

    // Expected Line
    chart.draw_series(LineSeries::new(
        (0..x_v.len()).map(|x| (x as f64, 2.0*x as f64 + 3.0)),
        &RED
    )).unwrap();

    // Predicted Line
    chart.draw_series(LineSeries::new(
        (0..x_v.len()).map(|x| (x as f64, 2.0*x as f64 + 3.0)),
        &BLUE
    )).unwrap();


    let mse_drawing_area = BitMapBackend::new("images/0.2.png", (1024, 768))
        .into_drawing_area();

    mse_drawing_area.fill(&WHITE).unwrap();
    
    let max_loss = final_mse.iter().map(|&(_,m)| m).fold(f64::NEG_INFINITY, |prev,curr| prev.max(curr));
    let mut chart = ChartBuilder::on(&mse_drawing_area)
        .build_cartesian_2d(-2.0..final_mse.len() as f64, -2.0..max_loss + 10.0)
        .unwrap();

    // MSE Line
    chart.draw_series(LineSeries::new(
        final_mse,
        &BLUE
    )).unwrap();

}

fn predict(x: &Array1<f64>, w: f64, b: f64) -> Array1<f64> {
    x.mapv(|x_i| w * x_i + b)
}

fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|e| e.powi(2)).mean().unwrap()
}

fn train(x: &Array1<f64>, y: &Array1<f64>, epochs: usize, lr: f64) -> (f64, f64, Vec<(f64,f64)> ) {
    let mut w = 0.0;
    let mut b = 0.0;
    let mut mse = Vec::new();

    for epoch in 0..epochs {
        let preds = predict(x, w, b);

        let error = &preds - y;

        let dw = 2.0 * Zip::from(x).and(&error).fold(0.0, |acc, &x_i, &e_i| acc + x_i * e_i) / x.len() as f64;
        let db = 2.0 * error.sum() / x.len() as f64;

        w -= lr * dw;
        b -= lr * db;
        let loss = mean_squared_error(y, &preds);
        mse.push((epoch as f64, loss));

        if epoch % 100 == 0 {
            println!("Epoch {epoch}: Loss = {:.4}, w = {:.4}, b = {:.4}", loss, w, b);
        }
    }

    (w, b, mse)
}
