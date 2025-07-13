use std::fs::File;
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use ndarray::{Array2, s};

fn main() {
    // multi-variate linear regression on MEDV
    // this doesn't filter out or check for any correlations
    let (x,y) = load_dataset("data/bostonhousing.csv");
    let (final_weights, final_bias) = train(&x, &y, 100000, 0.0000001);
}

fn load_dataset(path: &str) -> (Array2<f64>, Array2<f64>) {
    let file = File::open(path).expect("File not Found");
    let mut reader = ReaderBuilder::new()
        .from_reader(file);

    let array: Array2<f64> = reader.deserialize_array2_dynamic().expect("Failure to read array");
    //println!("{array:?}");
    
    let (x,y) = (
        array.slice(s![.., ..-1]).to_owned(),
        array.slice(s![.., -1..]).to_owned(),
    );
    (x,y)

}

fn predict(x : &Array2<f64>, weights : &Array2<f64>, bias: f64) -> Array2<f64> {
    x.dot(weights) + bias
}

fn train(x: &Array2<f64>, y: &Array2<f64>, epochs: usize, lr: f64) -> (Array2<f64>, f64) {
    let (n_samples, n_features) = x.dim();
    let mut weights = Array2::<f64>::zeros((n_features, 1));

    let mut bias = 0.0;

    for epoch in 0..epochs {
        let preds = predict(x, &weights, bias);
        let error = &preds - y;

        let dw = x.t().dot(&error) * (2.0 / n_samples as f64);
        let db = 2.0 * error.mean().unwrap();

        weights = &weights - &(dw * lr);
        bias -= lr * db;

        if epoch % 100 == 0 {
            let loss = mean_squared_error(y, &preds);
            println!("Epoch {epoch}: Loss = {:.4}", loss);
        }
    }

    (weights, bias)
}

fn mean_squared_error(y: &Array2<f64>, preds: &Array2<f64>) -> f64 {
    let diff = y - preds;
     diff.mapv(|e| e.powi(2)).mean().unwrap()
}
