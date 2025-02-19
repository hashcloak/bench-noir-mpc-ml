
use noir_mpc_ml::ml::train_multi_class;
use noir_mpc_ml::quantized::Quantized;

fn main(inputs: [[Quantized; 4]; 50], labels: [[Quantized; 50]; 3]) -> pub [([Quantized; 4], Quantized); 3] {
    let epochs = 1;
    let learning_rate_ratio = Quantized::new(131);

    let parameters = train_multi_class(epochs, inputs, labels, learning_rate_ratio);
    parameters
}
