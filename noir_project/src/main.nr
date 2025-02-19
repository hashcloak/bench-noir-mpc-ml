
use noir_mpc_ml::ml::train_multi_class;
use noir_mpc_ml::quantized::Quantized;

fn main(inputs: [[Quantized; 13]; 30], labels: [[Quantized; 30]; 3]) -> pub [([Quantized; 13], Quantized); 3] {
    let epochs = 1;
    let learning_rate_ratio = Quantized::new(218);

    let parameters = train_multi_class(epochs, inputs, labels, learning_rate_ratio);
    parameters
}
