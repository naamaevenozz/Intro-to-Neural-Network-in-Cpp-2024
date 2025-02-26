#include "Matrix.h"
#include "MlpNetwork.h"
#include "Dense.h"

MlpNetwork::MlpNetwork(const Matrix weights[MLP_SIZE],
                       const Matrix biases[MLP_SIZE])
        : _layers{
        Dense(weights[0],biases[0],activation::relu),
        Dense(weights[1],biases[1],activation::relu),
        Dense(weights[2],biases[2],activation::relu),
        Dense(weights[3],biases[3],activation::softmax)
}{
}


// Overloaded operator()
digit MlpNetwork::operator()(const Matrix &m) const {
    Matrix matrix = _layers[0](m);
    matrix = _layers[1](matrix);
    matrix = _layers[2](matrix);
    matrix = _layers[3](matrix);

    int index = matrix.argmax();
    float probability = matrix[index];

    digit di;
    di.value = index;
    di.probability = probability;
    return di;
}

