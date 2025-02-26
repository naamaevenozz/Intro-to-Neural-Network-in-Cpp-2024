#include "Matrix.h"
#include "Activation.h"
#include "Dense.h"

Dense::Dense(const Matrix& weights, const Matrix& bias,
              activation::ActivationFunction activation)
        : _weights(weights), _bias(bias), _activation(activation) {}
const Matrix& Dense::get_weights() const {
    return _weights;
}

const Matrix& Dense::get_bias() const {
    return _bias;
}

activation::ActivationFunction Dense::get_activation() const {
    return _activation;
}

Matrix Dense::operator()(const Matrix& input) const {
    Matrix result = _weights * input + _bias;
    return _activation(result);
}

