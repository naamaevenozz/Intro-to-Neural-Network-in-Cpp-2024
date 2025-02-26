// Dense.h
#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"

class Dense {
private:
    Matrix _weights;
    Matrix _bias;
    activation::ActivationFunction _activation;

public:
    Dense(const Matrix& weights, const Matrix& bias,
          activation::ActivationFunction activation);

    const Matrix& get_weights() const;

    const Matrix& get_bias() const;

    activation::ActivationFunction get_activation() const;

    Matrix operator()(const Matrix& input) const;
};

#endif // DENSE_H









