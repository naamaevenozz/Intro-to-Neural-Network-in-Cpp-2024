#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

namespace activation {

    typedef Matrix (*ActivationFunction)(const Matrix&);

    Matrix relu(const Matrix& m);
    Matrix softmax(const Matrix& m);

}

#endif // ACTIVATION_H