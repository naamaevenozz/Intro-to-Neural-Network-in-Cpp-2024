#include "Activation.h"
#include <cmath>
#include "Matrix.h"
namespace activation {
    typedef Matrix (*ActivationFunction)(const Matrix&);

    Matrix relu(const Matrix& x)  {
        Matrix result(x.get_rows(),x.get_cols());
        for (int i = 0; i < x.get_rows(); ++i) {
            for (int j = 0; j < x.get_cols(); ++j) {
                if(x(i,j)<0)
                {
                    result(i,j)=0;
                }
                else
                {
                    result(i,j)=x(i,j);
                }
            }
        }
        return result;
    }
    Matrix softmax(const Matrix& x) {
        float sum=0.0;
        Matrix result(x.get_rows(),x.get_cols());
        for (int i = 0; i < x.get_rows(); ++i) {
            for (int j = 0; j < x.get_cols(); ++j) {
                result(i,j)=std::exp(x(i,j));
                sum+=result(i,j);
            }
        }
        for (int i = 0; i < x.get_rows(); ++i) {
            for (int j = 0; j < x.get_cols(); ++j) {
                result(i,j)/=sum;
            }
        }
        return result;
    }
};