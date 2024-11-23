#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iomanip> // For std::setprecision

using namespace std;

//struct Constants
//{
//    double w = 2 * M_PI;
//    double w_0 = 3.0 / 2.0 * w;
//    double gamma = 1.5;
//    double f = gamma * w_0 * w_0;
//    double y0 = 0.0, y1 = 0.0;
//    double beta = 3.0 / 4.0 * M_PI;
//    // C grid 
//    int C_elem = 1000;
//    double C_fin = 10.0;
//    double C_in = 1.0;
//    // K grid 
//    int K_elem = 1000;
//    double K_fin = 100.0;
//    double K_in = 40.0;
//    // t grid 
//    double t_max = 10.0;
//    double dt = 0.01;
//    int grid_size = static_cast<int>(t_max / dt) + 1; // static_cast should do the conversion from float to int in this case
//}; 

//end


constexpr size_t MATRIX_COLUMNS = 2;
constexpr float GAMMA = 1.5f;
constexpr float W = 2 * M_PI;
constexpr float DT = 0.001f;

__device__ __constant__  float d_H[4];  

__device__ void JM(float* A, const double &x1,const double &C,const  double &K) {
    A[0] = 0;
    A[1] = -K * cosf(x1);
    A[2] = -1;
    A[3] = -C;
}

__device__ void inverse(float* M) {
    M[0] =  M[3]/(M[0]*M[3] - M[2]*M[1]);
    M[1] = -M[1]/(M[0]*M[3] - M[2]*M[1]);
    M[2] = -M[2]/(M[0]*M[3] - M[2]*M[1]);
    M[3] =  M[0]/(M[0]*M[3] - M[2]*M[1]);
}


//__device__ inline float& idx(float A[], size_t i, size_t k) {
//    int col = (int)A[1];
//    return A[i + col * k];
//}

__device__ inline float& mat(float M[], size_t i, size_t j) {
    return M[i + MATRIX_COLUMNS * j];
}

// matrix are [n,m ,...column 1...,.....column 2....,...]
__device__ void myqr(float* A,float* Q, float* R){

    int n = 2;
    int m = 2;
    
    for (int i = 0; i < n * m; ++i) {
        R[i] = 0.0f;
        Q[i] = 0.0f;
    }

    for(int k = 0; k < m ; k++){
        for(int j = 0; j < k-1; j++){
            
            double sum = 0;
            for(int i = 0; i < n; i++){
                sum =+ mat(Q, i, j)*mat(A, i, k);
            }
            mat(R, j, k) = sum;
            for (int i = 0; i < n; ++i) {
               mat(A, i, k) = mat(A, i, k) - mat(R, j, k) * mat(Q, i, j);
            }
        }
        float norm_squared = 0.0f;
        for (int i = 0; i < n; ++i) {
            norm_squared +=  mat(A, i, k)* mat(A, i, k);
        }
        mat(R, k, k) = sqrtf(norm_squared);
        // Update Q
        for (int i = 0; i < n; ++i) {
           mat(Q, i, k) = mat(A, i, k)/ mat(R, k, k);
        }
        
    }
    int nn = 2;
    for (int ii = 0; ii < nn; ii++) {
        if (R[ii + n*ii] < 0) {
            Q[ii + n*ii] *= -1;
            R[ii + n*ii] *= -1;
        }
    }
};

__device__ void PendulumODE(float &f, float &x, float &dxdt, float &t, float &K, float &C){
    f = GAMMA * K * cosf(W * t) - K * sinf(x) - C * dxdt;
}


__global__ void SolveODEs(float* x, float* dxdt, float* K, float* C, int max_iter, int count){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float t = 0;
    float A[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float Q[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float Qk[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float R[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_JM_plus[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float d_JM_minus[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float ll_1 = 0.0f;
    float tmp = 0.0f;
    float ll_curr = 0.0f;
    float f = 0.0f;
    if(id < count)
    {
    myqr(Qk, Q, R);
    for(int i = 1; i < max_iter; i++)
        {
            t = DT*i;
            // Fowar Euler for each array associated with the correct index x_{n+1} = x_{n} + h*f(x_{n}, t)
            PendulumODE(f, x[id], dxdt[id], t, K[id], C[id]);
            // fare un integratore numerico decente <----- ms 
            x[id] = x[id] +  DT*dxdt[id];
            dxdt[id] = dxdt[id] +  DT*f;
            JM(A, x[id], C[id], K[id]);
            for(int j = 0; j < 2; j++){
                for(int k = 0;k < 2; k++){
                    mat(d_JM_plus, j, k)  = mat(d_H, j, k)  + 0.5 * DT * mat(A,j,k); 
                    mat(d_JM_minus, j, k) = mat(d_H, j, k)  - 0.5 * DT * mat(A,j,k);
                }
            }
            inverse(d_JM_minus);
            for(int j = 0; j < 2; j++){
                for(int k = 0;k < 2; k++){
                    A[j + 2*k]  = d_JM_plus[j + 2*k]* d_JM_minus[j + 2*k];
                    Q[j + 2*k]  = A[j + 2*k]*Q[j + 2*k];
                }
            }
            for (int i = 0; i < 4; ++i) {
                Qk[i] = Q[i];
            }
            myqr(Qk, Q, R);
            tmp = logf(R[0]);
            ll_curr += tmp;
            ll_1 = ll_curr/t;
        };
    }
    x[id] = ll_1;
}

int main()
{
    int count    = 1000000;
    int max_iter = 10000;

    float *h_x    = new float[count];
    float *h_dxdt = new float[count];
    
    // Initial conditions
    for(int i = 0; i < count; i++)
        {
            h_x[i]    = 0;
            h_dxdt[i] = 0;
        };
    
    float *d_x, *d_dxdt;
    // Allocate space for copies of the integers on the GPU
    if(cudaMalloc(&d_x, sizeof(float)*count) != cudaSuccess)
        {
            cout << "Error allocating memory!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            return 0;
    };
    
    if(cudaMalloc(&d_dxdt, sizeof(float)*count) != cudaSuccess)
        {
            cout << "Error allocating memory!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            cudaFree(d_x);
            return 0;
    };

    // Copy the integer's values from the CPU to the GPU
    if(cudaMemcpy(d_x, h_x, sizeof(float)*count, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cout << "Could not copy!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            return 0;
        }; 

    if(cudaMemcpy(d_dxdt, h_dxdt, sizeof(float)*count, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cout << "Could not copy!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            return 0;
        }; 
    
    // Hessian matrix
    float h_H[4];
    h_H[0] = 1;
    h_H[1] = 0;
    h_H[2] = 0;
    h_H[3] = -1;
    // Copy data from hostArray to constantArray
    cudaMemcpyToSymbol(d_H, h_H, sizeof(float)*4);

    int C_elem = 1000;
    double C_fin;
    double C_in;
    double K_fin;
    double K_in;

    cout << "Enter initial K: ";
    cin >> K_in;
    cout << "Enter final K: ";
    cin >> K_fin;
    cout << "Enter initial C: ";
    cin >> C_in;
    cout << "Enter final C: ";
    cin >> C_fin;
    
    double step_C = (C_fin - C_in) / (C_elem - 1);

    int K_elem = 1000;

    double step_K = (K_fin - K_in) / (K_elem - 1);

    int array_size = C_elem * K_elem;

    float *h_samples_C = new float[array_size];
    float *h_samples_K = new float[array_size];

    for (int i = 0; i < C_elem; ++i) {
        for (int j = 0; j < K_elem; ++j) {
            h_samples_K[j + K_elem * i] = K_in + j * step_K;
            h_samples_C[j + K_elem * i] = C_in + i * step_C;
        }
    }

    float *d_samples_K, *d_samples_C;

    if(cudaMalloc(&d_samples_K, sizeof(float)*count) != cudaSuccess)
        {
            cout << "Error allocating memory!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            delete[] h_samples_K;
            delete[] h_samples_C;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            return 0;
    };

    if(cudaMalloc(&d_samples_C, sizeof(float)*count) != cudaSuccess)
        {
            cout << "Error allocating memory!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            delete[] h_samples_K;
            delete[] h_samples_C;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            cudaFree(d_samples_K);
            return 0;
    };

    if(cudaMemcpy(d_samples_C, h_samples_C, sizeof(float)*count, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cout << "Could not copy!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            delete[] h_samples_K;
            delete[] h_samples_C;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            cudaFree(d_samples_K);
            cudaFree(d_samples_C);
            return 0;
        }; 
    
    if(cudaMemcpy(d_samples_K, h_samples_K, sizeof(float)*count, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cout << "Could not copy!" << endl;
            delete[] h_x;
            delete[] h_dxdt;
            delete[] h_samples_K;
            delete[] h_samples_C;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            cudaFree(d_samples_K);
            cudaFree(d_samples_C);
            return 0;
        }; 

    SolveODEs<<< count / 256 + 1, 256>>>(d_x, d_dxdt, d_samples_K, d_samples_C, max_iter, count);

    if(cudaDeviceSynchronize()!= cudaSuccess)
        {
            delete[] h_x;
            delete[] h_dxdt;
            delete[] h_samples_K;
            delete[] h_samples_C;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            cudaFree(d_samples_K);
            cudaFree(d_samples_C);
            cout << "Failed Synchronization" << endl;
            return 0;
    };

    if(cudaMemcpy(h_x, d_x, sizeof(float)*count, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            delete[] h_x;
            delete[] h_dxdt;
            delete[] h_samples_K;
            delete[] h_samples_C;
            cudaFree(d_x); 
            cudaFree(d_dxdt); 
            cudaFree(d_samples_K);
            cudaFree(d_samples_C);
            cout << "Nope!" << endl;
            return 0;
        }; // associate pointer to value from the GPU to the CPU
    
    cudaFree(d_x); // free the memory associated to the value in the GPU
    cudaFree(d_dxdt); // free the memory associated to the value in the GPU
    cudaFree(d_samples_K); // free the memory associated to the value in the GPU
    cudaFree(d_samples_C); // free the memory associated to the value in the GPU

    // free the dynamically allocated memory
    delete[] h_dxdt;
    delete[] h_samples_K;
    delete[] h_samples_C;
    for(int i =0; i < 5000; i++){
           cout << "It's " << h_x[i] << endl;
       };
    //// Output the array to a text file, changing column every 1000 elements
    std::ofstream outfile("output.txt");
    if (outfile.is_open()) {
        int col_counter = 0;
        for (int i = 0; i < count; ++i) {
            outfile << std::setprecision(4) << h_x[i] << "\t";
            col_counter++;
            if (col_counter == 1000) {
                outfile << std::endl;
                col_counter = 0;
            }
        }
        outfile.close();
        std::cout << "Array saved to output.txt" << std::endl;
    } else {
        std::cout << "Unable to open file" << std::endl;
    }

    delete[] h_x;

    return 0;
}
