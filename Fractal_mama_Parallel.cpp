#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <filesystem> // For creating directories
#include <chrono> // For time measurements
#include <utility> // For std::pair
#include <thread>

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

using namespace Eigen;

using namespace std;
using namespace boost::numeric::odeint;
using namespace std::chrono;

typedef boost::array< double , 2> state_type;
runge_kutta4< state_type > stepper;
    

struct Constants
{
    double w = 105 * M_PI;
    double w_0 = 3.0 / 2.0 * w;
    double gamma = 1.5;
    double f = gamma * w_0 * w_0;
    double y0 = 0.0, y1 = 0.0;
    double beta = 3.0 / 4.0 * M_PI;
    // C grid 
    int C_elem = 1000;
    double C_fin = 10.0;
    double C_in = 1.0;
    VectorXd samples_C = VectorXd::LinSpaced(C_elem, C_in, C_fin);
    // K grid 
    int K_elem = 1000;
    double K_fin = 100.0;
    double K_in = 40.0;
    VectorXd samples_K = VectorXd::LinSpaced(K_elem, K_in, K_fin);
    // t grid 
    double t_max = 10.0;
    double dt = 0.01;
    int grid_size = static_cast<int>(t_max / dt) + 1; // static_cast should do the conversion from float to int in this case
    VectorXd t = VectorXd::LinSpaced(grid_size, 0, t_max); // vector of the eigen3 lib
}; 


//[ integrate_observer
struct push_back_state_and_time
{
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};
//]

struct pendulum_ode
{
    Constants data; 
    const double &K_value;
    const double &C_value;

    pendulum_ode( const double &K,  const double &C) : K_value( K ),  C_value( C ){ }

    void operator()( const state_type &x , state_type &dxdt, double t ) const
    {
        dxdt[0] = x[1];
        dxdt[1] = data.gamma*K_value* cos(data.w * t) - K_value*sin(x[0]) -C_value* x[1];
    }
};

MatrixXd JM(const double x1,const double c,const  double k) {
    MatrixXd result(2, 2);
    result << 0, -1, -k * cos(x1), -c;
    return result;
}

MatrixXd d_JM() {
    MatrixXd result(2, 2);
    result << 1, 0, 0, -1;
    return result;
}

pair<MatrixXd, MatrixXd> myqr(MatrixXd A) {
    HouseholderQR<MatrixXd> qr(A);
    MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixQR().triangularView<Upper>();
    int nn = 2;
    for (int ii = 0; ii < nn; ii++) {
        if (R(ii, ii) < 0) {
            Q.col(ii) *= -1;
            R.row(ii) *= -1;
        }
    }
    return {Q, R};
}

std::pair<double, double> LCE(const size_t &steps,const vector<state_type> &x_vec,const vector<double> &times, const double &K,  const double &C) {
    Constants data; 

    double ll_1(2); // Initialize vector with two zeros
    double ll_2(2); // Initialize vector with two zeros
    
    // Define and initialize ll_curr as a vector of size 2 filled with zeros
    VectorXd ll_curr = VectorXd::Zero(2);
    VectorXd tmp = VectorXd::Zero(2);
    // Y0 = np.eye(2)
    MatrixXd Yk = MatrixXd::Identity(2, 2);
    MatrixXd Q = MatrixXd::Identity(2, 2);
    MatrixXd R = MatrixXd::Identity(2, 2);
    // Compute QR decomposition of Y0
    HouseholderQR<MatrixXd> qr(Yk);
    MatrixXd Qk = qr.householderQ();
    MatrixXd Rk = qr.matrixQR().triangularView<Upper>();

    for (size_t i = 1; i <=  steps; i++){
        // Assuming d_JM() and A are already defined
        MatrixXd d_JM_result = d_JM(); // Assuming d_JM() returns a MatrixXd
        MatrixXd A = JM(x_vec[i][0], C, K); // Assuming A is already defined as a MatrixXd

        // Solve the least squares problem directly
        // Define matrices d_JM and A, and scalar dt

        //MatrixXd d_JM_plus = d_JM() + 0.5 * data.dt * A;
        //MatrixXd d_JM_minus = d_JM() - 0.5 * data.dt * A;

        //MatrixXd F = d_JM_plus.jacobiSvd(ComputeThinU | ComputeThinV).solve(d_JM_minus);
        MatrixXd F = (d_JM() + data.dt/2.0 * A).jacobiSvd(ComputeThinU | ComputeThinV).solve(d_JM() - data.dt/2.0 * A);
        Qk = F * Qk;
        Yk = F * Yk;
        // Perform QR decomposition
        auto [Q, R] = myqr(Qk);
        Qk = Q;
        Rk = R;
        tmp = (Rk.diagonal().array().log());
        ll_curr += tmp;
        ll_1 = ll_curr[0]/times[i];
        ll_2 = ll_curr[1]/times[i];
    }
    return {ll_1, ll_2};
}

void saveMatrixXdToFile(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << matrix << std::endl;
        outFile.close();
        std::cout << "Matrix saved to file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Define Constants and other necessary types and functions

int main() {
    Constants data;
    MatrixXd ll_1 = MatrixXd::Zero(data.C_elem, data.K_elem);
    MatrixXd ll_2 = MatrixXd::Zero(data.C_elem, data.K_elem);
    std::pair<double, double> lyapunovExponent;

    // Check if the 'data' subfolder exists
    if (!filesystem::exists("data")) {
        filesystem::create_directory("data");
    } else {
        filesystem::remove_all("data");
        filesystem::create_directory("data");
    }

    auto start = high_resolution_clock::now();

    // Parallelized loop using threads
    unsigned int num_threads = std::thread::hardware_concurrency()-1;
    vector<thread> threads;
    for (size_t i = 0; i < data.K_elem; ++i) {
        threads.emplace_back([&, i]() {
            for (size_t j = 0; j < data.C_elem; ++j) {
                vector<state_type> x_vec;
                vector<double> times;

                state_type x_0 = {.0, .0}; // initial conditions
                size_t steps = integrate_const(stepper, pendulum_ode(data.samples_K[i], data.samples_C[j]), x_0, 0.0, data.t_max, data.dt, push_back_state_and_time(x_vec, times));

                lyapunovExponent = LCE(steps, x_vec, times, data.samples_K[i], data.samples_C[j]);
                ll_1(i, j) = lyapunovExponent.first;
                ll_2(i, j) = lyapunovExponent.second;

                // Clear vectors
                x_vec.clear();
                times.clear();
            }
            cout << i << endl;
        });
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end - start);
    cout << "Total time taken: " << duration.count() << " seconds" << endl;

    // Save ll_1 to a file
    saveMatrixXdToFile(ll_1, "data/ll_1.txt");
    // Save ll_2 to a file
    saveMatrixXdToFile(ll_2, "data/ll_2.txt");

    return 0;
}