#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "global.h"

int main(int argc, char *argv[]){
    // use for debug
    return 0;
}

int UKF(MatrixXd observed_data){
    
    MatrixXd obs_y      = MatrixXd::Zero(NUM_METRICS, 1);
    MatrixXd sys_x      = MatrixXd::Zero(NUM_STATE,   1);
    MatrixXd cov_x      = MatrixXd::Zero(NUM_STATE,   NUM_STATE);
    MatrixXd est_x      = MatrixXd::Zero(NUM_STATE,   1);
    MatrixXd est_y      = MatrixXd::Zero(NUM_METRICS, 1);
    MatrixXd sig_x      = MatrixXd::Zero(NUM_STATE,   SIGMA_POINTS);
    MatrixXd sig_y      = MatrixXd::Zero(NUM_METRICS, SIGMA_POINTS);
    MatrixXd cov_yy     = MatrixXd::Zero(NUM_METRICS, NUM_METRICS);
    MatrixXd cov_xy     = MatrixXd::Zero(NUM_STATE,   NUM_METRICS);
    MatrixXd est_cov    = MatrixXd::Zero(NUM_STATE,   NUM_STATE);
    MatrixXd sys_x_prev = MatrixXd::Zero(NUM_STATE,   1);
    MatrixXd cov_x_prev = MatrixXd::Zero(NUM_STATE,   NUM_STATE);
    MatrixXd process_noise(NUM_STATE,   NUM_STATE);   // if variables are independent of each other, matrix is symmetric
    MatrixXd measure_noise(NUM_METRICS, NUM_METRICS);
    MatrixXd kalman_gain  (NUM_STATE,   NUM_STATE);

    //
    // initialization for sys_x_prev & cov_x_prev
    // 
    sys_x_prev       = system_average;
    cov_x_prev       = system_covariance;
    process_noise    = system_noise_w;
    measure_noise    = system_noise_v;
    obs_y            = observed_data;

    //
    // calculate sigma point from the state one step before
    // 
    LLT<MatrixXd> lltOfP(cov_x_prev);
    MatrixXd L   = lltOfP.matrixL();
    sig_x.col(0) = sys_x_prev;
    for (int i = 0; i < NUM_STATE; i++) sig_x.col(i+1)           = sys_x_prev + GAMMA_PARAM*L.col(i);
    for (int i = 0; i < NUM_STATE; i++) sig_x.col(i+NUM_STATE+1) = sys_x_prev - GAMMA_PARAM*L.col(i);

    //
    // state update for sigma points
    // 
    for (int i = 0; i < SIGMA_POINTS; i++) { sig_x.col(i) = generative_model(sig_x.col(i)); }

    //
    // update estimates: x, p
    // 
    for (int i = 0; i < SIGMA_POINTS; i++) {
        double weight_m = i == 0 ? LAMBDA_PARAM / (L_PARAM + LAMBDA_PARAM) : 1 / (2*L_PARAM + 2*LAMBDA_PARAM);
        est_x += weight_m*sig_x.col(i);
    }
    for (int i = 0; i < SIGMA_POINTS; i++) {
        double weight_c = i == 0 ? LAMBDA_PARAM / (L_PARAM + LAMBDA_PARAM) + 1 - ALPHA_PARAM*ALPHA_PARAM + BETA_PARAM : 1 / (2*L_PARAM + 2*LAMBDA_PARAM);
        est_cov += weight_c*(sig_x.col(i) - est_x)*((sig_x.col(i) - est_x).transpose());
    }
    est_cov += process_noise;

    //
    // NOT augment sigma points and update
    // 
    LLT<MatrixXd> lltOfCOV(est_cov);
    L = lltOfCOV.matrixL();
    sig_x.col(0) = est_x;
    for (int i = 0; i < NUM_STATE; i++) sig_x.col(i+1)           = est_x + GAMMA_PARAM*L.col(i);
    for (int i = 0; i < NUM_STATE; i++) sig_x.col(i+NUM_STATE+1) = est_x - GAMMA_PARAM*L.col(i);
    sig_y = feature_vector*sigmoid(sig_x);

    //
    // update estimates: y using [k-1]
    // 
    for (int i = 0; i < SIGMA_POINTS; i++) {
        double weight_m = i == 0 ? LAMBDA_PARAM / (L_PARAM + LAMBDA_PARAM) : 1 / (2*L_PARAM + 2*LAMBDA_PARAM);
        est_y += weight_m*sig_y.col(i);
    }

    //
    // measurement update
    // 
    for (int i = 0; i < SIGMA_POINTS; i++) {
        double weight_c = i == 0 ? LAMBDA_PARAM / (L_PARAM + LAMBDA_PARAM) + 1 - ALPHA_PARAM*ALPHA_PARAM + BETA_PARAM : 1 / (2*L_PARAM + 2*LAMBDA_PARAM);
        cov_yy += weight_c * (sig_y.col(i) - est_y)*((sig_y.col(i) - est_y).transpose()); 
        cov_xy += weight_c * (sig_x.col(i) - est_x)*((sig_y.col(i) - est_y).transpose()); 
    }
    cov_yy += measure_noise;

    //
    // obtain K, and get estimates
    // 
    kalman_gain = cov_xy  * cov_yy.inverse();
    sys_x       = est_x   + kalman_gain*(obs_y-est_y);
    cov_x       = est_cov - kalman_gain*cov_yy*(kalman_gain.transpose());
    
    //
    // for next step
    //
    system_average    = sys_x;
    system_covariance = cov_x;
    if(DEBUG_MSG_ON==1){ cout << "    z: "; for(int i=0;i<sys_x.rows();i++) for(int j=0;j<sys_x.cols();j++) cout << sys_x(i,j) << " "; cout << endl << endl; }

    //
    // posterior probability
    //
    for (int i = 0; i < NUM_STATE; i++) {
        confidence(i, 0) = prob_mnorm(attractor_vector.col(i), sys_x, cov_x);
    }
    return 1;

}

MatrixXd generative_model(MatrixXd d) {
    MatrixXd ones_nn = MatrixXd::Ones(NUM_STATE, NUM_STATE);
    MatrixXd ones_n  = MatrixXd::Ones(NUM_STATE, 1);
    MatrixXd unit    = MatrixXd::Identity(NUM_STATE, NUM_STATE);
    MatrixXd z       = d;
    MatrixXd L       = B_LAT*(unit - ones_nn);
    return z + DELTA_T * K_CONST * (L*sigmoid(z) + B_LIN*(G_CONST*ones_n - z));
}

MatrixXd sigmoid(MatrixXd m){
    MatrixXd _m = MatrixXd::Zero(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); i++) { for (int j = 0; j < m.cols(); j++) { _m(i, j) = 1.0 / (1 + exp(-SIG_R*(m(i, j) - SIG_O))); } }
    return _m;
}

double prob_mnorm(MatrixXd x, MatrixXd m, MatrixXd p) {
    double   a = (1.0 / pow(2 * PI, m.size() / 2)) * (1.0 / sqrt(p.determinant()));
    MatrixXd b = (x-m).transpose()*p.inverse()*(x-m);
    return a*exp(-0.5*b(0, 0));
}

void init_z(void){
  for (int i = 0; i < NUM_STATE; i++){ system_average(i,0) = dist(engine); }
}

void change_f_dim(int f_dim){ 
    NUM_METRICS = f_dim; 
    system_noise_v.resize(NUM_METRICS, NUM_METRICS); system_noise_v = MatrixXd::Zero(NUM_METRICS,NUM_METRICS); 
    feature_vector.resize(NUM_METRICS, NUM_STATE);   feature_vector = MatrixXd::Zero(NUM_METRICS,NUM_STATE); 
}

void change_k_dim(int k_dim){ 
    NUM_STATE     = k_dim; 
    L_PARAM       = NUM_STATE;
    KAPPA_PARAM   = 3 - static_cast<double>(L_PARAM);
    LAMBDA_PARAM  = ALPHA_PARAM*ALPHA_PARAM *(L_PARAM+KAPPA_PARAM)-L_PARAM;
    GAMMA_PARAM   = sqrt(L_PARAM+LAMBDA_PARAM);
    SIGMA_POINTS  = 2 * NUM_STATE + 1;
    system_average.resize(NUM_STATE, 1);            system_average    = MatrixXd::Zero(NUM_STATE, 1); 
    system_covariance.resize(NUM_STATE, NUM_STATE); system_covariance = MatrixXd::Zero(NUM_STATE, NUM_STATE);
    system_noise_w.resize(NUM_STATE, NUM_STATE);    system_noise_w    = MatrixXd::Zero(NUM_STATE, NUM_STATE);
    attractor_vector.resize(NUM_STATE, NUM_STATE);  attractor_vector  = MatrixXd::Zero(NUM_STATE, NUM_STATE);
    confidence.resize(NUM_STATE, 1);                confidence        = MatrixXd::Zero(NUM_STATE, 1);
    feature_vector.resize(NUM_METRICS, NUM_STATE);  feature_vector    = MatrixXd::Zero(NUM_METRICS, NUM_STATE);
    init_attractor();
}


void init_attractor(void){
    for (int i = 0; i < NUM_STATE; i++) {
        for (int j = 0; j < NUM_STATE; j++) {
            if (i == j) { attractor_vector(i, j) = G_CONST;  }
            else        { attractor_vector(i, j) = -G_CONST; }
        }
    }
}

void init_nrm_variables(void){
    normalized_m       = MatrixXd::Zero(NUM_METRICS, 1);
    normalized_s       = MatrixXd::Zero(NUM_METRICS, 1);
    is_normalize_param_set = false;
}

void set_d_uncertain(double d){
    for (int i = 0; i < NUM_STATE; i++) {
        for (int j = 0; j < NUM_STATE; j++) {
            if (i == j) { system_noise_w(i, j)   = d; }
            else        { system_noise_w(i, j)   = 0; }
        }
    }
}

void set_s_uncertain(double s){
    for (int i = 0; i < NUM_METRICS; i++) {
        for (int j = 0; j < NUM_METRICS; j++) {
            if (i == j) { system_noise_v(i, j)   = s; }
            else        { system_noise_v(i, j)   = 0; }
        }
    }
}

void upd_feature(MatrixXd d) {
    if(!is_normalize_param_set){
        cerr << "normalized param is not set" << endl;
        feature_vector(j,i) = d(i,j);
        //exit(0);
    }
    else{
        for (int i = 0; i < NUM_STATE; i++) {
            for (int j = 0; j < NUM_METRICS; j++) {
                feature_vector(j,i) = (d(i,j) - normalized_m(j,0))/normalized_s(j,0);
                if(DEBUG_MSG_ON==1){ cout << "M("<<j<<","<<i<<") is "<<d(i,j)<<endl; } 
            }
        }
    }
}

void debug_msg(int i){
    DEBUG_MSG_ON = i;
}

void set_norm_param(MatrixXd M, MatrixXd S){
    for (int j = 0; j < NUM_METRICS; j++) {
        normalized_m(j,0) = M(j,0);
        normalized_s(j,0)  = S(j,0);
    }
}

bam::bam(int k_dim, int f_dim, double q, double r){ 
    DEBUG_MSG_ON = 0;
    init_z(); 
    change_k_dim(k_dim);
    change_f_dim(f_dim);
    init_nrm_variables();
    set_d_uncertain(q*q); 
    set_s_uncertain(r*r); 
}

MatrixXd bam::get_z(void){ return system_average;    }
MatrixXd bam::get_p(void){ return system_covariance; }
MatrixXd bam::get_c(void){ return confidence;        }

void bam::ukf_z(MatrixXd d){ 
    MatrixXd input = MatrixXd::Zero(NUM_METRICS,1); 
    if(!is_normalize_param_set){
        cerr << "normalized param is not set" << endl;
        input(i,0) = d(0,i);
        //exit(0);
    }
    else{
        for(int i=0; i<NUM_METRICS; i++){ input(i,0)=(d(0,i)-normalized_m(i,0))/normalized_s(i,0); } 
        
        if(DEBUG_MSG_ON==1){ 
            cout << "input: ";
            for(int i=0; i<NUM_METRICS-1; i++){
                cout << input(i,0) << ", "; 
            }
            cout << input(NUM_METRICS-1,0) << endl;
        }
    }
    UKF(input); 
}

void bam::set_f_dim(int f_dim){ change_f_dim(f_dim); }
void bam::set_k_dim(int k_dim){ change_k_dim(k_dim); }
void bam::set_q(double q)     { set_d_uncertain(q);  }
void bam::set_r(double r)     { set_d_uncertain(r);  }
void bam::upd_f(MatrixXd M)   { upd_feature(M);      }
void bam::msg_on(void)        { debug_msg(1);        }

void bam::set_norm_prm(MatrixXd M, MatrixXd S){ set_norm_param(M,S); is_normalize_param_set = true; }

PYBIND11_MODULE(bam_module, m) {
    py::class_<bam>(m, "bam")
    .def(py::init<int,int,double,double>())
    .def("get_z", &bam::get_z)
    .def("get_p", &bam::get_p)
    .def("get_c", &bam::get_c)
    .def("set_f_dim", &bam::set_f_dim)
    .def("set_k_dim", &bam::set_k_dim)
    .def("set_q", &bam::set_q)
    .def("set_r", &bam::set_r)
    .def("ukf_z", &bam::ukf_z)
    .def("upd_f", &bam::upd_f)
    .def("msg_on", &bam::msg_on)
    .def("set_norm_prm", &bam::set_norm_prm)
  ;
}

