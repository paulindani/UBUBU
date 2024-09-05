%addpath("C:\Matlab\methods\")

grad_per_ess_multi=poisson_multi_approx(2000,400,64,32,1.5,1/(2*sqrt(2)),20,1/16);
save("poisson_multi_approx_grad_per_ess.mat"); 

grad_per_ess_multi=poisson_multi_approx(2000,400,64,64,1.5,1/(2*sqrt(2)),20,1/16);
save("poisson_multi_approx_grad_per_ess_g1.mat"); 

grad_per_ess_multi=poisson_multi(2000,400,64,32,1.5,1/(2*sqrt(2)));
save("poisson_multi_grad_per_ess.mat"); 

grad_per_ess_multi=poisson_multi_approx(100,100,4,4,1.5,1/(2*sqrt(2)),20);


[res,grad_per_ess_multi,bootstrap_grad_per_ess_multi]=poisson_multi(2000,400,64,48,1.5,1,200);
save("results_poisson_multi_1.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');

[res,grad_per_ess_hmc,bootstrap_grad_per_ess_hmc]=poisson_hmc(200,50,256,118,0.17,200);
save("results_poisson_hmc_1.mat","grad_per_ess_hmc","bootstrap_grad_per_ess_hmc",'-v7.3'); 

[res,grad_per_ess_multi,bootstrap_grad_per_ess_multi]=poisson_multi_approx(2000,400,256,48,0.5,1,10,1/64,200);
save("results_poisson_multi_approx_1.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');