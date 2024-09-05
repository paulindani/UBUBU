
%compile_on_gpu("mnist_hmc.m",{5,5,4,3,0.05,200});
[res,grad_per_ess_hmc,bootstrap_grad_per_ess_hmc]=mnist_hmc_mex(200,40,800,50,1.4,800);
save("results_mnist_hmc_1.mat","grad_per_ess_hmc","bootstrap_grad_per_ess_hmc",'-v7.3');

%compile_on_gpu("mnist_multi.m",{5,5,4,4,1,1,10});
[res,grad_per_ess_multi,bootstrap_grad_per_ess_multi]=mnist_multi_mex(2000,400,64,128,1.5,1, 800);
save("results_mnist_multi_1.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');




%compile_on_gpu("mnist_hmc_precond.m",{5,5,4,3,0.05,200});
[res,grad_per_ess_hmc,bootstrap_grad_per_ess_hmc]=mnist_hmc_precond_mex(200,40,800,10,0.1,800);
save("results_mnist_hmc_precond_1.mat","grad_per_ess_hmc","bootstrap_grad_per_ess_hmc",'-v7.3');

%compile_on_gpu("mnist_multi_precond.m",{5,5,4,4,1,1,10});
[res,grad_per_ess_multi,bootstrap_grad_per_ess_multi]=mnist_multi_precond_mex(200,20,64,128,1,1, 800);
save("results_mnist_multi_precond_1.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');

%compile_on_gpu("mnist_multi_precond_approx.m",{5,5,4,4,1,1,10,1/16,10});
[res,grad_per_ess_multi,bootstrap_grad_per_ess_multi]=mnist_multi_precond_approx_mex(400,40,256,96,0.1,1,40,1/64,400);
save("results_mnist_multi_precond_approx_1.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');

%compile_on_gpu("mnist_multi_precond_svrg.m",{5,5,4,4,1,1,10,1/16,10});
[res,grad_per_ess_multi,bootstrap_grad_per_ess_multi]=mnist_multi_precond_svrg_mex(400,40,256,96,0.12,1,10,1/64,400);
save("results_mnist_multi_precond_svrg_1.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');

