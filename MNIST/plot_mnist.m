format short

load("results_mnist_hmc_1.mat")
plothist(grad_per_ess_hmc,"Gradients/ESS - RHMC - MNIST,d=7850,\kappa≈7.2\cdot10^3",0.57)
xlim([130,260])
sprintf("Max grads/ESS for RHMC: %0.2f", max(grad_per_ess_hmc))
sprintf("95%% bootstrap confidence interval for max grads/ESS for RHMC: (%0.2f,%0.2f)", bootstrap_grad_per_ess_hmc.ci025_max,bootstrap_grad_per_ess_hmc.ci975_max)


load("results_mnist_multi_1.mat");
plothist(grad_per_ess_multi,"Gradients/ESS - UBUBU - MNIST,d=7850,\kappa≈7.2\cdot10^3",0.15,0.75)
sprintf("Max grads/ESS for UBUBU: %0.2f", max(grad_per_ess_multi))
sprintf("95%% bootstrap confidence interval for max grads/ESS for UBUBU: (%0.2f,%0.2f)", bootstrap_grad_per_ess_multi.ci025_max,bootstrap_grad_per_ess_multi.ci975_max)



load("results_mnist_hmc_precond_1.mat")
plothist(grad_per_ess_hmc(1:7850),"Gradients/ESS - Preconditioned RHMC - MNIST,d=7850,\kappa≈1",0.58)
sprintf("Max grads/ESS for Preconditioned RHMC: %0.2f", max(grad_per_ess_hmc(1:7850)))
sprintf("95%% bootstrap confidence interval for max grads/ESS for Preconditioned RHMC: (%0.2f,%0.2f)", bootstrap_grad_per_ess_hmc.ci025_max,bootstrap_grad_per_ess_hmc.ci975_max)

load("results_mnist_multi_precond_1.mat")
plothist(grad_per_ess_multi(1:7850),"Gradients/ESS - Preconditioned UBUBU - MNIST,d=7850,\kappa≈1",0.6)
sprintf("Max grads/ESS for Preconditioned UBUBU: %0.2f", max(grad_per_ess_multi(1:7850)))
sprintf("95%% bootstrap confidence interval for max grads/ESS for Preconditioned UBUBU: (%0.2f,%0.2f)", bootstrap_grad_per_ess_multi.ci025_max,bootstrap_grad_per_ess_multi.ci975_max)

load('results_mnist_multi_precond_svrg_1.mat')
plothist(grad_per_ess_multi(1:7850),"Gradients/ESS - Preconditioned UBUBU-SG - MNIST,d=7850,\kappa≈1",0.6,0.75)
sprintf("Max grads/ESS for Preconditioned SVRG UBUBU: %0.2f", max(grad_per_ess_multi(1:7850)))
sprintf("95%% bootstrap confidence interval for max grads/ESS for Preconditioned UBUBU-SG: (%0.2f,%0.2f)", bootstrap_grad_per_ess_multi.ci025_max,bootstrap_grad_per_ess_multi.ci975_max)

load("results_mnist_multi_precond_approx_1.mat")
plothist(grad_per_ess_multi(1:7850),"Gradients/ESS - Precond UBUBU-Approx - MNIST,d=7850,\kappa≈1", 0.6)
sprintf("Max grads/ESS for Preconditioned Approx UBUBU: %0.3f", max(grad_per_ess_multi(1:7850)))
sprintf("95%% bootstrap confidence interval for max grads/ESS for Preconditioned Approx UBUBU: (%0.3f,%0.3f)", bootstrap_grad_per_ess_multi.ci025_max,bootstrap_grad_per_ess_multi.ci975_max)


load('results_mnist_hmc_precond_1.mat')
plothist(grad_per_ess_hmc(7851:end),"Gradients/ESS - Preconditioned RHMC - Test data - MNIST - \kappa≈1",0.58)
sprintf("Max grads/ESS for Preconditioned RHMC - Test set: %0.2f", max(grad_per_ess_hmc(7851:end)))
max_grad_per_ess=max(bootstrap_grad_per_ess_hmc.grad_per_ess_arr(7851:end,:),[],1);
ci025_max=quantile(max_grad_per_ess,0.025);   
ci975_max=quantile(max_grad_per_ess,0.975);
sprintf("95%% bootstrap confidence interval for max grads/ESS for Preconditioned RHMC - Test set: (%0.2f,%0.2f)", ci025_max,ci975_max)

load("results_mnist_multi_precond_approx_1.mat")
plothist(grad_per_ess_multi(7851:end),"Gradients/ESS - Precond UBUBU-Approx - Test data - MNIST - \kappa≈1",0.6)
sprintf("Max grads/ESS for Preconditioned Approx UBUBU - Test set: %0.2f", max(grad_per_ess_multi(7851:end)))
max_grad_per_ess=max(bootstrap_grad_per_ess_multi.grad_per_ess_arr(7851:end,:),[],1);
ci025_max=quantile(max_grad_per_ess,0.025);   
ci975_max=quantile(max_grad_per_ess,0.975);
sprintf("95%% bootstrap confidence interval for max grads/ESS for Preconditioned UBUBU-Approx - Test set: (%0.2f,%0.2f)", ci025_max,ci975_max)




