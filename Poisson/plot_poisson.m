load("results_poisson_hmc_1.mat");
plothist(grad_per_ess_hmc,"Grads/ESS - RHMC - Poisson soccer model, d=89526, \kappa≈4\cdot10^3",0.55)
ylim([0,8000]);
sprintf("Max grads/ESS for RHMC: %0.2f", max(grad_per_ess_hmc))
sprintf("95%% bootstrap confidence interval for max grads/ESS for RHMC: (%0.2f,%0.2f)", bootstrap_grad_per_ess_hmc.ci025_max,bootstrap_grad_per_ess_hmc.ci975_max)

load("results_poisson_multi_1.mat");
plothist(grad_per_ess_multi,"Grads/ESS - UBUBU - Poisson soccer model, d=89526, \kappa≈4\cdot10^3",0.56)
ylim([0,17000]);
sprintf("Max grads/ESS for UBUBU: %0.2f", max(grad_per_ess_multi))
sprintf("95%% bootstrap confidence interval for max grads/ESS for UBUBU: (%0.2f,%0.2f)", bootstrap_grad_per_ess_multi.ci025_max,bootstrap_grad_per_ess_multi.ci975_max)

load("results_poisson_multi_approx_1.mat");
plothist(grad_per_ess_multi,"Grads/ESS - Approx UBUBU - Poisson model, d=89526, \kappa≈4\cdot10^3",0.58)
ylim([0,32000]);
ytickformat('%5.0f')
ax = gca;
ax.YAxis.Exponent = 0;
sprintf("Max grads/ESS for Approx UBUBU: %0.2f", max(grad_per_ess_multi))
sprintf("95%% bootstrap confidence interval for max grads/ESS for Approx UBUBU: (%0.2f,%0.2f)", bootstrap_grad_per_ess_multi.ci025_max,bootstrap_grad_per_ess_multi.ci975_max)

