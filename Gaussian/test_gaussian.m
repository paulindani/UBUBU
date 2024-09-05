addpath("C:\Matlab\methods\")
clear all;

%parpool(feature('numcores')) %start parallel pool  


run1e5=true;
save_means=false;
tot=4+run1e5;

%ALL COMPONENTS
%kappa=4

grad_per_ess_hmc=cell(tot,1);
res_hmc=cell(tot,1);
[res_hmc{1},grad_per_ess_hmc{1},bootstrap_grad_per_ess_hmc{1}]=gaussian_hmc(200,0,512,1,1.2,10,4,400);
[res_hmc{2},grad_per_ess_hmc{2},bootstrap_grad_per_ess_hmc{2}]=gaussian_hmc(200,0,512,2,0.60,100,4,400);
[res_hmc{3},grad_per_ess_hmc{3},bootstrap_grad_per_ess_hmc{3}]=gaussian_hmc(200,0,512,2,0.42,1000,4,400);
[res_hmc{4},grad_per_ess_hmc{4},bootstrap_grad_per_ess_hmc{4}]=gaussian_hmc(200,0,512,4,0.25,10000,4,400);
if(run1e5)
    [res_hmc{5},grad_per_ess_hmc{5},bootstrap_grad_per_ess_hmc{5}]=gaussian_hmc(200,0,512,8,0.15,100000,4,400);
end
if(save_means)
    save("gaussian_hmc_kappa_4.mat","grad_per_ess_hmc","res_hmc","bootstrap_grad_per_ess_hmc",'-v7.3');
else
    save("gaussian_hmc_kappa_4.mat","grad_per_ess_hmc","bootstrap_grad_per_ess_hmc",'-v7.3');
end

compile_on_gpu("gaussian_multi.m",{5,5,4,4,1,1,10,4,10});


grad_per_ess_multi=cell(tot,1);
res_multi=cell(tot,1);
[res_multi{1},grad_per_ess_multi{1},bootstrap_grad_per_ess_multi{1}]=gaussian_multi(200,10,128,96,1.5,1,10,4,400);
[res_multi{2},grad_per_ess_multi{2},bootstrap_grad_per_ess_multi{2}]=gaussian_multi(200,10,128,96,1.5,1,100,4,400);
[res_multi{3},grad_per_ess_multi{3},bootstrap_grad_per_ess_multi{3}]=gaussian_multi(200,10,128,96,1.5,1,1000,4,400);
[res_multi{4},grad_per_ess_multi{4},bootstrap_grad_per_ess_multi{4}]=gaussian_multi(200,10,128,96,1.5,1,10000,4,400);
if(run1e5)
    [res_multi{5},grad_per_ess_multi{5},bootstrap_grad_per_ess_multi{5}]=gaussian_multi(200,10,128,96,1.5,1,100000,4,400);
end
if(save_means)
save("gaussian_multi_kappa_4.mat","res_multi","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');
else
save("gaussian_multi_kappa_4.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');
end


%ALL COMPONENTS
%kappa=100

grad_per_ess_hmc=cell(tot,1);
res_hmc=cell(tot,1);
[res_hmc{1},grad_per_ess_hmc{1},bootstrap_grad_per_ess_hmc{1}]=gaussian_hmc(200,0,512,4,1,10,100,400);
[res_hmc{2},grad_per_ess_hmc{2},bootstrap_grad_per_ess_hmc{2}]=gaussian_hmc(200,0,512,8,0.5,100,100,400);
[res_hmc{3},grad_per_ess_hmc{3},bootstrap_grad_per_ess_hmc{3}]=gaussian_hmc(200,0,512,15,0.25,1000,100,400);
[res_hmc{4},grad_per_ess_hmc{4},bootstrap_grad_per_ess_hmc{4}]=gaussian_hmc(200,0,512,33,0.15,10000,100,400);
if(run1e5)
[res_hmc{5},grad_per_ess_hmc{5},bootstrap_grad_per_ess_hmc{5}]=gaussian_hmc(200,0,512,59,0.17,100000,100,400);
end
if(save_means)
save("gaussian_hmc_kappa_100.mat","res_hmc","grad_per_ess_hmc","bootstrap_grad_per_ess_hmc",'-v7.3');
else
save("gaussian_hmc_kappa_100.mat","grad_per_ess_hmc","bootstrap_grad_per_ess_hmc",'-v7.3');
end


grad_per_ess_multi=cell(tot,1);
res_multi=cell(tot,1);
[res_multi{1},grad_per_ess_multi{1},bootstrap_grad_per_ess_multi{1}]=gaussian_multi(200,20,128,96,2,1,10,100,400);
[res_multi{2},grad_per_ess_multi{2},bootstrap_grad_per_ess_multi{2}]=gaussian_multi(200,20,128,96,2,1,100,100,400);
[res_multi{3},grad_per_ess_multi{3},bootstrap_grad_per_ess_multi{3}]=gaussian_multi(200,20,128,96,2,1,1000,100,400);
[res_multi{4},grad_per_ess_multi{4},bootstrap_grad_per_ess_multi{4}]=gaussian_multi(200,20,128,96,2,1,10000,100,400);
if(run1e5)
[res_multi{5},grad_per_ess_multi{5},bootstrap_grad_per_ess_multi{5}]=gaussian_multi(200,20,128,96,2,1,100000,100,400);
end
if(save_means)
save("gaussian_multi_kappa_100.mat","res_multi","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');
else
save("gaussian_multi_kappa_100.mat","grad_per_ess_multi","bootstrap_grad_per_ess_multi",'-v7.3');    
end