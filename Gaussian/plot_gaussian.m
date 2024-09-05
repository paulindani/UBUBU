function plot_gaussian
r_multi=load("gaussian_multi_kappa_4.mat");
r_hmc=load("gaussian_hmc_kappa_4.mat");
grad_per_ess_hmc=r_hmc.grad_per_ess_hmc;
grad_per_ess_multi=r_multi.grad_per_ess_multi;
bootstrap_grad_per_ess_hmc=r_hmc.bootstrap_grad_per_ess_hmc;
bootstrap_grad_per_ess_multi=r_multi.bootstrap_grad_per_ess_multi;

plot_hist_comp1
plot_ess_all_comp1
plot_ess_norm_comp1


r_multi=load("gaussian_multi_kappa_100.mat");
r_hmc=load("gaussian_hmc_kappa_100.mat");
grad_per_ess_hmc=r_hmc.grad_per_ess_hmc;
grad_per_ess_multi=r_multi.grad_per_ess_multi;
bootstrap_grad_per_ess_hmc=r_hmc.bootstrap_grad_per_ess_hmc;
bootstrap_grad_per_ess_multi=r_multi.bootstrap_grad_per_ess_multi;

plot_hist_comp2
plot_ess_all_comp2
plot_ess_norm_comp2


function plot_hist_comp1
plothist(grad_per_ess_hmc{5}," Gradients/ESS - RHMC - Gaussian,d=10^5,\kappa=4",0.56,0.75)
plothist(grad_per_ess_multi{5}," Gradients/ESS - UBUBU - Gaussian,d=10^5,\kappa=4",0.56,0.75)
end
function plot_hist_comp2
plothist(grad_per_ess_hmc{5}," Gradients/ESS - RHMC - Gaussian,d=10^5,\kappa=100",0.56,0.75)
plothist(grad_per_ess_multi{5}," Gradients/ESS - UBUBU - Gaussian,d=10^5,\kappa=100",0.56,0.75)
end

function plot_ess_all_comp1
max_grad_per_ess_hmc=zeros(1,5);
sd_max_grad_per_ess_hmc=zeros(1,5);
max_grad_per_ess_multi=zeros(1,5);
err_neg_max_grad_per_ess_hmc=zeros(1,5);
err_pos_max_grad_per_ess_hmc=zeros(1,5);
err_neg_max_grad_per_ess_multi=zeros(1,5);
err_pos_max_grad_per_ess_multi=zeros(1,5);

for(it=1:5)  
    max_grad_per_ess_hmc(it)=max(grad_per_ess_hmc{it}(1:10^(it)));
    max_grad_per_ess_multi(it)=max(grad_per_ess_multi{it}(1:10^(it)));
    err_neg_max_grad_per_ess_hmc(it)=max_grad_per_ess_hmc(it)-bootstrap_grad_per_ess_hmc{it}.ci025_max(end);
    err_pos_max_grad_per_ess_hmc(it)=-grad_per_ess_hmc{it}(end)+bootstrap_grad_per_ess_hmc{it}.ci975_max(end);
    err_neg_max_grad_per_ess_multi(it)=max_grad_per_ess_multi(it)-bootstrap_grad_per_ess_multi{it}.ci025_max(end);
    err_pos_max_grad_per_ess_multi(it)=-grad_per_ess_multi{it}(end)+bootstrap_grad_per_ess_multi{it}.ci975_max(end);
end

d=[10, 100,1000,10000,100000];

figure, errorbar(d,max_grad_per_ess_hmc,err_neg_max_grad_per_ess_hmc,err_pos_max_grad_per_ess_hmc,'MarkerFaceColor',[0 0.447 0.741]);%'s','MarkerFaceColor',[0 0.447 0.741])
ax=gca;
ax.XScale='log';
ax.YScale='log';
set(gcf,'position',[200 200 750 500])

fontsize(16,"points")
ylim([3,300]);
xlim([7,150000]);

hold on;
plot(d,d.^(1/4)*4,'LineWidth',1.5);
errorbar(d,max_grad_per_ess_multi,err_neg_max_grad_per_ess_multi,err_pos_max_grad_per_ess_multi,'MarkerFaceColor',[0 0 0]);%'-o','MarkerFaceColor',[0 0 0])
legend('Max grads/ESS vs dimension - RHMC', '4\cdotd^{1/4}','Max grads/ESS vs dimension - UBUBU','Location',[0.33 0.7 0.3 0.2],'Fontsize',16);
title('Max gradients/ESS vs dimension - Gaussian - \kappa=4')
xlabel('Dimension')
ylabel('Max gradients/ESS')

end



function plot_ess_all_comp2
max_grad_per_ess_hmc=zeros(1,5);
sd_max_grad_per_ess_hmc=zeros(1,5);
max_grad_per_ess_multi=zeros(1,5);
err_neg_max_grad_per_ess_hmc=zeros(1,5);
err_pos_max_grad_per_ess_hmc=zeros(1,5);
err_neg_max_grad_per_ess_multi=zeros(1,5);
err_pos_max_grad_per_ess_multi=zeros(1,5);

for(it=1:5)  
    max_grad_per_ess_hmc(it)=max(grad_per_ess_hmc{it}(1:10^(it)));
    max_grad_per_ess_multi(it)=max(grad_per_ess_multi{it}(1:10^(it)));
    err_neg_max_grad_per_ess_hmc(it)=max_grad_per_ess_hmc(it)-bootstrap_grad_per_ess_hmc{it}.ci025_max(end);
    err_pos_max_grad_per_ess_hmc(it)=-grad_per_ess_hmc{it}(end)+bootstrap_grad_per_ess_hmc{it}.ci975_max(end);
    err_neg_max_grad_per_ess_multi(it)=max_grad_per_ess_multi(it)-bootstrap_grad_per_ess_multi{it}.ci025_max(end);
    err_pos_max_grad_per_ess_multi(it)=-grad_per_ess_multi{it}(end)+bootstrap_grad_per_ess_multi{it}.ci975_max(end);
end

d=[10, 100,1000,10000,100000];

figure, errorbar(d,max_grad_per_ess_hmc,err_neg_max_grad_per_ess_hmc,err_pos_max_grad_per_ess_hmc,'MarkerFaceColor',[0 0.447 0.741]);%'s','MarkerFaceColor',[0 0.447 0.741])
ax=gca;
ax.XScale='log';
ax.YScale='log';
set(gcf,'position',[200 200 750 500])

fontsize(16,"points")
ylim([10,1600]);
xlim([7,150000]);

hold on;
plot(d,d.^(1/4)*18,'LineWidth',1.5);
errorbar(d,max_grad_per_ess_multi,err_neg_max_grad_per_ess_multi,err_pos_max_grad_per_ess_multi,'MarkerFaceColor',[0 0 0]);%'-o','MarkerFaceColor',[0 0 0])
legend('Max grads/ESS vs dimension - RHMC', '18\cdotd^{1/4}','Max grads/ESS vs dimension - UBUBU','Location',[0.33 0.7 0.3 0.2],'Fontsize',16);
title('Max gradients/ESS vs dimension - Gaussian - \kappa=100')
xlabel('Dimension')
ylabel('Max gradients/ESS')

end

function plot_ess_norm_comp1
    norm_grad_per_ess_hmc=zeros(1,5);
    norm_grad_per_ess_multi=zeros(1,5);
    err_neg_norm_grad_per_ess_hmc=zeros(1,5);
    err_pos_norm_grad_per_ess_hmc=zeros(1,5);
    err_neg_norm_grad_per_ess_multi=zeros(1,5);
    err_pos_norm_grad_per_ess_multi=zeros(1,5);

    %sd_norm_grad_per_ess_hmc=zeros(1,5);
    %sd_norm_grad_per_ess_multi=zeros(1,5);
    
for(it=1:5)  
    norm_grad_per_ess_multi(it)=grad_per_ess_multi{it}(end);
    norm_grad_per_ess_hmc(it)=grad_per_ess_hmc{it}(end);   
    err_neg_norm_grad_per_ess_multi(it)=grad_per_ess_multi{it}(end)-max(bootstrap_grad_per_ess_multi{it}.ci025(end),0.001);
    err_pos_norm_grad_per_ess_multi(it)=-grad_per_ess_multi{it}(end)+bootstrap_grad_per_ess_multi{it}.ci975(end);
    err_neg_norm_grad_per_ess_hmc(it)=grad_per_ess_hmc{it}(end)-bootstrap_grad_per_ess_hmc{it}.ci025(end);
    err_pos_norm_grad_per_ess_hmc(it)=-grad_per_ess_hmc{it}(end)+bootstrap_grad_per_ess_hmc{it}.ci975(end);
    %rm_grad_per_ess_hmc(it)=bootstrap_grad_per_ess_hmc{it}.sd(end);
    %sd_norm_grad_per_ess_multi(it)=bootstrap_grad_per_ess_multi{it}.sd(end);
end

d=[10, 100,1000,10000,100000];

figure, errorbar(d,norm_grad_per_ess_hmc,err_neg_norm_grad_per_ess_hmc,err_pos_norm_grad_per_ess_hmc,'MarkerFaceColor',[0 0.447 0.741]);%'s','MarkerFaceColor',[0 0.447 0.741])
ax=gca;
ax.XScale='log';
ax.YScale='log';
set(gcf,'position',[200 200 750 500])

fontsize(16,"points")
ylim([3,300]);
xlim([7,150000]);

hold on;
plot(d,d.^(1/4)*4,'LineWidth',1.5);
errorbar(d,norm_grad_per_ess_multi,err_neg_norm_grad_per_ess_multi, err_pos_norm_grad_per_ess_multi,'MarkerFaceColor',[0 0 0]);%'-o','MarkerFaceColor',[0 0 0])
legend('Grads/ESS vs dimension - Norm - Gaussian - RHMC', '4\cdotd^{1/4}','Grads/ESS vs dimension - Norm - UBUBU','Location',[0.33 0.7 0.3 0.2],'Fontsize',16);
title('Gradients/ESS vs dimension - Norm - Gaussian - \kappa=4')
xlabel('Dimension')
ylabel('Gradients/ESS')

end


function plot_ess_norm_comp2
    norm_grad_per_ess_hmc=zeros(1,5);
    norm_grad_per_ess_multi=zeros(1,5);
    err_neg_norm_grad_per_ess_hmc=zeros(1,5);
    err_pos_norm_grad_per_ess_hmc=zeros(1,5);
    err_neg_norm_grad_per_ess_multi=zeros(1,5);
    err_pos_norm_grad_per_ess_multi=zeros(1,5);

    %sd_norm_grad_per_ess_hmc=zeros(1,5);
    %sd_norm_grad_per_ess_multi=zeros(1,5);
    
for(it=1:5)  
    norm_grad_per_ess_multi(it)=grad_per_ess_multi{it}(end);
    norm_grad_per_ess_hmc(it)=grad_per_ess_hmc{it}(end);   
    err_neg_norm_grad_per_ess_multi(it)=grad_per_ess_multi{it}(end)-max(bootstrap_grad_per_ess_multi{it}.ci025(end),0.001);
    err_pos_norm_grad_per_ess_multi(it)=-grad_per_ess_multi{it}(end)+bootstrap_grad_per_ess_multi{it}.ci975(end);
    err_neg_norm_grad_per_ess_hmc(it)=grad_per_ess_hmc{it}(end)-bootstrap_grad_per_ess_hmc{it}.ci025(end);
    err_pos_norm_grad_per_ess_hmc(it)=-grad_per_ess_hmc{it}(end)+bootstrap_grad_per_ess_hmc{it}.ci975(end);
    %rm_grad_per_ess_hmc(it)=bootstrap_grad_per_ess_hmc{it}.sd(end);
    %sd_norm_grad_per_ess_multi(it)=bootstrap_grad_per_ess_multi{it}.sd(end);
end

d=[10, 100,1000,10000,100000];

figure, errorbar(d,norm_grad_per_ess_hmc,err_neg_norm_grad_per_ess_hmc,err_pos_norm_grad_per_ess_hmc,'MarkerFaceColor',[0 0.447 0.741]);%'s','MarkerFaceColor',[0 0.447 0.741])
ax=gca;
ax.XScale='log';
ax.YScale='log';
set(gcf,'position',[200 200 750 500])

fontsize(16,"points")
ylim([10,1600]);
xlim([7,150000]);

hold on;
plot(d,d.^(1/4)*18,'LineWidth',1.5);
errorbar(d,norm_grad_per_ess_multi,err_neg_norm_grad_per_ess_multi, err_pos_norm_grad_per_ess_multi,'MarkerFaceColor',[0 0 0]);%'-o','MarkerFaceColor',[0 0 0])
legend('Grads/ESS vs dimension - Norm - Gaussian - RHMC', '18\cdotd^{1/4}','Grads/ESS vs dimension - Norm - UBUBU','Location',[0.33 0.7 0.3 0.2],'Fontsize',16);
title('Gradients/ESS vs dimension - Norm - Gaussian - \kappa=100')
xlabel('Dimension')
ylabel('Gradients/ESS')

end
end
