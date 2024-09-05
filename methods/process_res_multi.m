function grad_per_ess=process_res_multi(res,bind,print_info)
if(nargin<=2)
print_info=false;
end
if(nargin==1)
bind=1:length(res);
end
maxlevel=res{1}.maxlevel;
maxruns=res{1}.maxruns;
wholerep=length(res);
test_dim=res{1}.test_dim;
if(isfield(res{1},"rho"))
    rho=res{1}.rho;
else
rho=1/4;
end

test_mean=zeros(test_dim,1);
test_mean_var=zeros(test_dim,1);
test_squaremean=zeros(test_dim,1);
test_var=zeros(test_dim,1);
means=zeros(test_dim, maxruns,wholerep);
squaremeans=zeros(test_dim, maxruns,wholerep);

blockstart=res{1}.blockstart;
blockend=res{1}.blockend;
rep=res{1}.rep;
repruns=zeros(maxlevel+1,wholerep);
detlevels=res{1}.detlevels;
problevels=res{1}.problevels;

for(it=1:wholerep)
repruns(1:(maxlevel+1),it)=res{bind(it)}.repruns;
means(:,:,it)=res{bind(it)}.means;
squaremeans(:,:,it)=res{bind(it)}.squaremeans;
end

test_mean=mean(means(1:test_dim,blockstart(1):blockend(1),:),[2,3]);
test_mean_var=var(means(1:test_dim,blockstart(1):blockend(1),:),0,[2,3])/rep;
test_squaremean=mean(squaremeans(1:test_dim,blockstart(1):blockend(1),:),[2,3]);
if(print_info)
    test_mean_lev_0_comp1=test_mean(1)
    max_test_mean_lev_0_var=max(test_mean_var)
end

if(maxlevel>0)

for(lev=1:(res{1}.detlevels-1))  
    if(print_info)
        lev
        max_lev_test_mean_diff=max(abs(mean(means(1:test_dim,blockstart(lev+1):blockend(lev+1),:),[2,3])))
    end
    test_mean=test_mean+mean(means(1:test_dim,blockstart(lev+1):blockend(lev+1),:),[2,3]);   
    test_mean_var=test_mean_var+var(means(1:test_dim,blockstart(lev+1):blockend(lev+1),:),0,[2,3])/repruns(lev+1,1);
    if(print_info)
        max_lev_test_mean_var=max(var(means(1:test_dim,blockstart(lev+1):blockend(lev+1),:),0,[2,3])/repruns(lev+1,1))
    end
    test_squaremean=test_squaremean+mean(squaremeans(1:test_dim,blockstart(lev+1):blockend(lev+1),:),[2,3]);
end

mean_last_arr=zeros(test_dim,wholerep);
square_mean_last=zeros(test_dim,1);

mean_last_detlevel_arr=reshape(means(1:test_dim,blockstart(detlevels+1),1:wholerep),[test_dim,wholerep]);

squaremean_last_detlevel_arr=reshape(squaremeans(1:test_dim,blockstart(detlevels+1),1:wholerep),[test_dim,wholerep]);

mean_last_arr=mean_last_detlevel_arr*(1/(1-rho));
squaremean_last_arr=squaremean_last_detlevel_arr*(1/(1-rho));

for(lev=(detlevels+1):maxlevel)
    for(wholeit=1:wholerep)
        if(repruns(lev+1,wholeit)==1)
            mean_last_arr(1:test_dim,wholeit)=mean_last_arr(1:test_dim,wholeit)...
            +(reshape(means(1:test_dim,blockstart(lev+1),wholeit),[test_dim,1])-rho^(lev-detlevels)*mean_last_detlevel_arr(1:test_dim,wholeit))/problevels(lev+1);
            squaremean_last_arr(1:test_dim,wholeit)=squaremean_last_arr(1:test_dim,wholeit)...
            +(reshape(squaremeans(1:test_dim,blockstart(lev+1),wholeit),[test_dim,1])-rho^(lev-detlevels)*squaremean_last_detlevel_arr(1:test_dim,wholeit))/problevels(lev+1);
        end
    end
end

test_mean=test_mean+mean(mean_last_arr,2);
if(print_info)
    lev=detlevels
    max_mean_diff_lev=max(abs(mean(mean_last_arr,2)))
    max_var_lev=max(var(mean_last_arr,0,2))
end
test_mean_var=test_mean_var+var(mean_last_arr,0,2);
test_squaremean=test_squaremean+mean(squaremean_last_arr,2);
end

test_post_var=test_squaremean-test_mean.^2;
ess=test_post_var./test_mean_var; %addition
grad_per_ess=(res{1}.ngradtot)*(ess).^(-1);

if(print_info)
    test_mean1=test_mean(1)
end
end