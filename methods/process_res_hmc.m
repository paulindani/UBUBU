function grad_per_ess=process_res_hmc(res,bind)
%nbeta=res.nbeta;
if(nargin<2)
    bind=1:length(res);
end
rep=length(res);
test_dim=res{1}.test_dim;
means=zeros(test_dim,rep);
squaremeans=zeros(test_dim,rep);
for it=1:rep
    means(:,it)=res{bind(it)}.means;   
    squaremeans(:,it)=res{bind(it)}.squaremeans;   
end
test_mean=mean(means,2);
test_mean_var=var(means,0,2);

test_squaremean=mean(squaremeans,2);
test_post_var=test_squaremean-test_mean.^2;
ess=test_post_var./test_mean_var;

grad_per_ess=(res{1}.ngradtot)*(ess).^(-1);
end