function [res,grad_per_ess,bootstrap_grad_per_ess]=gaussian_multi(niter,burnin,rep,wholerep,hrat,gamrat,nbeta,kappa,bootstrapsamp)

tic
to_data_type=@(x)(double(x));

precvec=to_data_type(1+(0:(nbeta-1))*((kappa-1)/(nbeta-1)));

beta_min=zeros(nbeta,1);
m=min(precvec);
M=max(precvec);

h=hrat/realsqrt(M);
gam=realsqrt(m)*gamrat;

options=struct;
options.r=4;
options.c=1/16;
options.maxlevel=8;
options.max_parallel_chain=16;
options.beta_min=to_data_type(beta_min);
options.nbeta=nbeta;
test_function=@(x)([x;normb(x)]);
options.test_dim=length(test_function(beta_min));

res=cell(wholerep,1);
parfor it=1:wholerep
res{it}=multilevel_ubu(niter,burnin,rep,h,gam, @(x)grad_lpost(x,precvec),test_function, options);
end

grad_per_ess=process_res_multi(res);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_multi);
toc

end

function normbeta = normb(beta)
            normbeta=realsqrt(sum((beta).^2,1));
end

function J = lprior(beta)
    J=zeros(1,size(beta,2));
end

function J = llik(beta,precvec)
    J=precvec*(beta.^2)/2;
end


function J = lpost(beta,precvec)
    J=lprior(beta)+llik(beta,precvec);
end

function grad=grad_llik(beta,precvec)
    reps=size(beta,2);
    grad=(precvec'*ones(1,reps)).*beta;
end

function grad = grad_lprior(beta)
    grad=0*beta;
end

function grad = grad_lpost(beta,precvec)
    grad=grad_lprior(beta)+grad_llik(beta,precvec);
end



function hess = hessll(beta,precvec)
    hess=spdiags(precvec,0,nbeta,nbeta);
end

function [J, grad]=lpostwithgrad(beta,precvec)
    J = lpost(beta,precvec);
    grad = grad_lpost(beta,precvec);
end









