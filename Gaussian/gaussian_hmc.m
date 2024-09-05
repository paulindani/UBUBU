function [res,grad_per_ess,bootstrap_grad_per_ess]=gaussian_hmc(niter,burnin,rep,numsteps,hrat,nbeta,kappa,bootstrapsamp)

myStream = RandStream('threefry4x64_20');
RandStream.setGlobalStream(myStream)

precvec=1+(0:(nbeta-1))*((kappa-1)/(nbeta-1));


m=min(precvec);
M=max(precvec);

h=hrat/realsqrt(M);
partial=1/sqrt(2);



test_function=@(x)([x;normb(x)]);
tic
res=cell(1,rep);

parfor(pit=1:rep)
x0=((precvec').^(-1/2)).*randn(nbeta,1);
res{pit}=hmcsampler(niter,burnin,numsteps, partial,h,@(x)lpost(x,precvec),@(x)grad_lpost(x,precvec),test_function, x0);
res{pit}.mean_acc
end

grad_per_ess=process_res_hmc(res);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_hmc);
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


