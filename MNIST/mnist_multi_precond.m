function [res,grad_per_ess,bootstrap_grad_per_ess]=mnist_multi_precond(niter,burnin,rep,wholerep,hrat,gamrat,bootstrapsamp)
%#codegen
coder.gpu.kernelfun;
%addpath("C:\Matlab\methods\")

to_data_type=@(x)(single(x));
data_type_string="single";


v=load("C:\Matlab\MNIST\mnist.mat");
%beta_data=load("C:\Matlab\MNIST\beta_min10RR.mat");
beta_data=load("C:\Matlab\MNIST\beta_min10_precond.mat");

training=v.training;
test=v.test;
n=length(training.images);  
nbeta=(28*28+1)*10;
npix=28*28+1;

Xd=to_data_type([ones(1,n);reshape(training.images,28*28,n)]');
ndata=n;
X=Xd(1:ndata,:);
n=ndata;

Xp=X';
yd=double(training.labels);
y=yd(1:ndata);
yp=y';
reg=to_data_type(10);

npred=1000;
images2=test.images(:,:,1:npred);
labels2=test.labels(1:npred);
Xpred=to_data_type([ones(1,npred);reshape(images2,28*28,npred)]');
ypred=to_data_type(labels2);
Xpredp=Xpred';

%err=norm(beta_min-beta_data.Rinvmx\beta_data0.beta_min)
m=to_data_type(beta_data.m)
M=to_data_type(beta_data.M)
Rinvmx=to_data_type(beta_data.Rinvmx);
beta_min=to_data_type(beta_data.beta_min);

pr = prob(beta_min);
ind=(pr>0.05).*(pr<0.95);
nind=sum(ind);

% grad=grad_lpost(to_data_type(beta_min));
% norm(grad)
% return;

% tic
% grad=zeros(nbeta,64);
% for(it=1:240*4)
%     grad=grad_llik(to_data_type(beta_min*rand(1,64)));
% end
% toc

%res=0;grad_per_ess=0;bootstrap_grad_per_ess=0;
%return;

    % num_eigs=1000;
    % hessian=(beta_data.hessian+beta_data.hessian')/2;   
    % V=zeros(nbeta,num_eigs);
    % D=zeros(num_eigs);
    % [V,D]=eigs(hessian,num_eigs);
    % dD=diag(D);
    % sqrtD=diag(sqrt(dD./min(dD)));
    % Rmx=zeros(nbeta,nbeta);
    % Rmx=V*(sqrtD-eye(num_eigs))*(V')+eye(nbeta);
    % 
    % Rinvmx=inv(Rmx);
    % beta_min=Rmx*beta_min;
    % hessianprecond=Rinvmx*hessian*Rinvmx';
    % eH=real(eig(hessianprecond));
    % m=to_data_type(min(eH))
    % M=to_data_type(max(eH))
    % condition_number=M/m
    % 
    % save beta_min10RR.mat beta_min Rinvmx m M;
    % grad_per_ess=0;
    % return;

% m=to_data_type(reg);
% M=to_data_type(beta_data.M);


h=to_data_type(hrat/realsqrt(M));
gam=to_data_type(realsqrt(m)*gamrat);
% if(mM=='M')
% gam=to_data_type(realsqrt(M));
% elseif(mM=='m')
% gam=to_data_type(realsqrt(m));
% else
% gam=to_data_type(realsqrt(2*m));
% end

options=struct;
options.r=4;
options.c=1/16;
options.maxlevel=8;
options.max_parallel_chain=128;
options.beta_min=beta_min;
test_function=@(x)([x;prob_ind(x,ind)]);
options.test_dim=nbeta+nind;
%test_function=@(x)(x);
%options.test_dim=nbeta;
%test_function=@(x)(vecnorm(x-beta_min*ones(1,size(x,2))));
%options.test_dim=1;

options.data_type_string=data_type_string;
options.nbeta=nbeta;

tic
res=cell(wholerep,1);
for it=1:wholerep
res{it}=multilevel_ubu_single(niter,burnin,rep,h,gam, @grad_lpost,test_function, options);
end
grad_per_ess=process_res_multi(res);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_multi);
%grad_per_ess=process_res_multi(res,1:wholerep,true);
%bootstrap_grad_per_ess=0;
toc



    function J = lprior(beta)
        J=reg/2*sum(beta.^2,1);
    end


    function J = llik(beta)
        reps=size(beta,2);
        beta_arr=(reshape(beta,npix,reps*10))';
        betaX=beta_arr*Xp;
        expbetaX=exp(betaX);
        s=(reshape(sum(reshape(expbetaX,10,n*reps),1),reps,n));
        yprep=ones(reps,1)*yp;
        idx=1+10*(0:(n*reps-1))+(yprep(:))';
        betayX=(reshape(betaX(idx),reps,n));
        J=(sum(log(s),2)-sum(betayX,2))';
    end



    function J = lpost(beta)
        J=lprior(beta)+llik(beta);
    end

    function grad=grad_llik(beta)
    beta=Rinvmx*beta;        
    reps=size(beta,2);
    beta_arr=(reshape(beta,npix,reps*10))';
    betaX=reshape(beta_arr*Xp,10,n*reps);    
    maxbetaX=ones(10,1)*max(betaX,[],1);
    expbetaX=exp(betaX-maxbetaX);
    s10=reshape(ones(10,1)*sum(expbetaX,1),reps*10,n);    
    rat=reshape(expbetaX,reps*10,n)./s10;
    yprep=ones(reps,1)*yp;
    idx=1+10*(0:(n*reps-1))+(yprep(:))';
    rat(idx)=rat(idx)-1;
    grad=reshape(Xp*rat',nbeta,reps);
    grad=Rinvmx*grad;
    end

    function grad = grad_lprior(beta)
        beta=Rinvmx*beta;            
        grad=reg*beta;
        grad=Rinvmx*grad;
    end
    
    function grad = grad_lpost(beta)
        grad=grad_lprior(beta)+grad_llik(beta);
    end

    function pr = prob(beta)       
        beta=Rinvmx*beta;
        reps=size(beta,2);
        npred=size(Xpred,1);       
        beta_arr=(reshape(beta,npix,reps*10))';
        betaX=reshape(beta_arr*Xpredp,10,npred*reps);    
        maxbetaX=ones(10,1)*max(betaX,[],1);
        expbetaX=exp(betaX-maxbetaX);
        s10=ones(10,1)*sum(expbetaX,1);
        p=reshape(expbetaX./s10,10,reps,npred);
        pr=reshape(permute(p,[1,3,2]),10*npred,reps);
    end

    function pr_ind = prob_ind(beta,ind)       
        pr_all=prob(beta);
        pr_ind=pr_all(ind==1,:);
    end
end






