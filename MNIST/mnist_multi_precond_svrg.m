function [res,grad_per_ess,bootstrap_grad_per_ess]=mnist_multi_precond_svrg(niter,burnin,rep,wholerep,hrat,gamrat,no_batches,c,bootstrapsamp)
%function ind=mnist_multi_precond_approx(niter,burnin,rep,wholerep,hrat,gamrat,repfullgrad,c,bootstrapsamp)
%#codegen
coder.gpu.kernelfun;
%addpath("C:\Matlab\methods\")

to_data_type=@(x)(single(x));

v=load("C:\Matlab\MNIST\mnist.mat");
beta_data=load("C:\Matlab\MNIST\beta_min10_precond.mat");

training=v.training;
test=v.test;
n=length(training.images);  
nbeta=(28*28+1)*10;
npix=28*28+1;
reg=to_data_type(10);

Xd=to_data_type([ones(1,n);reshape(training.images,28*28,n)]');
ndata=n;
X=Xd(1:ndata,:);
yd=double(training.labels);
y=yd(1:ndata);
n=ndata;
%rperm=randperm(n);
%X=X(rperm,:);
%y=y(rperm);

Xp=X';
yp=y';

Xp_by_batch=reshape(Xp, [28*28+1,n/no_batches,no_batches]);
yp_by_batch=reshape(yp,[no_batches,n/no_batches]);


% npred=1000;
% images2=test.images(:,:,1:npred);
% labels2=test.labels(1:npred);
% Xpred=to_data_type([ones(1,npred);reshape(images2,28*28,npred)]');
% ypred=to_data_type(labels2);
% Xpredp=Xpred';

m=to_data_type(beta_data.m)
M=to_data_type(beta_data.M)
Rinvmx=to_data_type(beta_data.Rinvmx);
beta_min=to_data_type(beta_data.beta_min);


h=to_data_type(hrat/realsqrt(M));
gam=to_data_type(realsqrt(m)*gamrat);

options=struct;
options.r=4;
options.c=c;
options.maxlevel=8;
options.max_parallel_chain=128;
options.no_batches=no_batches;
options.beta_min=beta_min;
options.nbeta=nbeta;
options.Hprodv=@(x)(x);
options.invcholHprodv=@(x)(x);
options.exp_hM=@(x,h)([x(1:nbeta,:)*cos(h)+x((nbeta+1):(2*nbeta),:)*sin(h);-x(1:nbeta,:)*sin(h)+x((nbeta+1):(2*nbeta),:)*cos(h)]);
options.beta_min=beta_min;
options.grad_lpost=@grad_lpost;
options.grad_llik=@grad_llik;
test_function=@(x)(x);
options.test_dim=nbeta;

tic
res=cell(wholerep,1);
for rit=1:wholerep
rit
res{rit}=multilevel_ubu_svrg_single(niter,burnin,rep,h,gam, @grad_lprior,@grad_llik_svrg,test_function, options);
end
grad_per_ess=process_res_multi(res);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_multi);
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

    function grad=grad_llik_svrg(beta,batch)
        beta=Rinvmx*beta;        
        reps=size(beta,2);
        tbeta=pagetranspose(reshape(beta,[npix,10,reps]));
        %beta_arr=(reshape(beta,npix,reps*10))';
        Xp_batch=Xp_by_batch(:,:,batch);
        betaX=pagemtimes(tbeta,Xp_batch);
        %maxbetaX=tensorprod(ones(10,1),max(betaX,[],1),2,1);
        mbetaX=max(betaX,[],1);
        maxbetaX=zeros(size(betaX),"single");
        for(it=1:10)
            maxbetaX(it,:,:)=mbetaX;
        end
        expbetaX=exp(betaX-maxbetaX);      
        %s10=tensorprod(ones(10,1),sum(expbetaX,1),2,1);
        sexpbetaX=sum(expbetaX,1);
        s10=zeros(size(betaX),"single");
        for(it=1:10)
            s10(it,:,:)=sexpbetaX;
        end

        rat=expbetaX./s10;
        yprep=yp_by_batch(batch, :);
        
        idx=1+10*(0:((n/no_batches)*reps-1))+(yprep(:))';
        rat(idx)=rat(idx)-1;
        grad=reshape(pagemtimes(Xp_batch,'none', rat,'transpose'),[nbeta,reps]);
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
 
end






