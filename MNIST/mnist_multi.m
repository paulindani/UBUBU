function [res,grad_per_ess,bootstrap_grad_per_ess]=mnist_multi(niter,burnin,rep,wholerep,hrat,gamrat,bootstrapsamp)
%#codegen
coder.gpu.kernelfun;
%addpath("C:\Matlab\methods\")
%ubu_svrg2_multi_mex(2000,400,128,0.5,1000,'n',-1);

%data_type_const=@()("double");
to_data_type=@(x)(single(x));

v=load("C:\Matlab\MNIST\mnist.mat");
beta_data=load("C:\Matlab\MNIST\beta_min10.mat");


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
reg=10.0;

npred=length(test.images);  
Xpred=[ones(1,npred);reshape(test.images,28*28,npred)]';
ypred=test.labels;
Xpredp=Xpred';

beta_min=to_data_type(beta_data.beta_min);
% hess = hess_lpost(beta_min);
% hess=(hess+hess')/2;
% eH=eig(hess);
% m=min(eH)
% M=max(eH)
m=to_data_type(reg)
M=to_data_type(beta_data.M)

% grad=grad_lpost(single(beta_min));
% format long
% norm(grad)
% return;

% x=randn(nbeta,1);
% grad=grad_lpost(single(x));
% grad2=grad_lpost(double(x));
% norm(grad2-grad)
% 
% return;


h=hrat/realsqrt(M);
gam=realsqrt(m)*gamrat;

options=struct;
options.r=4;%2*sqrt(2);
options.c=1/16;
options.maxlevel=8;
options.max_parallel_chain=128;
options.beta_min=beta_min;
options.nbeta=nbeta;
options.test_dim=nbeta;
test_function=@(x)(x);

tic
res=cell(wholerep,1);
for it=1:wholerep
res{it}=multilevel_ubu_single(niter,burnin,rep,h,gam, @grad_lpost,test_function, options);
end

grad_per_ess=process_res_multi(res);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_multi);
toc
% 
% res=multilevel_ubu(niter,burnin,rep,wholerep,h,gam, @grad_lpost, test_function, options);
% grad_per_ess=process_res_multi(res);
% toc
    function J = lprior(beta)%,reg)
        J=reg/2*sum(beta.^2,1);
    end


    function J = llik(beta)%,Xp,yp,npix,n)
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



    function J = lpost(beta)%,reg)
        J=lprior(beta,reg)+llik(beta);
    end

    function grad=grad_llik(beta)%,Xp,yp,npix,n)       
    reps=size(beta,2);
    nbeta=size(beta,1);
    beta_arr=(reshape(beta,npix,reps*10))';
    % betaX=beta_arr*Xp;
    % expbetaX=exp(betaX);
    % s10=(reshape(ones(10,1)*sum(reshape(expbetaX,10,n*reps),1),reps*10,n));
    % rat=(expbetaX./s10);
    betaX=reshape(beta_arr*Xp,10,n*reps);    
    maxbetaX=ones(10,1)*max(betaX,[],1);
    expbetaX=exp(betaX-maxbetaX);
    s10=reshape(ones(10,1)*sum(expbetaX,1),reps*10,n);    
    rat=reshape(expbetaX,reps*10,n)./s10;
    %rat=reshape(softmax(betaX-maxbetaX),reps*10,n);
    %rat=reshape(softmax(betaX),reps*10,n);

    yprep=ones(reps,1)*yp;
    idx=1+10*(0:(n*reps-1))+(yprep(:))';
    rat(idx)=rat(idx)-1;
    grad=reshape(Xp*rat',nbeta,reps);
    end

    function grad = grad_lprior(beta)%,reg)
        grad=reg*beta;
    end
    
    function grad = grad_lpost(beta)%,reg,Xp,yp,npix,n)
        grad=grad_lprior(beta)+grad_llik(beta);%,Xp,yp,npix,n);
    end

    % function hess = hess_lpost(beta)
    % betaarr=(reshape(beta,npix,10))';
    % betaX=betaarr*Xp;
    % expbetaX=exp(betaX);
    % sum_expbetaX=sum(expbetaX,1);
    % hess=zeros(nbeta);
    % for(l=1:10)
    %     l
    %     hess(((l-1)*npix+1):(l*npix),((l-1)*npix+1):(l*npix))=X'*diag(expbetaX(l,:)./(sum_expbetaX))*X;
    % end
    % for(l=1:10)
    %     l
    %     for(k=1:10)
    %         hess(((l-1)*npix+1):(l*npix),((k-1)*npix+1):(k*npix))=hess(((l-1)*npix+1):(l*npix),((k-1)*npix+1):(k*npix))-X'*diag(expbetaX(l,:).*expbetaX(k,:)./(sum_expbetaX.^2))*X;
    %     end
    % end
    % hess=hess+reg*eye(nbeta);
    % end
end






