function [res,grad_per_ess,bootstrap_grad_per_ess]=mnist_hmc(niter,burnin,rep,numsteps,hrat,bootstrapsamp)
%#codegen
coder.gpu.kernelfun;


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
reg=to_data_type(10);

npred=length(test.images);  
Xpred=to_data_type([ones(1,npred);reshape(test.images,28*28,npred)]');
ypred=to_data_type(test.labels);
Xpredp=Xpred';

beta_min=to_data_type(beta_data.beta_min);
m=to_data_type(reg);
M=to_data_type(beta_data.M);


h=to_data_type(hrat/realsqrt(M));
partial=0.7;

test_function=@(x)(x);

tic
res=cell(1,rep);

for pit=1:rep
res{pit}=hmcsampler_single(niter,burnin,numsteps, partial,h,@lpost,@grad_lpost,test_function, beta_min);
%res{pit}.mean_acc
end
mean_acc=0;
for pit=1:rep
mean_acc=mean_acc+res{pit}.mean_acc;
end
mean_acc=mean_acc/rep

grad_per_ess=process_res_hmc(res);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_hmc);
toc


    function J = lprior(beta)
            J=reg/2*sum(beta.^2,1);
    end


    function J = llik(beta)
        reps=size(beta,2);
        beta_arr=(reshape(beta,npix,reps*10))';

        betaX=reshape((beta_arr)*(Xp),10,n*reps);    
        maxbetaX=ones(10,1)*max(betaX,[],1);
        betaX=betaX-maxbetaX;
        expbetaX=exp(betaX);
        %betaX=beta_arr*single(Xp);
        %expbetaX=exp(betaX);
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
    reps=size(beta,2);
    beta_arr=(reshape(beta,npix,reps*10))';
    % betaX=beta_arr*Xp;
    % expbetaX=exp(betaX);
    % s10=(reshape(ones(10,1)*sum(reshape(expbetaX,10,n*reps),1),reps*10,n));
    % rat=(expbetaX./s10);
    betaX=reshape(beta_arr*Xp,10,n*reps);    
    maxbetaX=ones(10,1)*max(betaX,[],1);
    betaX=betaX-maxbetaX;
    expbetaX=exp(betaX);
    s10=reshape(ones(10,1)*sum(expbetaX,1),reps*10,n);    
    rat=reshape(expbetaX,reps*10,n)./s10;

    yprep=ones(reps,1)*yp;
    idx=1+10*(0:(n*reps-1))+(yprep(:))';
    rat(idx)=rat(idx)-1;
    grad=reshape(Xp*rat',nbeta,reps);
    end

    function grad = grad_lprior(beta)
        grad=reg*beta;
    end
    
    function grad = grad_lpost(beta)
        grad=grad_lprior(beta)+grad_llik(beta);
    end

    function hess = hess_lpost(beta)
    betaarr=(reshape(beta,npix,10))';
    betaX=betaarr*Xp;
    expbetaX=exp(betaX);
    sum_expbetaX=sum(expbetaX,1);
    hess=zeros(nbeta);
    for(l=1:10)
            hess(((l-1)*npix+1):(l*npix),((l-1)*npix+1):(l*npix))=X'*diag(expbetaX(l,:)./(sum_expbetaX))*X;
    end
    for(l=1:10)
        for(k=1:10)
            hess(((l-1)*npix+1):(l*npix),((k-1)*npix+1):(k*npix))=hess(((l-1)*npix+1):(l*npix),((k-1)*npix+1):(k*npix))-X'*diag(expbetaX(l,:).*expbetaX(k,:)./(sum_expbetaX.^2))*X;
        end
    end
    hess=hess+reg*eye(nbeta);
    end


end






