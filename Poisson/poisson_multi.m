  function [res,grad_per_ess,bootstrap_grad_per_ess]=poisson_multi(niter,burnin,rep,wholerep,hrat,gamrat,bootstrapsamp)

%addpath("C:\Matlab\methods\")


beta_data=load("C:\Matlab\Poisson\beta_min.mat");
reg=beta_data.reg;
reg0=beta_data.reg0;
G=beta_data.G;
beta_min=beta_data.beta_min;
X=beta_data.X;
y=beta_data.y;
numteams=beta_data.numteams;
weeks=beta_data.weeks;
nbeta=2*numteams*weeks;

ind_attack=sub2ind([numteams,weeks],X.attacker,X.week);
ind_defense=sub2ind([numteams,weeks],X.defender,X.week);
attack_map=zeros(numteams*weeks,3);
defense_map=zeros(numteams*weeks,3);
for(it=1:(2*G))
    X_ind=X.attacker(it)+numteams*(X.week(it)-1);
    if(attack_map(X_ind,1)==0)
        attack_map(X_ind,1)=it;
    elseif(attack_map(X_ind,2)==0)
        attack_map(X_ind,2)=it;
    else
        attack_map(X_ind,3)=it;
    end
end
for(it=1:(2*G))
    X_ind=X.defender(it)+numteams*(X.week(it)-1);
    if(defense_map(X_ind,1)==0)
        defense_map(X_ind,1)=it;
    elseif(defense_map(X_ind,2)==0)
        defense_map(X_ind,2)=it;
    else
        defense_map(X_ind,3)=it;
    end
end
attack_map=attack_map+(attack_map==0)*(2*G+1);
defense_map=defense_map+(defense_map==0)*(2*G+1);

Hprior=zeros(weeks,weeks);
for(Hit=1:(weeks-1))
    Hprior(Hit:(Hit+1),Hit:(Hit+1))=Hprior(Hit:(Hit+1),Hit:(Hit+1))+[1,-1;-1,1];
end

pars=struct;
pars.Hprior=Hprior;
pars.numteams=numteams;
pars.weeks=weeks;
pars.reg0=reg0;
pars.reg=reg;
pars.y=y;
pars.ind_defense=ind_defense;
pars.ind_attack=ind_attack;
pars.attack_map=attack_map;
pars.defense_map=defense_map;
pars.nbeta=nbeta;
pars.G=G;

m=reg0
hessian_prior=hess_lprior(beta_min,pars);
pars.hessian_prior=hessian_prior;
hessian_llik=hess_llik(beta_min,pars);
hessian=hessian_prior+hessian_llik;

M=normest_new(hessian)


h=hrat/realsqrt(M);

gam=realsqrt(m)*gamrat;

n=2*G;



options=struct;
options.r=4;
options.c=1/16;
options.maxlevel=8;
options.max_parallel_chain=16;
options.beta_min=beta_min;
options.test_dim=nbeta;
options.nbeta=nbeta;
test_function=@(x)(x);




tic
res=cell(wholerep,1);
parfor it=1:wholerep
    res{it}=multilevel_ubu(niter,burnin,rep,h,gam, @(x)(grad_lpost(x,pars)),test_function, options);
end
grad_per_ess=process_res_multi(res,1:wholerep,true);
bootstrap_grad_per_ess=bootstrap_res(res,bootstrapsamp,@process_res_multi);
toc

end



    function res=pospart(x)
        res=x.*(x>0);
    end
    function res=softplus(x)
        xl0=(x<0);
        res=(xl0).*log(1+exp(-pospart(-x)))+(1-xl0).*(x+log(1+exp(-pospart(x))));
    end

    function res=softplusder(x)
        res=1./(1+exp(-x));
    end
    function res=softplusder2(x)
        res=1./(2+exp(-x)+exp(x));
    end

    function J = lprior(beta,pars)
        numteams=pars.numteams;
        weeks=pars.weeks;
        reg0=pars.reg0;

        J=reg0/2*sum(beta.^2,1);
        reps=size(beta,2);
        attack=reshape(beta(1:(numteams*weeks),1:reps),[numteams,weeks,reps]);
        defense=reshape(beta((numteams*weeks+1):(2*numteams*weeks),1:reps),[numteams,weeks,reps]);
        J=J+reshape(reg/2*(sum((attack(:,2:weeks,:)-attack(:,1:(weeks-1),:)).^2,[1, 2])+sum((defense(:,2:weeks,:)-defense(:,1:(weeks-1),:)).^2,[1, 2])),[1,reps]);
    end

    function grad = grad_lprior(beta,pars)
        grad=pars.hessian_prior*beta;
        % reps=size(beta,2);
        % attack=reshape(beta(1:(numteams*weeks),1:reps),[numteams,weeks,reps]);
        % defense=reshape(beta((numteams*weeks+1):(2*numteams*weeks),1:reps),[numteams,weeks,reps]);
        % 
        % grad_attack=pagemtimes(attack,Hprior);
        % grad_defense=pagemtimes(defense,Hprior);
        % grad=reg0*beta;
        % grad(1:(numteams*weeks),1:reps)=grad(1:(numteams*weeks),1:reps)+reg*reshape(grad_attack,[numteams*weeks,reps]);
        % grad((numteams*weeks+1):(numteams*weeks*2),1:reps)=grad((numteams*weeks+1):(numteams*weeks*2),1:reps)+reg*reshape(grad_defense,[numteams*weeks,reps]);
    end
    

    function J = llik(beta,pars)
        reps=size(beta,2);
        numteams=pars.numteams;
        weeks=pars.weeks;
        reg0=pars.reg0;
        y=pars.y;

        attack=beta(1:(numteams*weeks),1:reps);
        defense=beta((numteams*weeks+1):(2*numteams*weeks),1:reps);
        xvar=attack(ind_attack,1:reps)+defense(ind_defense,1:reps);
        lambda=softplus(xvar);
        J=sum(-(y*ones(1,reps)).*log(lambda(1:(2*G),1))+lambda(1:(2*G),1:reps),1);
    end

    function grad=grad_llik(beta,pars)
        numteams=pars.numteams;
        weeks=pars.weeks;
        y=pars.y;
        ind_defense=pars.ind_defense;
        ind_attack=pars.ind_attack;
        attack_map=pars.attack_map;
        defense_map=pars.defense_map;

        reps=size(beta,2);
        attack=beta(1:(numteams*weeks),1:reps);
        defense=beta((numteams*weeks+1):(2*numteams*weeks),1:reps);
        xvar=attack(ind_attack,1:reps)+defense(ind_defense,1:reps);
        lambda=softplus(xvar);

        grad_terms=[(-(y*ones(1,reps))./lambda+1).*softplusder(xvar);zeros(1,reps)];
        grad2_attack=reshape(grad_terms(attack_map(:),1:reps),numteams*weeks,3,reps);
        grad2_defense=reshape(grad_terms(defense_map(:),1:reps),numteams*weeks,3,reps);
        
        %grad=[sum(grad2_attack,2);sum(grad2_defense,2)];
        s_attack=squeeze(sum(grad2_attack,2));
        s_defense=squeeze(sum(grad2_defense,2));

        grad=[s_attack;s_defense];

    end

    function hess=hess_lprior(beta,pars)

        numteams=pars.numteams;
        weeks=pars.weeks;
        reg0=pars.reg0;
        nbeta=pars.nbeta;
        Hprior=pars.Hprior;
        reg=pars.reg;

        hess=reg0*speye(nbeta);

        for(teamit=1:(numteams))
            hess(teamit:numteams:(numteams*weeks),teamit:numteams:(numteams*weeks))=...
            hess(teamit:numteams:(numteams*weeks),teamit:numteams:(numteams*weeks))+reg*Hprior;

            hess((numteams*weeks+teamit):numteams:(2*numteams*weeks),(numteams*weeks+teamit):numteams:(2*numteams*weeks))=...
            hess((numteams*weeks+teamit):numteams:(2*numteams*weeks),(numteams*weeks+teamit):numteams:(2*numteams*weeks))+reg*Hprior;

            %hess(((teamit-1)*weeks+1):(teamit*weeks),((teamit-1)*weeks+1):(teamit*weeks))=...
            %hess(((teamit-1)*weeks+1):(teamit*weeks),((teamit-1)*weeks+1):(teamit*weeks))+reg*Hprior;
        end
    end

    function hess=hess_llik(beta,pars)
        numteams=pars.numteams;
        weeks=pars.weeks;
        y=pars.y;
        nbeta=pars.nbeta;
        ind_defense=pars.ind_defense;
        ind_attack=pars.ind_attack;
        G=pars.G;

        hess=sparse(nbeta,nbeta);

        attack=beta(1:(numteams*weeks));
        defense=beta((numteams*weeks+1):(2*numteams*weeks));
        xvar=attack(ind_attack)+defense(ind_defense);
        lambda=softplus(xvar);

        %hess_terms=[(-y./lambda+1).*softplusder2(xvar)+(y./(lambda.^2)).*((softplusder(xvar)).^2);0];
        hess_terms=(-y./lambda+1).*softplusder2(xvar)+(y./(lambda.^2)).*((softplusder(xvar)).^2);
        %hess2_attack=reshape(hess_terms(attack_map(:)),numteams*weeks,3);
        %hess2_defense=reshape(hess_terms(defense_map(:)),numteams*weeks,3);
        
        %s_attack=sum(hess2_attack,2);
        %s_defense=sum(hess2_defense,2);
        ind_defense2=numteams*weeks+ind_defense;
        for(git=1:(2*G))
            hess(ind_attack(git),ind_attack(git))=hess(ind_attack(git),ind_attack(git))+hess_terms(git);
            hess(ind_attack(git),ind_defense2(git))=hess(ind_attack(git),ind_defense2(git))+hess_terms(git);
            hess(ind_defense2(git),ind_attack(git))=hess(ind_defense2(git),ind_attack(git))+hess_terms(git);           
            hess(ind_defense2(git),ind_defense2(git))=hess(ind_defense2(git),ind_defense2(git))+hess_terms(git);                   
        end
        
    end

    function hess=hess_lpost(beta,pars)
        hess=hess_lprior(beta,pars)+hess_llik(beta,pars);
    end

    % function J = lpost(beta)
    %     J=lprior(beta)+llik(beta);
    % end



    function grad = grad_lpost(beta,pars)
        grad=grad_lprior(beta,pars)+grad_llik(beta,pars);
    end







