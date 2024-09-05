The code reproduces the examples in the UBUBU paper with deterministic gradients.
Error estimates are also given via bootstrapping.

The methods folder contains the UBUBU and RHMC (randomized HMC) algorithms (hmcsampler.m, multilevel_ubu.m, multilevel_ubu_approx.m, multilevel_svrg.m, and their single precision variants with _single prefix in the end), as well as various other functions.
The Gaussian, MNIST, and Poisson folders contain 3 examples.

res=hmcsampler(niter,burnin,numsteps,partial,h,lpost,grad_lpost,test_function,x0)
niter is the number of interations, 
burnin is the number of burnin-steps, 
numsteps is the expected number of leapfrog steps between accept/reject steps (this is a geometric random variable in RHMC), 
partial is the partial refreshment parameter in the velocity update after each accept/reject step (i.e. v->partial * v + sqrt(1-partial^2)* Z for some standard gaussian Z).
lpost is the log-posterior function
grad_lpost is the gradient of the log-posterior function
test_function is the test function that we want to use, it is a vector valued function 
x0 is the initial position


res=multilevel_ubu(niter,burnin,rep,h,gam, grad_lpost, test_function, options)
niter is the number of interations, 
burnin is the number of burnin-steps, 
rep is the number of parallel chains at level 0
gam - gamma friction parameter of the kinetic Langevin diffusion
grad_lpost is the gradient of the log-posterior function
test_function is the test function that we want to use, it is a vector valued function 
options struct contains further options of the algorithm, such as parameters r, c, the dimension of the target distribution nbeta, the dimension of the test function test_dim, the minimizer of the log posterior beta_min


res=multilevel_ubu_approx(niter,burnin,rep,h,gam, grad_lpost, test_function, options)
niter is the number of interations, 
burnin is the number of burnin-steps, 
rep is the number of parallel chains at level 0
gam - gamma friction parameter of the kinetic Langevin diffusion
grad_lpost is the gradient of the log-posterior function
test_function is the test function that we want to use, it is a vector valued function 
options struct contains further options of the algorithm, such as parameters r, c, the dimension of the target distribution nbeta, the dimension of the test function test_dim, the minimizer of the log posterior beta_min, and repfullgrad, which corresponds to the parameter tau of the algorithm (number of iterations between each new full gradient evaluation).


res=multilevel_ubu_svrg(niter,burnin,rep,h,gam, grad_lprior, grad_llik_stoch, test_function, options)
niter is the number of interations, 
burnin is the number of burnin-steps, 
rep is the number of parallel chains at level 0
gam - gamma friction parameter of the kinetic Langevin diffusion
grad_llik_stoch is the stochastic gradient function, which is parametrized in terms of beta and an index parameter corresponding to which batch is used for evaluating the stochastic gradients
grad_lprior is the gradient of the log-prior
test_function is the test function that we want to use, it is a vector valued function 
options struct contains further options of the algorithm, such as parameters r, c, the dimension of the target distribution nbeta, the dimension of the test function test_dim, the minimizer of the log posterior beta_min, and no_batches, the number of batches (i.e. each time, 1/no_batches proportion of the dataset is used for evaluating the stochastic gradient, and the full gradient is evaluated once in evey no_batches iterations).

There are 3 examples implemented: Gaussian targets with dimensions 10,100,1000,10000,100000, MNIST, and a Poisson soccer model.
The algorithms can be run by the test_gaussian.m, test_mnist.m, and test_poisson.m files in the respective folders.
These implement the log posterior and its gradient for the examples, and then call the sampling algorithms from the methods folder (in a parallelized way, or using gpu).
The results are already included in .mat files.
The plots can be done using plot_gaussian.m, plot_mnist.m and plot_poisson.m.