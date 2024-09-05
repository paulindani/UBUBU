function compile_on_gpu(script_name,pars)
envCfg = coder.gpuEnvConfig('host');
envCfg.BasicCodegen = 1;
%envCfg.Quiet = 1;
coder.checkGpuInstall(envCfg);
cfg = coder.gpuConfig('mex');
%cfg.EnableMemcpy=true;
%cfg.OptimizeReductions=true;
%cfg.SIMDAcceleration='Full';

%cfg.EnableOpenMP=true;
%cfg.GpuConfig.CompilerFlags = '--fmad=false';
codegen(script_name,"-config", cfg, "-args", pars);
end