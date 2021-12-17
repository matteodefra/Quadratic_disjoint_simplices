# Quadratic disjoint simplices

Solution quadratic problem with disjoint simplices constraints.

## Problem formulation

This package aims to solve the following quadratic problem

<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\begin{array}{ll}&space;\text{minimize&space;}&space;&&space;x^\intercal&space;Q&space;x&space;&plus;&space;q^\intercal&space;x&space;\\[1em]&space;\text{subject&space;to&space;}&space;&&space;\sum_{i&space;\in&space;I^k}&space;x_i&space;=&space;\mathbf{1}&space;\,,&space;\forall&space;k&space;\in&space;K&space;\\[1em]&space;\text{}&space;&&space;x&space;\ge&space;0&space;\end{array}" title="\begin{array}{ll} \text{minimize } & x^\intercal Q x + q^\intercal x \\[1em] \text{subject to } & \sum_{i \in I^k} x_i = \mathbf{1} \,, \forall k \in K \\[1em] \text{} & x \ge 0 \end{array}" />

subject to disjoint simplices constraints. 
The problem is tackled by solving the dual problem

<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\phi(\lambda)&space;=&space;\max_{\lambda}&space;\&space;x^\intercal&space;Q&space;x&space;&plus;&space;q^\intercal&space;x&space;-&space;\lambda^\intercal&space;x" title="\phi(\lambda) = \max_{\lambda} \ x^\intercal Q x + q^\intercal x - \lambda^\intercal x" />

by using different iterative methods based on the standard subgradient method. In particular, we iteratively compute

<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\begin{align*}&space;x_{t}&space;=&space;\arg&space;\min_{Ax&space;=&space;b}&space;\{&space;\&space;x^\intercal&space;Q&space;x&space;&plus;&space;q^\intercal&space;x&space;-&space;\lambda_t^\intercal&space;x&space;\&space;\}&space;\\[1em]&space;\phi(\lambda)&space;=&space;\max_{\lambda}&space;\{&space;\&space;x_t^\intercal&space;Q&space;x_t&space;&plus;&space;q^\intercal&space;x_t&space;-&space;\lambda^\intercal&space;x_t&space;\&space;\}&space;\end{align*}" title="\begin{align*} x_{t} = \arg \min_{Ax = b} \{ \ x^\intercal Q x + q^\intercal x - \lambda_t^\intercal x \ \} \\[1em] \phi(\lambda) = \max_{\lambda} \{ \ x_t^\intercal Q x_t + q^\intercal x_t - \lambda^\intercal x_t \ \} \end{align*}" />

and then update <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\lambda" title="\lambda" /> based on three different subgradient rules derived from the following [paper](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\begin{align*}&space;\lambda_t&space;=&space;P_\mathcal{X}&space;\{&space;\lambda_t&space;&plus;&space;\eta&space;diag(G_t)^{-1/2}&space;g_t&space;\}&space;\\[1em]&space;\lambda_t&space;=&space;P_\mathcal{X}&space;\{&space;-H_t^{-1}&space;t\,&space;\eta\,&space;\overline{g}_t&space;\}&space;\\[1em]&space;\lambda_t&space;=&space;P_\mathcal{X}&space;\{&space;\lambda_t&space;-&space;\eta\,&space;H_t^{-1}\,&space;g_t&space;\}&space;\\[1em]&space;\end{align*}" title="\begin{align*} \lambda_t = P_\mathcal{X} \{ \lambda_t + \eta diag(G_t)^{-1/2} g_t \} \\[1em] \lambda_t = P_\mathcal{X} \{ -H_t^{-1} t\, \eta\, \overline{g}_t \} \\[1em] \lambda_t = P_\mathcal{X} \{ \lambda_t - \eta\, H_t^{-1}\, g_t \} \\[1em] \end{align*}" />

where <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;P_\mathcal{X}" title="P_\mathcal{X}" /> is the projection over the non-negative orthant with \
<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\mathcal{X}&space;=&space;\{&space;x&space;\ge&space;0&space;\}" title="\mathcal{X} = \{ x \ge 0 \}" />  \
<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;g_t" title="g_t" /> is the subgradient \
<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;G_t&space;=&space;\sum_{k=1}^t&space;g_k&space;g_k^\intercal" title="G_t = \sum_{k=1}^t g_k g_k^\intercal" /> \
is the full outer product of the subgradient \
<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;H_t" title="H_t" /> is an approximation of the hessian \
<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\overline{g}_t" title="\overline{g}_t" /> is the average of the subgradient until <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;t" title="t" /> and finally \
<img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\eta" title="\eta" /> is the chosen stepsize. \
To check the derived rules and the theoretical analysis, check [report](report/report.pdf)


## Execution

The code is entirely contained in a single script, to reduce allocation and execution time of Julia. 
To test the code, clone this repository, open your Julia REPL and run 
```
    ]
```
to enter in pkg mode
```
    instantiate
```
to download modules declared in [Manifest.toml](Manifest.toml). Then
```
    include(src/main.jl)
```
to compile the code in the REPL.
```
    initialize(params)
```
will execute the program, asking for the problem dimension <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;n" title="n" /> and the number of disjoint simplices <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;K" title="K" />.\
params are the following:
- stepsize: the stepsize <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\eta" title="\eta" /> that you want to adopt;
- rule: one among the three update rules for <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\lambda" title="\lambda" /> described above;  
- <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\alpha" title="\alpha" />: hyperparameter for stepsize 2 and 3;
- <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\beta" title="\beta" />: hyperparameter for stepsize 2 and 3;
- <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\delta" title="\delta" />: hyperparameter for rule 2 and 3;
- h: hyperparameter for stepsize 0 and 1;
- max_iter: maximum iterations;


## Results

Below are two examples of results obtained with _constant length_ and _constant step_ stepsizes, using the different update rules of <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\lambda" title="\lambda" /> on a problem of size <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;n" title="n" /> 5000 and number of constraints <img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;K" title="K" /> 2500

![alt-text-1](images/n5000_K2500_step0.png "Step 0") ![alt-text-2](images/n5000_K2500_step1.png "Step 1")
