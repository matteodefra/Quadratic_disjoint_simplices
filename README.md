# Quadratic disjoint simplices

Solution quadratic problem with disjoint simplices constraints.

## Problem formulation

This package aims to solve the following quadratic problem

$$ 
\begin{array}{ll} 
\text{minimize } & x^\intercal Q x + q^\intercal x \\[1em]
\text{subject to } & \sum_{i \in I^k} x_i = \mathbf{1} \,, \forall k \in K \\[1em]
\text{} & x \ge 0
\end{array} 
$$
<!-- 
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;\text{minimize&space;}&space;&&space;x^\intercal&space;Q&space;x&space;&plus;&space;q^\intercal&space;x&space;\\&space;\text{subject&space;to&space;}&space;&&space;A&space;x&space;=&space;\mathbf{1}&space;\\&space;\text{}&space;&&space;x&space;\ge&space;0&space;\end{align*}" title="\begin{align*} \text{minimize } & x^\intercal Q x + q^\intercal x \\ \text{subject to } & A x = \mathbf{1} \\ \text{} & x \ge 0 \end{align*}" /> -->

subject to disjoint simplices constraints. 
The problem is tackled by solving the dual problem

$$
\phi(\lambda) = \max_{\lambda} \ x^\intercal Q x + q^\intercal x - \lambda^\intercal x 
$$

by using different iterative methods based on the standard subgradient method. In particular, we iteratively compute

$$
\begin{align*}
x_{t} = \arg \min_{Ax = b} \{ \ x^\intercal Q x + q^\intercal x - \lambda_t^\intercal x \ \} \\[1em]
\phi(\lambda) = \max_{\lambda} \{ \ x_t^\intercal Q x_t + q^\intercal x_t - \lambda^\intercal x_t \ \}
\end{align*}
$$

and then update $\lambda$ based on three different subgradient rules derived from the following [paper](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

$$
\begin{align*}
\lambda_t = P_\mathcal{X} \{ \lambda_t + \eta diag(G_t)^{-1/2} g_t \} \\[1em]
\lambda_t = P_\mathcal{X} \{ -H_t^{-1} t\, \eta\, \overline{g}_t \} \\[1em]
\lambda_t = P_\mathcal{X} \{ \lambda_t - \eta\, H_t^{-1}\, g_t \} \\[1em]
\end{align*}
$$

where $P_\mathcal{X}$ is the projection over the non-negative orthant $\mathcal{X} = \{ x \ge 0 \}$, $g_t$ is the subgradient, $G_t = \sum_{k=1}^t g_k g_k^\intercal$ is the full outer product of the subgradient, $H_t$ is an approximation of the hessian and $\overline{g}_t$ is the average of the subgradient until $t$ and $\eta$ is the chosen stepsize.
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
will execute the program, asking for the problem dimension $n$ and the number of disjoint simplices $K$.\
params are the following:
- stepsize: the stepsize $\eta$ that you want to adopt;
- rule: one among the three update rules for $\lambda$ described above;  
- $\alpha$: hyperparameter for stepsize 2 and 3;
- $\beta$: hyperparameter for stepsize 2 and 3;
- $\delta$: hyperparameter for rule 2 and 3;
- $h$: hyperparameter for stepsize 0 and 1;
- max_iter: maximum iterations;