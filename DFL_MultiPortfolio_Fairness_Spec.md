# DFL on Multi-Portfolio Optimization with Fairness: Complete Implementation Specification

## 1. Background: What Is the Multi-Portfolio Problem?

### The Setting
A financial adviser manages **n client accounts** (portfolios), each investing in the same pool of **m risky assets**. Each account has:
- Initial holdings **w_i** ∈ ℝ^m (in currency units)
- A utility function **u_i(x_i)** measuring benefit from rebalancing trades x_i
- A feasible trade set **C_i** (e.g., budget, turnover, risk constraints)

### Why Is It Different From Single Portfolio?
When the manager executes trades, all accounts' orders are **aggregated** into a single market order. This creates **market impact costs** — the price moves against you because your own trading activity reduces liquidity. The key insight: each account's transaction cost depends on what **all other accounts** are trading.

**Example:** If Account A buys $1M of AAPL and Account B also buys $500K of AAPL, the combined $1.5M order moves the price more than two separate smaller orders would. Both accounts pay higher costs because of each other.

### The Three Key Challenges
1. **Cost Splitting:** How to divide the shared market impact costs fairly among accounts
2. **Multi-Objective:** Each account's net utility is a separate objective to optimize
3. **Coordination Benefits:** Joint optimization should benefit everyone vs. independent optimization

---

## 2. Mathematical Formulation (From Iancu & Trichakis 2014)

### Notation
| Symbol | Meaning |
|--------|---------|
| n | Number of accounts (portfolios) |
| m | Number of risky assets |
| **w_i** ∈ ℝ^m | Initial holdings of account i (in currency) |
| **x_i** ∈ ℝ^m | Rebalancing trades for account i |
| **x_i^+** = max(x_i, 0) | Buy orders for account i |
| **x_i^-** = max(-x_i, 0) | Sell orders for account i |
| u_i(x_i) | Utility of account i from trades (concave, in currency) |
| t_j(x^+, x^-) | Market impact cost for asset j (convex, increasing) |
| τ_ij | Cost charged to account i for trading asset j |
| τ_i = Σ_j τ_ij | Total cost charged to account i |
| U_i = u_i(x_i) − τ_i | Net utility of account i |
| f(U_1,...,U_n) | Welfare function (concave, increasing) |

### Utility Functions (Common Choices)
- **Expected profit:** u_i(x_i) = μ^T x_i
- **Risk-adjusted:** u_i(x_i) = μ^T x_i − λ_i (w_i + x_i)^T Σ (w_i + x_i)

where μ ∈ ℝ^m is expected return, Σ is covariance matrix, λ_i ≥ 0 is risk aversion.

### Market Impact Cost Function
The paper uses a **quadratic, separable** model:

t_j(x^+, x^-) = α_j ((x^+)^2 + (x^-)^2)

where α_j > 0 is the impact coefficient for asset j. Total cost for all assets:

t(Σ_i x_i^+, Σ_i x_i^-) = Σ_j α_j ((Σ_i x_{ij}^+)^2 + (Σ_i x_{ij}^-)^2)

### Feasible Trade Set (from Numerical Study 1)
```
C_i = { x_i ∈ ℝ^m : 
    1^T x_i = 0,                                    # self-financing
    ||x_i||_1 ≤ 10% · 1^T w_i,                      # turnover constraint
    (w_i + x_i)^T Σ (w_i + x_i) ≤ (σ_i · 1^T w_i)^2  # risk constraint
}
```

### The Independent Solution (Baseline)
Each account optimizes in isolation, ignoring others:

```
For each i: solve max_{x_i ∈ C_i} { u_i(x_i) − t(x_i^+, x_i^-) }
```

Then aggregate trades and split costs pro rata. The realized utility is:

U_i^IND = u_i(x_i^IND) − Σ_j [x_{ij}^IND / Σ_a x_{aj}^IND] · t_j(Σ_a (x_{aj}^IND)^+, Σ_a (x_{aj}^IND)^-)

**Problem:** This underestimates true costs and is NOT Pareto optimal.

### The MPO Formulation (Problem 15 in paper)

```
maximize  f(u_1(x_1) − τ_1, ..., u_n(x_n) − τ_n)

subject to:
    x_i ∈ C_i,                              ∀i ∈ {1,...,n}
    x_i = x_i^+ − x_i^-,                   ∀i
    x_i^+, x_i^- ≥ 0,                      ∀i
    τ_i = Σ_j τ_ij,                        ∀i
    
    # Cost lower bound: at least standalone cost
    t_j(x_{ij}^+, x_{ij}^-) ≤ τ_ij,       ∀i, j          (15a)
    
    # Cost upper bound: at most externality 
    # (reformulated for convexity)
    t_j(Σ_{a≠i} x_{aj}^+, Σ_{a≠i} x_{aj}^-) ≤ Σ_{a≠i} τ_{aj},   ∀i, j   (15b)
    
    # Budget balance: total charges = total costs
    t_j(Σ_a x_{aj}^+, Σ_a x_{aj}^-) ≤ Σ_a τ_{aj},   ∀j          (15c)
    
    # Coordination benefit: at least as good as independent
    u_i(x_i) − τ_i ≥ U_i^IND,             ∀i                    (15d)
```

### Welfare Function: α-Fairness
The paper focuses on **maximin** (Rawlsian fairness):

f(U_1, ..., U_n) = min_i { (U_i − U_i^IND) / U_i^IND }

This maximizes the minimum relative improvement across all accounts.

More generally, **α-fairness** is:

f_α(U_1, ..., U_n) = Σ_i U_i^{1-α} / (1-α)    for α ≥ 0, α ≠ 1
f_1(U_1, ..., U_n) = Σ_i log(U_i)               for α = 1 (Nash bargaining / proportional fairness)

- α = 0: Utilitarian (social welfare, sum of utilities)
- α → 1: Proportional fairness (Nash bargaining)  
- α → ∞: Maximin (Rawlsian)

### About Closed-Form Solutions

**No closed-form exists** for the full MPO problem because:
1. The market impact cost t_j couples all accounts through (Σ_i x_{ij}^+)^2
2. The cost split variables τ_ij add n×m additional decision variables with cross-account constraints
3. The coordination benefit constraints (15d) require solving the independent problem first

**However,** the problem IS convex (for concave u_i, convex t, concave f), so:
- cvxpy can solve it efficiently
- cvxpylayers can differentiate through it for DFL
- Typical size: O(nm) variables, O(nm) constraints → tractable for n ≤ ~50, m ≤ ~500

---

## 3. DFL Formulation: End-to-End Learning for MPO

### The Predict-Then-Optimize Pipeline

```
Features z_t → Prediction Model g_θ(z_t) → Predicted Parameters (μ̂, etc.)
    → MPO Solver (Problem 15) → Optimal Trades x*(μ̂), Cost Split τ*(μ̂)
    → Evaluate on True Parameters → Decision Loss
```

### What We Predict
The prediction model forecasts **expected returns μ** from features:

μ̂ = g_θ(z_t)

where z_t ∈ ℝ^{m×p} are asset features (momentum, value, volatility, etc.) and θ are model parameters.

**What we assume known (not predicted):**
- Covariance matrix Σ (estimated from historical data, held fixed)
- Market impact coefficients α_j (calibrated from market data, held fixed)
- Risk aversion parameters λ_i (given by clients)
- Initial holdings w_i (known)

### DFL Training Loss

**Two-stage (baseline):** Train g_θ to minimize MSE on μ:
```
L_MSE(θ) = ||μ̂ − μ_true||^2
```

**Decision-focused:** Train g_θ to minimize decision regret:
```
L_DFL(θ) = f(U_1*(μ_true), ..., U_n*(μ_true)) − f(U_1*(μ̂), ..., U_n*(μ̂))
```

where U_i*(μ) = u_i(x_i*(μ); μ_true) − τ_i*(μ) is the **true** net utility when using trades optimized under predicted μ̂ but evaluated under true μ.

### Gradient Flow

```
∂L_DFL/∂θ = −(∂f/∂U) · (∂U/∂x, ∂U/∂τ) · (∂x*/∂μ̂, ∂τ*/∂μ̂) · ∂μ̂/∂θ
```

The key piece is **(∂x*/∂μ̂, ∂τ*/∂μ̂)** — the Jacobian of the optimal solution with respect to predicted parameters. This is computed by **differentiating through the KKT conditions** of the convex MPO problem, which cvxpylayers does automatically.

---

## 4. Simplified Formulation for Implementation

For a clean first implementation, I recommend simplifying Problem (15) while keeping the core multi-portfolio + fairness structure.

### Simplified MPO (Recommended Starting Point)

Use the **compact cross-asset formulation** (Problem 19 in paper, Theorem 2 shows equivalence):

```
maximize  f_α(u_1(x_1) − τ_1, ..., u_n(x_n) − τ_n)

subject to:
    x_i ∈ C_i,                                    ∀i
    x_i = x_i^+ − x_i^-,                          ∀i
    x_i^+, x_i^- ≥ 0,                             ∀i
    
    # Standalone cost lower bound
    t(x_i^+, x_i^-) ≤ τ_i,                        ∀i
    
    # Externality upper bound (reformulated)
    t(Σ_{a≠i} x_a^+, Σ_{a≠i} x_a^-) ≤ Σ_{a≠i} τ_a,   ∀i
    
    # Total budget balance
    t(Σ_a x_a^+, Σ_a x_a^-) ≤ Σ_a τ_a,
    
    # Coordination benefit
    u_i(x_i) − τ_i ≥ U_i^IND,                     ∀i
```

This has O(n) cost split variables instead of O(nm).

### Further Simplification: Drop Coordination Constraints Initially

For initial experiments, you can drop constraint (15d) — the coordination benefit constraint — because:
1. It requires solving the independent problem first (nested optimization)
2. α-fairness with appropriate α naturally prevents severe unfairness
3. You can add it back later as a refinement

### Concrete Parameter Choices (from Numerical Study 1)

```python
# Problem dimensions
n_accounts = 3       # or 6 for Study 2
m_assets = 100       # number of assets
n_factors = 20       # factor model

# Market impact: t_j(x+, x-) = α_j * ((x+)^2 + (x-)^2)
alpha_j ~ Uniform(2, 10)   # impact coefficients

# Expected returns: μ_j ~ Uniform(-0.20, 0.40)  (annualized)
# Volatilities: σ_j ~ Uniform(0.15, 0.45) (annualized)

# Account properties:
# - Account volatility targets: σ_1=5%, σ_2=10%, σ_3=20%
# - Account 2 is twice as large as accounts 1 and 3
# - Risk aversion (Study 2): λ_i ~ Uniform(1e-4, 2.5e-4)

# Constraints:
# - Self-financing: 1^T x_i = 0
# - Turnover: ||x_i||_1 ≤ 0.10 * (1^T w_i)
# - Risk: (w_i + x_i)^T Σ (w_i + x_i) ≤ (σ_i * 1^T w_i)^2
```

---

## 5. Complete Implementation Plan

### Step 0: Data Generation

Since the paper uses synthetic data, we generate it similarly:

```python
"""
Data generation for MPO experiments.
Follows Numerical Study 1 from Iancu & Trichakis (2014).
"""
import numpy as np

def generate_mpo_instance(n_accounts=3, m_assets=100, n_factors=20, seed=42):
    rng = np.random.RandomState(seed)
    
    # Factor model: r_j = μ_j + a_j^T f + ε_j
    # Factor loadings
    A = rng.randn(m_assets, n_factors) * 0.1  # exposure coefficients
    
    # Expected returns (annualized)
    mu = rng.uniform(-0.20, 0.40, m_assets)
    
    # Idiosyncratic volatility
    sigma_idio = rng.uniform(0.05, 0.20, m_assets)
    
    # Covariance matrix: Σ = A A^T + diag(σ_idio^2)
    Sigma = A @ A.T + np.diag(sigma_idio**2)
    
    # Ensure annualized volatilities are in [15%, 45%]
    # Scale A and sigma_idio accordingly (simplified)
    vols = np.sqrt(np.diag(Sigma))
    target_vols = rng.uniform(0.15, 0.45, m_assets)
    scale = target_vols / vols
    Sigma = np.outer(scale, scale) * Sigma
    
    # Market impact coefficients
    alpha = rng.uniform(2, 10, m_assets)
    
    # Initial holdings: random index-tracking portfolios
    # with target volatilities σ_1=5%, σ_2=10%, σ_3=20%
    target_account_vols = [0.05, 0.10, 0.20]
    account_sizes = [1.0, 2.0, 1.0]  # Account 2 is twice as large
    
    W = []  # list of w_i vectors
    for i in range(n_accounts):
        # Generate random portfolio weights (positive, sum to 1)
        weights = rng.dirichlet(np.ones(m_assets))
        # Scale to match target volatility (approximately)
        port_vol = np.sqrt(weights @ Sigma @ weights)
        weights = weights * (target_account_vols[i % len(target_account_vols)] / port_vol)
        # Normalize
        weights = weights / weights.sum()
        # Convert to currency units
        w_i = weights * account_sizes[i % len(account_sizes)] * 1e6  # $1M base
        W.append(w_i)
    
    return {
        'mu': mu,             # (m,) expected returns
        'Sigma': Sigma,       # (m, m) covariance matrix
        'alpha': alpha,       # (m,) impact coefficients
        'W': W,               # list of n (m,) initial holdings
        'n_accounts': n_accounts,
        'm_assets': m_assets,
    }
```

### Step 1: Solve the Independent Problem (Baseline)

```python
"""
Independent solution: optimize each account in isolation.
"""
import cvxpy as cp
import numpy as np

def solve_independent(mu, Sigma, alpha, w_i, sigma_i, lambda_i=0.0):
    """
    Solve single-account problem:
        max  μ^T x_i - λ_i (w_i+x_i)^T Σ (w_i+x_i) - Σ_j α_j ((x_ij^+)^2 + (x_ij^-)^2)
        s.t. 1^T x_i = 0 (self-financing)
             ||x_i||_1 ≤ 0.10 * (1^T w_i) (turnover)
             (w_i+x_i)^T Σ (w_i+x_i) ≤ (σ_i * 1^T w_i)^2 (risk)
    
    Returns: x_i_opt, U_i_IND (utility under own cost only)
    """
    m = len(mu)
    total_wealth = np.sum(w_i)
    
    x = cp.Variable(m)
    x_plus = cp.Variable(m, nonneg=True)
    x_minus = cp.Variable(m, nonneg=True)
    
    # Utility: expected profit (or risk-adjusted)
    if lambda_i > 0:
        utility = mu @ x - lambda_i * cp.quad_form(w_i + x, Sigma)
    else:
        utility = mu @ x
    
    # Own transaction cost (trading alone)
    own_cost = cp.sum(cp.multiply(alpha, cp.square(x_plus) + cp.square(x_minus)))
    
    objective = cp.Maximize(utility - own_cost)
    
    constraints = [
        x == x_plus - x_minus,
        cp.sum(x) == 0,                                    # self-financing
        cp.norm(x, 1) <= 0.10 * total_wealth,              # turnover
        cp.quad_form(w_i + x, Sigma) <= (sigma_i * total_wealth)**2,  # risk
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, warm_start=True)
    
    return x.value, prob.value


def compute_independent_realized_utilities(mu, Sigma, alpha, W, sigmas, lambdas):
    """
    1. Solve each account independently
    2. Aggregate trades
    3. Compute realized utilities with pro rata cost split
    
    Returns: list of x_i^IND, list of U_i^IND (realized)
    """
    n = len(W)
    m = len(mu)
    
    X_ind = []
    for i in range(n):
        x_i, _ = solve_independent(mu, Sigma, alpha, W[i], sigmas[i], lambdas[i])
        X_ind.append(x_i)
    
    # Compute realized costs with aggregated trades
    X_ind = np.array(X_ind)  # (n, m)
    
    # Aggregate buy/sell
    X_plus = np.maximum(X_ind, 0)
    X_minus = np.maximum(-X_ind, 0)
    agg_buy = X_plus.sum(axis=0)     # (m,)
    agg_sell = X_minus.sum(axis=0)   # (m,)
    
    # Total market impact per asset
    total_cost_per_asset = alpha * (agg_buy**2 + agg_sell**2)  # (m,)
    
    # Pro rata split
    U_ind_realized = []
    for i in range(n):
        # utility
        if lambdas[i] > 0:
            u_i = mu @ X_ind[i] - lambdas[i] * (W[i] + X_ind[i]) @ Sigma @ (W[i] + X_ind[i])
        else:
            u_i = mu @ X_ind[i]
        
        # pro rata cost
        cost_i = 0.0
        for j in range(m):
            total_trade_j = np.abs(X_ind[:, j]).sum()
            if total_trade_j > 1e-10:
                cost_i += (np.abs(X_ind[i, j]) / total_trade_j) * total_cost_per_asset[j]
        
        U_ind_realized.append(u_i - cost_i)
    
    return X_ind, U_ind_realized
```

### Step 2: Solve the MPO Problem (with α-fairness)

```python
"""
MPO solver using compact formulation (Problem 19).
This is the core optimization that DFL will differentiate through.
"""
import cvxpy as cp
import numpy as np

def solve_mpo(mu, Sigma, alpha, W, sigmas, lambdas, 
              fairness_alpha=None, U_ind=None,
              use_coordination_constraint=False):
    """
    Solve the MPO problem (compact formulation, Problem 19).
    
    Args:
        mu: (m,) expected returns (THIS IS THE PREDICTED PARAMETER)
        Sigma: (m, m) covariance
        alpha: (m,) market impact coefficients
        W: list of n arrays, each (m,) initial holdings
        sigmas: list of n floats, target volatilities
        lambdas: list of n floats, risk aversion
        fairness_alpha: float, α-fairness parameter
            0 = utilitarian, 1 = proportional fairness, large = maximin
        U_ind: list of n floats, independent solution utilities (for coordination)
        use_coordination_constraint: bool
    
    Returns: trades X, cost splits tau, objective value
    """
    n = len(W)
    m = len(mu)
    
    # Decision variables
    X_plus = [cp.Variable(m, nonneg=True) for _ in range(n)]
    X_minus = [cp.Variable(m, nonneg=True) for _ in range(n)]
    X = [X_plus[i] - X_minus[i] for i in range(n)]
    tau = cp.Variable(n)  # total cost charged to each account
    
    # Utilities
    utilities = []
    for i in range(n):
        if lambdas[i] > 0:
            u_i = mu @ X[i] - lambdas[i] * cp.quad_form(W[i] + X[i], Sigma)
        else:
            u_i = mu @ X[i]
        utilities.append(u_i)
    
    # Net utilities: U_i = u_i - τ_i
    net_utilities = [utilities[i] - tau[i] for i in range(n)]
    
    # Market impact cost function (vectorized)
    def market_impact(x_plus_list, x_minus_list):
        """Total market impact cost = Σ_j α_j * ((Σ_i x_{ij}^+)^2 + (Σ_i x_{ij}^-)^2)"""
        agg_buy = sum(x_plus_list)   # (m,)
        agg_sell = sum(x_minus_list)  # (m,)
        return cp.sum(cp.multiply(alpha, cp.square(agg_buy) + cp.square(agg_sell)))
    
    # === Welfare function ===
    if fairness_alpha is None or fairness_alpha == 0:
        # Utilitarian: maximize sum of net utilities
        welfare = sum(net_utilities)
    elif fairness_alpha >= 50:
        # Maximin (approximate α→∞)
        # Use relative improvement if U_ind provided
        if U_ind is not None:
            min_improvement = cp.Variable()
            welfare = min_improvement
            # Will add constraints below
        else:
            welfare = cp.min(cp.hstack(net_utilities))
    elif fairness_alpha == 1:
        # Proportional fairness: maximize Σ log(U_i)
        # Need U_i > 0, which should hold if U_ind > 0
        welfare = sum(cp.log(net_u) for net_u in net_utilities)
    else:
        # General α-fairness: maximize Σ U_i^{1-α} / (1-α)
        # For 0 < α < 1: concave power function
        welfare = sum(cp.power(net_u, 1 - fairness_alpha) / (1 - fairness_alpha) 
                      for net_u in net_utilities)
    
    # === Constraints ===
    constraints = []
    
    # Trading constraints for each account
    for i in range(n):
        total_wealth_i = np.sum(W[i])
        constraints += [
            cp.sum(X[i]) == 0,                                          # self-financing
            cp.norm(X[i], 1) <= 0.10 * total_wealth_i,                  # turnover
            cp.quad_form(W[i] + X[i], Sigma) <= (sigmas[i] * total_wealth_i)**2,  # risk
        ]
    
    # Cost split constraints (compact formulation, Problem 19)
    # (a) Standalone lower bound: t(x_i^+, x_i^-) ≤ τ_i
    for i in range(n):
        standalone_cost = cp.sum(cp.multiply(alpha, cp.square(X_plus[i]) + cp.square(X_minus[i])))
        constraints.append(standalone_cost <= tau[i])
    
    # (b) Externality upper bound: t(Σ_{a≠i} x_a^+, Σ_{a≠i} x_a^-) ≤ Σ_{a≠i} τ_a
    for i in range(n):
        others_buy = sum(X_plus[j] for j in range(n) if j != i)
        others_sell = sum(X_minus[j] for j in range(n) if j != i)
        others_cost = cp.sum(cp.multiply(alpha, cp.square(others_buy) + cp.square(others_sell)))
        constraints.append(others_cost <= sum(tau[j] for j in range(n) if j != i))
    
    # (c) Budget balance: t(Σ_a x_a^+, Σ_a x_a^-) ≤ Σ_a τ_a
    total_cost = market_impact(X_plus, X_minus)
    constraints.append(total_cost <= cp.sum(tau))
    
    # (d) Coordination benefit (optional)
    if use_coordination_constraint and U_ind is not None:
        for i in range(n):
            constraints.append(net_utilities[i] >= U_ind[i])
    
    # For maximin with relative improvement
    if fairness_alpha is not None and fairness_alpha >= 50 and U_ind is not None:
        for i in range(n):
            constraints.append(
                (net_utilities[i] - U_ind[i]) >= min_improvement * abs(U_ind[i])
            )
    
    # Solve
    prob = cp.Problem(cp.Maximize(welfare), constraints)
    prob.solve(solver=cp.SCS, warm_start=True, max_iters=10000)
    
    # Extract results
    X_opt = np.array([X[i].value for i in range(n)])
    tau_opt = tau.value
    
    return X_opt, tau_opt, prob.value


def solve_mpo_cvxpylayer(mu_param, Sigma, alpha, W, sigmas, lambdas, 
                          fairness_alpha=0):
    """
    Create a CvxpyLayer version of the MPO problem for DFL.
    μ is treated as a PARAMETER that we differentiate with respect to.
    
    Returns: CvxpyLayer that maps μ → (x*, τ*)
    """
    from cvxpylayers.torch import CvxpyLayer
    import torch
    
    n = len(W)
    m = len(W[0])
    
    # PARAMETER: expected returns (what we predict)
    mu_p = cp.Parameter(m)
    
    # Decision variables (same as above)
    X_plus = [cp.Variable(m, nonneg=True) for _ in range(n)]
    X_minus = [cp.Variable(m, nonneg=True) for _ in range(n)]
    X = [X_plus[i] - X_minus[i] for i in range(n)]
    tau = cp.Variable(n)
    
    # Build utilities with parametric mu
    net_utilities = []
    for i in range(n):
        if lambdas[i] > 0:
            u_i = mu_p @ X[i] - lambdas[i] * cp.quad_form(W[i] + X[i], Sigma)
        else:
            u_i = mu_p @ X[i]
        net_utilities.append(u_i - tau[i])
    
    # Welfare function
    if fairness_alpha == 0:
        welfare = sum(net_utilities)
    elif fairness_alpha == 1:
        welfare = sum(cp.log(net_u) for net_u in net_utilities)
    else:
        welfare = sum(net_utilities)  # default to utilitarian
    
    # Constraints (same structure as above)
    constraints = []
    for i in range(n):
        total_wealth_i = float(np.sum(W[i]))
        constraints += [
            cp.sum(X[i]) == 0,
            cp.norm(X[i], 1) <= 0.10 * total_wealth_i,
            cp.quad_form(W[i] + X[i], Sigma) <= (sigmas[i] * total_wealth_i)**2,
        ]
    
    alpha_np = np.array(alpha)
    for i in range(n):
        standalone = cp.sum(cp.multiply(alpha_np, cp.square(X_plus[i]) + cp.square(X_minus[i])))
        constraints.append(standalone <= tau[i])
    
    for i in range(n):
        others_buy = sum(X_plus[j] for j in range(n) if j != i)
        others_sell = sum(X_minus[j] for j in range(n) if j != i)
        others_cost = cp.sum(cp.multiply(alpha_np, cp.square(others_buy) + cp.square(others_sell)))
        constraints.append(others_cost <= sum(tau[j] for j in range(n) if j != i))
    
    agg_buy = sum(X_plus)
    agg_sell = sum(X_minus)
    total_cost = cp.sum(cp.multiply(alpha_np, cp.square(agg_buy) + cp.square(agg_sell)))
    constraints.append(total_cost <= cp.sum(tau))
    
    prob = cp.Problem(cp.Maximize(welfare), constraints)
    
    # Create differentiable layer
    # variables_out = all x_plus, x_minus, tau we want gradients for
    all_vars = X_plus + X_minus + [tau]
    layer = CvxpyLayer(prob, parameters=[mu_p], variables=all_vars)
    
    return layer
```

### Step 3: DFL Training Loop

```python
"""
Decision-Focused Learning training loop for MPO.
"""
import torch
import torch.nn as nn
import numpy as np

class ReturnPredictor(nn.Module):
    """Simple MLP to predict expected returns from features."""
    def __init__(self, input_dim, m_assets, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m_assets),
        )
    
    def forward(self, z):
        return self.net(z)  # (batch, m_assets)


def train_dfl(model, mpo_layer, train_data, lr=1e-3, epochs=50,
              Sigma=None, alpha=None, W=None, sigmas=None, lambdas=None):
    """
    Train prediction model with decision-focused loss.
    
    Args:
        model: ReturnPredictor
        mpo_layer: CvxpyLayer from solve_mpo_cvxpylayer
        train_data: list of (features_z, true_mu) tuples
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(W)
    m = len(W[0])
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for z_t, mu_true in train_data:
            z_t = torch.FloatTensor(z_t)
            mu_true = torch.FloatTensor(mu_true)
            
            # Predict returns
            mu_hat = model(z_t)  # (m,)
            
            # Solve MPO with predicted returns (differentiable!)
            try:
                solution = mpo_layer(mu_hat, solver_args={'max_iters': 5000})
                # solution contains x_plus_i, x_minus_i, tau
                
                # Extract trades
                x_plus_list = solution[:n]
                x_minus_list = solution[n:2*n]
                tau_hat = solution[2*n]  # (n,)
                
            except Exception as e:
                print(f"Solver failed: {e}")
                continue
            
            # Compute decision quality under TRUE parameters
            # Net utility for each account under true μ
            decision_loss = 0.0
            for i in range(n):
                x_i = x_plus_list[i] - x_minus_list[i]
                w_i = torch.FloatTensor(W[i])
                
                # True utility
                if lambdas[i] > 0:
                    u_i_true = mu_true @ x_i - lambdas[i] * (w_i + x_i) @ torch.FloatTensor(Sigma) @ (w_i + x_i)
                else:
                    u_i_true = mu_true @ x_i
                
                # Net utility = utility - cost
                U_i = u_i_true - tau_hat[i]
                decision_loss -= U_i  # negative because we minimize loss
            
            optimizer.zero_grad()
            decision_loss.backward()
            optimizer.step()
            
            total_loss += decision_loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: avg loss = {total_loss / len(train_data):.6f}")


def train_two_stage(model, train_data, lr=1e-3, epochs=50):
    """Baseline: train with MSE loss on returns."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for z_t, mu_true in train_data:
            z_t = torch.FloatTensor(z_t)
            mu_true = torch.FloatTensor(mu_true)
            
            mu_hat = model(z_t)
            loss = mse(mu_hat, mu_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: MSE = {total_loss / len(train_data):.6f}")
```

### Step 4: Evaluation

```python
"""
Evaluation: compare Two-Stage vs DFL across fairness metrics.
"""

def evaluate(model, mpo_solver, test_data, W, Sigma, alpha, sigmas, lambdas,
             fairness_alpha_values=[0, 1, 50]):
    """
    For each test instance:
    1. Predict μ̂ = model(z)
    2. Solve MPO with different α-fairness values
    3. Evaluate under true μ
    4. Report: welfare, fairness metrics, per-account utilities
    """
    results = {}
    
    for alpha_f in fairness_alpha_values:
        welfare_vals = []
        min_improvements = []
        gini_coeffs = []
        
        for z_t, mu_true in test_data:
            mu_hat = model(torch.FloatTensor(z_t)).detach().numpy()
            
            # Solve MPO
            X_opt, tau_opt, _ = solve_mpo(
                mu_hat, Sigma, alpha, W, sigmas, lambdas,
                fairness_alpha=alpha_f
            )
            
            # Evaluate under true μ
            n = len(W)
            U_true = []
            for i in range(n):
                if lambdas[i] > 0:
                    u_i = mu_true @ X_opt[i] - lambdas[i] * (W[i] + X_opt[i]) @ Sigma @ (W[i] + X_opt[i])
                else:
                    u_i = mu_true @ X_opt[i]
                U_true.append(u_i - tau_opt[i])
            
            welfare_vals.append(sum(U_true))
            
            # Fairness metrics
            U_arr = np.array(U_true)
            gini = gini_coefficient(U_arr)
            gini_coeffs.append(gini)
        
        results[f'alpha={alpha_f}'] = {
            'mean_welfare': np.mean(welfare_vals),
            'mean_gini': np.mean(gini_coeffs),
        }
    
    return results


def gini_coefficient(utilities):
    """Compute Gini coefficient of utility distribution."""
    u = np.abs(utilities)
    if u.sum() == 0:
        return 0
    n = len(u)
    u_sorted = np.sort(u)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * u_sorted) / (n * np.sum(u_sorted))) - (n + 1) / n
```

---

## 6. Experimental Design

### Comparisons
| Method | Description |
|--------|-------------|
| **Two-Stage + Independent** | MSE-trained model → solve each account independently → pro rata split |
| **Two-Stage + MPO (α=0)** | MSE-trained model → utilitarian MPO |
| **Two-Stage + MPO (α=1)** | MSE-trained model → proportional fairness MPO |
| **Two-Stage + MPO (α→∞)** | MSE-trained model → maximin MPO |
| **DFL + MPO (α=0)** | DFL-trained model → utilitarian MPO |
| **DFL + MPO (α=1)** | DFL-trained model → proportional fairness MPO |
| **DFL + MPO (α→∞)** | DFL-trained model → maximin MPO |

### Metrics to Report
1. **Aggregate welfare:** Σ_i U_i (total net utility across all accounts)
2. **Worst-case utility:** min_i U_i (Rawlsian metric)
3. **Gini coefficient** of utility distribution (inequality measure)
4. **Per-account relative improvement** over independent baseline
5. **Price of fairness:** efficiency loss from α=0 to α→∞

### Key Research Questions
1. Does DFL improve welfare compared to two-stage, across all fairness levels?
2. Does the benefit of DFL vary with the fairness parameter α?
   - Hypothesis: DFL may be MORE beneficial under fairness constraints because fairness amplifies the cost of prediction errors
3. Does DFL training implicitly learn to balance accounts?
4. How does the number of accounts n affect DFL's advantage?

---

## 7. Implementation Notes

### Dependencies
```
pip install cvxpy cvxpylayers torch numpy scipy
```

### Known Issues / Gotchas
1. **cvxpylayers + SCS:** SCS is the recommended solver for cvxpylayers (supports conic programs). ECOS may not handle all constraint types.
2. **Numerical stability:** Market impact costs can create ill-conditioned problems when trades are very small. Add small regularization if needed.
3. **L1 norm constraint:** ||x||_1 ≤ c is NOT directly DCP. Reformulate as: x = x⁺ - x⁻, x⁺ ≥ 0, x⁻ ≥ 0, 1^T x⁺ + 1^T x⁻ ≤ c. (Already done in our formulation.)
4. **Quad form in constraints:** cp.quad_form(w+x, Sigma) requires Sigma to be PSD. Ensure this by adding small ridge: Sigma + 1e-6 * I.
5. **Log utility:** For α=1 (proportional fairness), need U_i > 0. May need to add explicit lower bound constraints.
6. **Coordination constraints:** U_i^IND depends on μ too, so if μ is a parameter, this creates a nested optimization. For DFL, you may want to compute U_i^IND once using the true μ and treat as constant.

### No Original Code
The original paper (2014) does NOT provide code. The formulations above are reconstructed from the paper's mathematical descriptions and numerical study parameters.

---

## 8. Summary: What Makes This Novel

1. **First DFL application to multi-portfolio optimization** (all existing DFL portfolio work is single-portfolio)
2. **First combination of DFL with α-fairness in finance** (only precedent is Verma et al. 2024 in healthcare)
3. **The prediction error propagation through coupled accounts** is fundamentally different from single-portfolio DFL — errors in μ̂ affect all n accounts simultaneously through shared market impact
4. **Natural experiment:** How does the DFL advantage change as fairness increases? This speaks to both the DFL and fairness literatures.
