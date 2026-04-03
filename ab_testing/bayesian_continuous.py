"""
Bayesian continuous metrics: auto-selects StudentT or LogNormal model.

Model selection rules:
  - All positive data AND skewness > 1.0  →  LogNormal (log-normal)
  - Contains zeros/negatives OR skewness ≤ 1.0  →  StudentT (heavy-tailed symmetric)

Prior (StudentT):   mu ~ Normal, sigma ~ HalfNormal, nu ~ Exponential
Prior (LogNormal):  mu_log ~ Normal, sigma_log ~ HalfNormal
Posterior: no closed form — uses NumPyro NUTS MCMC

Dependencies: pip install "jax[cpu]" numpyro
Advantage: JAX pure-Python JIT, no C++ compiler needed, works on Windows
"""
import numpy as np

from .metrics import BayesianResult, compute_bayesian_metrics


def _stratified_subsample(rng: np.random.Generator, data: np.ndarray, size: int) -> np.ndarray:
    """
    Stratified subsampling across 10 quantile strata.

    Compared to pure random sampling, each stratum's high/low-value users
    appear in the correct proportion, avoiding mean estimate volatility
    caused by randomly including/excluding high-value outliers in heavy-tailed revenue data.
    """
    if len(data) <= size:
        return data.copy()

    n_strata = 10
    sorted_idx = np.argsort(data)
    strata = np.array_split(sorted_idx, n_strata)

    # Largest-remainder allocation: floor base quotas, distribute remainder by fractional part
    quotas = [size * len(s) / len(data) for s in strata]
    ns = [int(q) for q in quotas]
    remainders = sorted(range(n_strata), key=lambda i: -(quotas[i] - ns[i]))
    for i in remainders[:size - sum(ns)]:
        ns[i] += 1

    samples = []
    for stratum, n in zip(strata, ns):
        n = min(n, len(stratum))
        chosen = rng.choice(stratum, size=n, replace=False)
        samples.append(data[chosen])

    return np.concatenate(samples)


def _compute_skewness(data: np.ndarray) -> float:
    """Compute skewness; returns 0 if std is zero."""
    std = data.std()
    if std == 0:
        return 0.0
    return float(np.mean(((data - data.mean()) / std) ** 3))


def _detect_distribution(data: np.ndarray) -> tuple[str, float]:
    """
    Detect distribution type to decide between log1p-transform and StudentT.

    log1p condition: skewness > 1.0
      - Uses log(1+x) transform; supports zeros (log(1+0)=0), not negatives
      - Falls back to StudentT when negatives are present

    Returns:
        (model_name, skewness) where model_name is 'lognormal' or 'student_t'
    """
    skewness = _compute_skewness(data)
    if np.any(data < 0) or skewness <= 1.0:
        return "student_t", skewness
    return "lognormal", skewness


class BayesianContinuous:
    """
    Args:
        historical_mean:   Historical data mean (original scale), sets mu prior
        historical_std:    Historical data std (original scale), sets sigma prior
        prior_strength:    Prior strength (equivalent historical sample size); higher = tighter prior
        nu_expected:       Expected degrees of freedom for StudentT prior, controls tail thickness
        max_mcmc_samples:  Maximum samples per group passed to MCMC; excess is stratified-subsampled
        mcmc_draws:        Number of NUTS posterior samples
        mcmc_tune:         Number of NUTS warm-up steps
        mde:               Minimum detectable effect (absolute, original scale)

    Attributes (available after fit):
        detected_model:    Model actually used — 'lognormal' or 'student_t'
        skewness:          Skewness of the combined data (set after fit)
    """

    def __init__(
        self,
        historical_mean: float = None,
        historical_std: float = None,
        prior_strength: int = 50,
        nu_expected: float = 30.0,
        max_mcmc_samples: int = 1000,
        mcmc_draws: int = 1000,
        mcmc_tune: int = 500,
        mde: float = 3.0,
        # kept for interface compatibility
        n_samples: int = 200_000,
        target_accept: float = 0.9,
    ):
        self.historical_mean  = historical_mean
        self.historical_std   = historical_std
        self.prior_strength   = prior_strength
        self.nu_expected      = nu_expected
        self.max_mcmc_samples = max_mcmc_samples
        self.mcmc_draws       = mcmc_draws
        self.mcmc_tune        = mcmc_tune
        self.mde              = mde

        self.detected_model: str = "unknown"
        self.skewness: float     = 0.0

    def fit(self, data_a: np.ndarray, data_b: np.ndarray) -> BayesianResult:
        """
        Auto-detect distribution, select StudentT or LogNormal model, run NUTS sampling.
        """
        try:
            import jax
            import jax.numpy as jnp
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
        except ImportError:
            raise ImportError(
                "Bayesian analysis for continuous metrics requires NumPyro.\n"
                'Please run: pip install "jax[cpu]" numpyro'
            )

        # ── Subsample ─────────────────────────────────────────────
        rng_np = np.random.default_rng()
        if len(data_a) > self.max_mcmc_samples:
            data_a = _stratified_subsample(rng_np, data_a, self.max_mcmc_samples)
        if len(data_b) > self.max_mcmc_samples:
            data_b = _stratified_subsample(rng_np, data_b, self.max_mcmc_samples)

        all_data = np.concatenate([data_a, data_b])

        # ── Distribution detection ────────────────────────────────
        self.detected_model, self.skewness = _detect_distribution(all_data)

        # ── NUTS common setup ─────────────────────────────────────
        n_chains = min(2, jax.local_device_count())

        def _run_mcmc(model):
            mcmc = MCMC(
                NUTS(model),
                num_warmup=self.mcmc_tune,
                num_samples=self.mcmc_draws,
                num_chains=n_chains,
                progress_bar=True,
            )
            mcmc.run(jax.random.PRNGKey(42), obs_a, obs_b)
            return mcmc.get_samples()

        # ── Model branch ──────────────────────────────────────────
        if self.detected_model == "lognormal":
            # log1p transform: log(1+x), zero-safe, compresses right tail
            # posterior is back-transformed via expm1 to original scale
            log1p_a = np.log1p(data_a)
            log1p_b = np.log1p(data_b)
            all_log1p = np.concatenate([log1p_a, log1p_b])

            if self.historical_mean is not None:
                mu_hat_log = float(np.log1p(self.historical_mean))
                if self.historical_std is not None:
                    cv = self.historical_std / (self.historical_mean + 1.0)
                    sigma_hat_log = float(np.sqrt(np.log1p(cv ** 2)))
                else:
                    sigma_hat_log = float(all_log1p.std())
            else:
                mu_hat_log    = float(all_log1p.mean())
                sigma_hat_log = float(all_log1p.std())

            prior_std_log = sigma_hat_log / (self.prior_strength ** 0.5)

            obs_a = jnp.array(log1p_a, dtype=jnp.float32)
            obs_b = jnp.array(log1p_b, dtype=jnp.float32)

            def _model_lognormal(obs_a, obs_b):
                mu_a    = numpyro.sample("mu_a",    dist.Normal(mu_hat_log, prior_std_log))
                mu_b    = numpyro.sample("mu_b",    dist.Normal(mu_hat_log, prior_std_log))
                sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(sigma_hat_log))
                sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(sigma_hat_log))
                numpyro.sample("obs_a", dist.Normal(mu_a, sigma_a), obs=obs_a)
                numpyro.sample("obs_b", dist.Normal(mu_b, sigma_b), obs=obs_b)

            samples   = _run_mcmc(_model_lognormal)
            # expm1 = exp(x) - 1, reverses log1p; posterior mean ≈ geometric mean on original scale
            samples_a = np.expm1(np.array(samples["mu_a"]))
            samples_b = np.expm1(np.array(samples["mu_b"]))

        else:
            # StudentT: model directly on original scale
            mu_hat    = self.historical_mean if self.historical_mean is not None \
                        else float(all_data.mean())
            sigma_hat = self.historical_std  if self.historical_std  is not None \
                        else float(all_data.std())

            obs_a = jnp.array(data_a, dtype=jnp.float32)
            obs_b = jnp.array(data_b, dtype=jnp.float32)

            def _model_student_t(obs_a, obs_b):
                mu_a    = numpyro.sample("mu_a",    dist.Normal(mu_hat, mu_hat / (self.prior_strength ** 0.5)))
                mu_b    = numpyro.sample("mu_b",    dist.Normal(mu_hat, mu_hat / (self.prior_strength ** 0.5)))
                sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(sigma_hat))
                sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(sigma_hat))
                nu      = numpyro.sample("nu",      dist.Exponential(1.0 / self.nu_expected))
                numpyro.sample("obs_a", dist.StudentT(nu, mu_a, sigma_a), obs=obs_a)
                numpyro.sample("obs_b", dist.StudentT(nu, mu_b, sigma_b), obs=obs_b)

            samples   = _run_mcmc(_model_student_t)
            samples_a = np.array(samples["mu_a"])
            samples_b = np.array(samples["mu_b"])

        return compute_bayesian_metrics(samples_a, samples_b, self.mde)
