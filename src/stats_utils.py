"""Shared statistics utilities for paper plots."""
import numpy as np
from scipy import stats as scipy_stats


def r2z(r):
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -0.9999, 0.9999))


def z2r(z):
    return np.tanh(z)


def mean_r_ci(r_vals, alpha=0.05):
    """Mean r via Fisher z-transform with t-based 95% CI."""
    z = r2z(r_vals)
    n = len(z)
    if n < 2:
        return float(z2r(np.mean(z))), np.nan, np.nan
    mz = np.mean(z)
    se = np.std(z, ddof=1) / np.sqrt(n)
    tc = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(z2r(mz)), float(z2r(mz - tc * se)), float(z2r(mz + tc * se))


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction (returns adjusted p-values)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    q = np.empty(n)
    q[order] = pvals[order] * n / (np.arange(1, n + 1))
    np.minimum.accumulate(q[order[::-1]], out=q[order[::-1]])
    return np.clip(q, 0, 1)


def fmt_p(p):
    if p < 0.0001:
        return '< 0.0001'
    return f'{p:.4f}'


def clean_style(df):
    """White background, black borders — clean for article copy-paste."""
    return df.style.set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('background-color', 'white')]},
        {'selector': 'th, td', 'props': [
            ('border', '1px solid black'), ('padding', '5px 10px'),
            ('background-color', 'white'), ('text-align', 'center'),
        ]},
        {'selector': 'th', 'props': [('font-weight', 'bold')]},
    ])


def second_order_stats(x_r, y_r, alpha=0.05):
    """
    Second-order Pearson r between z-transformed vectors + Wilcoxon signed-rank test.
    Returns dict with formatted strings and raw floats (prefixed with _).
    """
    xz = r2z(np.asarray(x_r, dtype=float))
    yz = r2z(np.asarray(y_r, dtype=float))
    n = len(xz)
    r_val, p_r = scipy_stats.pearsonr(xz, yz)
    if n > 3:
        z_rv = r2z(float(r_val))
        se_z = 1.0 / np.sqrt(n - 3)
        tc_z = scipy_stats.t.ppf(1 - alpha / 2, df=n - 3)
        ci_lo = float(z2r(z_rv - tc_z * se_z))
        ci_hi = float(z2r(z_rv + tc_z * se_z))
    else:
        ci_lo = ci_hi = np.nan
    diff = xz - yz
    try:
        w_stat, p_w = scipy_stats.wilcoxon(diff, alternative='two-sided')
        # Signed z via normal approximation: T+ determines direction of effect
        T_plus, _ = scipy_stats.wilcoxon(diff, alternative='greater')
        n_nz = int(np.sum(diff != 0))
        mu_w = n_nz * (n_nz + 1) / 4.0
        sign_z = 1.0 if T_plus >= mu_w else -1.0
        z_stat = sign_z * float(scipy_stats.norm.ppf(1.0 - p_w / 2.0))
    except ValueError:
        w_stat, p_w, z_stat = float('nan'), float('nan'), float('nan')
    d_mean = float(np.mean(diff))
    d_se = float(np.std(diff, ddof=1) / np.sqrt(n))
    td = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
    return {
        'n': n,
        'Correlation': f'{float(r_val):.2f}',
        '95% CI (r)': f'[{ci_lo:.2f}, {ci_hi:.2f}]',
        'p (r)': fmt_p(p_r),
        'Difference': f'{d_mean:.2f}',
        '95% CI (diff z)': f'[{d_mean - td * d_se:.2f}, {d_mean + td * d_se:.2f}]',
        'z': f'{z_stat:.2f}',
        'p (W)': fmt_p(float(p_w)),
        '_r': float(r_val),
        '_W': float(w_stat),
        '_z': z_stat,
        '_p_r': float(p_r),
        '_p_W': float(p_w),
    }
