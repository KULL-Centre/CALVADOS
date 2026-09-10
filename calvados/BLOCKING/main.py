from typing import cast

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from .block_tools import FloatArray, blocking, check, fblocking


class BlockAnalysis:
    """Estimate statistically reliable errors from correlated samples.

    The input series may represent one or more concatenated replicas and may
    be weighted directly or reweighted from a bias at temperature ``T``.
    Blocking statistics are calculated at initialization; :meth:`SEM` selects
    a converged block size and error, while the remaining methods construct
    probability densities, free-energy surfaces, or weighted averages.
    """

    def __init__(
        self,
        x: FloatArray,
        multi: int = 1,
        weights: FloatArray | None = None,
        bias: FloatArray | None = None,
        T: float | None = None,
        interval_low: float | None = None,
        interval_up: float | None = None,
        dt: float = 1,
    ) -> None:
        """Prepare weighted or unweighted blocking statistics for ``x``."""
        self.multi = multi
        self.x = check(x, self.multi)
        self.w = weights
        
        self.interval = [self.x.min(), self.x.max()]
        if (interval_low is not None) and (self.x.min() < interval_low):
            self.interval[0] = interval_low
        if (interval_up is not None) and (self.x.max() > interval_up):
            self.interval[1] = interval_up

        if (self.w is None) and (bias is not None):
            bias -= np.max(bias)
            self.kbT = 0.008314463 * cast(float, T)
            self.w = np.exp(bias/(self.kbT))
            self.w = check(self.w, self.multi)

        if self.w is None:
            self.stat = blocking(self.x, self.multi)
            self.av = self.x.mean()
        else:
            self.kbT = 0.008314463 * cast(float, T)
            assert self.w is not None
            self.w /= self.w.sum()
            self.stat = fblocking(self.x, self.w, self.kbT, self.multi, self.interval)

        self.stat[...,0] /= dt

    def SEM(self) -> None:
        """Select a converged block size and store it with the standard error."""

        def find_n_intersect(x: FloatArray, stat: FloatArray) -> int:
                """Score how many error intervals contain a candidate value."""
                c=0
                for i,p in enumerate(stat):
                    if (x <= p[1]+p[2]) and (x >= p[1]-p[2]):
                        c += 1
                        #c += norm(p[1],p[2]).pdf(x)
                return -c

        c = np.zeros(len(self.stat))
        for i,b in enumerate(self.stat):
            lower_bound = b[1]-b[2]
            upper_bound = b[1]+b[2]
            bnds = [(lower_bound, upper_bound)]
            c[i] -= minimize( fun=find_n_intersect, x0=b[1], args=self.stat[self.stat[...,0] > b[0]], bounds=bnds ).fun
        self.bs = self.stat[...,0][np.argmax(c)]
        self.sem = self.stat[...,1][np.argmax(c)]

        if self.bs > self.stat[-1,0]/3:
            print('WARNING: fixed point of the error may have not been reached!')
 
    def get_pdf(
        self, cv: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return grid points, kernel-density values, and blocking errors."""

        min_ = self.interval[0]
        max_ = self.interval[1]
        x = np.linspace( min_, max_, num = 100 )
        if cv is not None:
            cv = check(cv, self.multi)
            u = cast(
                FloatArray,
                gaussian_kde(cv, bw_method="silverman", weights=self.w).evaluate(x),
            )
        else:
            u = cast(
                FloatArray,
                gaussian_kde(
                    self.x, bw_method="silverman", weights=self.w
                ).evaluate(x),
            )

        N = int(len(self.x))
        Nb = int(N / self.bs)

        weights = cast(FloatArray, self.w)
        W = weights.sum()
        S = (weights**2).sum()

        blocks_pi: list[FloatArray] = []
        for n in range(1, Nb+1):
            end = int( self.bs * n )
            start = int( end - self.bs )
            pdf_i = cast(
                FloatArray,
                gaussian_kde(
                    self.x[start:end],
                    bw_method="silverman",
                    weights=weights[start:end],
                ).evaluate(x),
            )
            wi = weights[start:end].sum()
            blocks_pi.append( wi*(pdf_i-u)**2 )
    
        blocks_pi_array = np.array(blocks_pi)
        e = np.sqrt(blocks_pi_array.sum(axis=0) / (Nb*(W-S/W)))
    
        return x, u, e

    def get_fes(
        self, maxkj: float = 25, cv: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return coordinates, relative free energies, and propagated errors."""
        if cv is not None:
            x, H, E = self.get_pdf(cv)
        else:
            x, H, E = self.get_pdf()
        F = -self.kbT * np.log(H)
        FE = self.kbT * E / H

        F -= F.min()
        maxkj_ndx = np.where(F<maxkj)

        return x[maxkj_ndx], F[maxkj_ndx], FE[maxkj_ndx]

    def get_av_err(self, cv: FloatArray | None = None) -> tuple[float, float]:
        """Return the density-weighted average and its propagated error."""
        if cv is not None:
            x, H, E = self.get_pdf(cv)
        else:
            x, H, E = self.get_pdf()
        H /= H.sum()
        E /= H.sum()
        av = np.average(x, weights=H)
        err = np.sqrt((x**2*E**2).sum())
        return cast(float, av), cast(float, err)
