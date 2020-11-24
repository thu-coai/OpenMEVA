""" Script to implement Intraclass correlation coefficient (ICC).

Based on:
    https://www.rdocumentation.org/packages/irr/versions/0.84.1/topics/icc
"""
import numpy as np
from scipy.stats import f


def icc(ratings, model='oneway', type='agreement', unit='average', confidence_level=0.95):
    """Implement Intraclass correlation coefficient (ICC) for oneway and twoway models.

    Computes single score or average score ICCs as an index of interrater reliability
    of quantitative data. Additionally, F-test and confidence interval are computed.

    When considering which form of ICC is appropriate for an actual set of data, one has
    take several decisions (Shrout & Fleiss, 1979):
     - 1. Should only the subjects be considered as random effects ('"oneway"' model)
     or are subjects and raters randomly chosen from a bigger pool of persons ('"twoway"' model).
     - 2. If differences in judges' mean ratings are of interest, interrater '"agreement"'
      instead of '"consistency"' should be computed.
     - 3. If the unit of analysis is a mean of several ratings, unit should be changed to
     '"average"'. In most cases, however, single values (unit='"single"') are regarded.

    ICCs implemented:
    - ICC(1,1) -> model='oneway', type='agreement', unit='single'
    - ICC(2,1) -> model='twoway', type='agreement', unit='single'
    - ICC(3,1) -> model='twoway', type='consistency', unit='single'
    - ICC(1,k) -> model='oneway', type='agreement', unit='average'
    - ICC(2,k) -> model='twoway', type='agreement', unit='average'
    - ICC(3,k) -> model='twoway', type='consistency', unit='average'

    Parameters
    ----------
    ratings: array-like, shape (n_subjects, n_raters)
        Matrix with n subjects m raters
    model: string, optional (default='oneway')
        String specifying if a 'oneway' model with row effects random, or a
        'twoway' model with column and row effects random should be applied.
    type: string, optional (default='consistency')
        String specifying if 'consistency' or 'agreement' between raters
        should be estimated. If a 'oneway' model is used, only 'consistency'
         could be computed.
    unit: string, optional (default='single')
        String specifying the unit of analysis: Must be one of 'single' or
        'average'.
    confidence_level: float, optional (default=0.95)
        Confidence level of the interval.

    Returns
    -------
    coeff: float
        The intraclass correlation coefficient.
    Fvalue: float
        The value of the F-statistic.
    df1: int
        The numerator degrees of freedom.
    df2: int
        The denominator degrees of freedom.
    pvalue: float
        The two-tailed p-value.
    lbound: float
        The lower bound of the confidence interval.
    ubound: float
        The upper bound of the confidence interval.

    References
    ----------
        [1] - Bartko, J.J. (1966). The intraclass correlation coefficient as a measure of reliability.
        Psychological Reports, 19, 3-11.
        [2] - McGraw, K.O., & Wong, S.P. (1996), Forming inferences about some intraclass correlation
         coefficients. Psychological Methods, 1, 30-46.
        [3] - Shrout, P.E., & Fleiss, J.L. (1979), Intraclass correlation: uses in assessing rater
        reliability. Psychological Bulletin, 86, 420-428.
        [4] -
    """
    ratings = np.asarray(ratings)

    if (model, type, unit) not in {('oneway', 'agreement', 'single'),
                                   ('twoway', 'agreement', 'single'),
                                   ('twoway', 'consistency', 'single'),
                                   ('oneway', 'agreement', 'average'),
                                   ('twoway', 'agreement', 'average'),
                                   ('twoway', 'consistency', 'average'), }:
        raise ValueError('Using not implemented configuration.')

    n_subjects, n_raters = ratings.shape
    if n_subjects < 1:
        raise ValueError('Using one subject only. Add more subjects to calculate ICC.')

    SStotal = np.var(ratings, ddof=1) * (n_subjects * n_raters - 1)
    alpha = 1 - confidence_level

    MSr = np.var(np.mean(ratings, axis=1), ddof=1) * n_raters
    MSw = np.sum(np.var(ratings, axis=1, ddof=1) / n_subjects)
    MSc = np.var(np.mean(ratings, axis=0), ddof=1) * n_subjects
    MSe = (SStotal - MSr * (n_subjects - 1) - MSc * (n_raters - 1)) / ((n_subjects - 1) * (n_raters - 1))

    # Single Score ICCs
    if unit == 'single':
        if model == 'oneway':
            # ICC(1,1) One-Way Random, absolute
            coeff = (MSr - MSw) / (MSr + (n_raters - 1) * MSw)
            Fvalue = MSr / MSw
            df1 = n_subjects - 1
            df2 = n_subjects * (n_raters - 1)
            pvalue = 1 - f.cdf(Fvalue, df1, df2)

            # Confidence interval
            FL = Fvalue / f.ppf(1 - alpha, df1, df2)
            FU = Fvalue * f.ppf(1 - alpha, df2, df1)
            lbound = (FL - 1) / (FL + (n_raters - 1))
            ubound = (FU - 1) / (FU + (n_raters - 1))

        elif model == 'twoway':
            if type == 'agreement':
                # ICC(2,1) Two-Way Random, absolute
                coeff = (MSr - MSe) / (MSr + (n_raters - 1) * MSe + (n_raters / n_subjects) * (MSc - MSe))
                Fvalue = MSr / MSe
                df1 = n_subjects - 1
                df2 = (n_subjects - 1) * (n_raters - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # Confidence interval
                Fj = MSc / MSe
                vn = (n_raters - 1) * (n_subjects - 1) * (
                    (n_raters * coeff * Fj + n_subjects * (1 + (n_raters - 1) * coeff) - n_raters * coeff)) ** 2
                vd = (n_subjects - 1) * n_raters ** 2 * coeff ** 2 * Fj ** 2 + (
                        n_subjects * (1 + (n_raters - 1) * coeff) - n_raters * coeff) ** 2
                v = vn / vd

                FL = f.ppf(1 - alpha, n_subjects - 1, v)
                FU = f.ppf(1 - alpha, v, n_subjects - 1)
                lbound = (n_subjects * (MSr - FL * MSe)) / (FL * (
                        n_raters * MSc + (n_raters * n_subjects - n_raters - n_subjects) * MSe) + n_subjects * MSr)
                ubound = (n_subjects * (FU * MSr - MSe)) / (n_raters * MSc + (
                        n_raters * n_subjects - n_raters - n_subjects) * MSe + n_subjects * FU * MSr)

            elif type == 'consistency':
                # ICC(3,1) Two-Way Mixed, consistency
                coeff = (MSr - MSe) / (MSr + (n_raters - 1) * MSe)
                Fvalue = MSr / MSe
                df1 = n_subjects - 1
                df2 = (n_subjects - 1) * (n_raters - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # Confidence interval
                FL = Fvalue / f.ppf(1 - alpha, df1, df2)
                FU = Fvalue * f.ppf(1 - alpha, df2, df1)
                lbound = (FL - 1) / (FL + (n_raters - 1))
                ubound = (FU - 1) / (FU + (n_raters - 1))

    elif unit == 'average':
        if model == 'oneway':
            # ICC(1,k) One-Way Random, absolute
            coeff = (MSr - MSw) / MSr
            Fvalue = MSr / MSw
            df1 = n_subjects - 1
            df2 = n_subjects * (n_raters - 1)
            pvalue = 1 - f.cdf(Fvalue, df1, df2)

            # Confidence interval
            FL = (MSr / MSw) / f.ppf(1 - alpha, df1, df2)
            FU = (MSr / MSw) * f.ppf(1 - alpha, df2, df1)
            lbound = 1 - 1 / FL
            ubound = 1 - 1 / FU

        elif model == 'twoway':
            if type == 'agreement':
                # ICC(2,k) Two-Way Random, absolute
                coeff = (MSr - MSe) / (MSr + (MSc - MSe) / n_subjects)
                Fvalue = MSr / MSe
                df1 = n_subjects - 1
                df2 = (n_subjects - 1) * (n_raters - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # Confidence interval
                icc2 = (MSr - MSe) / (MSr + (n_raters - 1) * MSe + (n_raters / n_subjects) * (MSc - MSe))
                Fj = MSc / MSe
                vn = (n_raters - 1) * (n_subjects - 1) * (
                    (n_raters * icc2 * Fj + n_subjects * (1 + (n_raters - 1) * icc2) - n_raters * icc2)) ** 2
                vd = (n_subjects - 1) * n_raters ** 2 * icc2 ** 2 * Fj ** 2 + (
                        n_subjects * (1 + (n_raters - 1) * icc2) - n_raters * icc2) ** 2
                v = vn / vd

                FL = f.ppf(1 - alpha, n_subjects - 1, v)
                FU = f.ppf(1 - alpha, v, n_subjects - 1)
                lb2 = (n_subjects * (MSr - FL * MSe)) / (FL * (
                        n_raters * MSc + (n_raters * n_subjects - n_raters - n_subjects) * MSe) + n_subjects * MSr)
                ub2 = (n_subjects * (FU * MSr - MSe)) / (n_raters * MSc + (
                        n_raters * n_subjects - n_raters - n_subjects) * MSe + n_subjects * FU * MSr)
                lbound = lb2 * n_raters / (1 + lb2 * (n_raters - 1))
                ubound = ub2 * n_raters / (1 + ub2 * (n_raters - 1))

            elif type == 'consistency':
                # ICC(3,k) Two-Way Mixed, consistency
                coeff = (MSr - MSe) / MSr
                Fvalue = MSr / MSe
                df1 = n_subjects - 1
                df2 = (n_subjects - 1) * (n_raters - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # Confidence interval
                FL = Fvalue / f.ppf(1 - alpha, df1, df2)
                FU = Fvalue * f.ppf(1 - alpha, df2, df1)
                lbound = 1 - 1 / FL
                ubound = 1 - 1 / FU

    return coeff, Fvalue, df1, df2, pvalue, lbound, ubound
