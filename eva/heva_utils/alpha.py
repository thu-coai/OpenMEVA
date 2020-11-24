
import numpy as np

def krippendorff_alpha(rate, metric_type):
    '''
    based on https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py

    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        [value1, value2, value3, ...],  # rating for item1
        [value1, value2, value3, ...],  # rating for item2
        [value1, value2, value3, ...],  # rating for item3
        ...                            # more ratings
    ]
    metric_type: type of function calculating the pairwise distance
    '''

    def nominal_metric(a, b):
        return a != b

    def interval_metric(a, b):
        return (a-b)**2

    def ratio_metric(a, b):
        return ((a-b)/(a+b))**2

    if metric_type == "nominal":
        metric = nominal_metric
    elif metric_type == "interval":
        metric = interval_metric
    elif metric_type == "ratio":
        metric = ratio_metric
    else:
        raise Exception("metric_type must be one of ['nominal', 'interval', 'ratio'], but get %s"%metric_type)

    n = sum(len(pv) for pv in rate)  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    Do = 0.
    for grades in rate:
        gr = np.asarray(grades)
        Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in rate:
        d1 = np.asarray(g1)
        for g2 in rate:
            De += sum(np.sum(metric(d1, gj)) for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.
