import numpy as np
import ot

# Simulate dataset and format_dataset functions (assuming they are defined elsewhere)
from utils import simulate_dataset, format_dataset
from disparate import disparate  # Assuming this is a custom function for computing disparate impact

def geometric_repair(X0, X1, lmbd):
    # Compute linear transport mappings using LinearTransport
    ot_mapping_01 = ot.da.LinearTransport()
    ot_mapping_01.fit(Xs=X0, Xt=X1)  # Fit mapping from X0 to X1

    ot_mapping_10 = ot.da.LinearTransport()
    ot_mapping_10.fit(Xs=X1, Xt=X0)  # Fit mapping from X1 to X0

    # Compute weights
    w0, w1 = X0.shape[0], X1.shape[0]
    w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)

    # Compute barycenters using the transform method
    barycenter_0 = w1 * ot_mapping_01.transform(Xs=X0) + w0 * X0
    barycenter_1 = w0 * ot_mapping_10.transform(Xs=X1) + w1 * X1

    # Repair datasets
    X0_repaired = lmbd * barycenter_0 + (1 - lmbd) * X0
    X1_repaired = lmbd * barycenter_1 + (1 - lmbd) * X1

    return X0_repaired, X1_repaired

def DI_list_geometric_repair(X0, X1, clf):
    # Compute linear transport mappings using LinearTransport
    ot_mapping_01 = ot.da.LinearTransport()
    ot_mapping_01.fit(Xs=X0, Xt=X1)  # Fit mapping from X0 to X1

    ot_mapping_10 = ot.da.LinearTransport()
    ot_mapping_10.fit(Xs=X1, Xt=X0)  # Fit mapping from X1 to X0

    # Define the geometric repair function
    def geometric_repair(X0, X1, lmbd):
        w0, w1 = X0.shape[0], X1.shape[0]
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)

        # Compute barycenters using the transform method
        barycenter_0 = w1 * ot_mapping_01.transform(Xs=X0) + w0 * X0
        barycenter_1 = w0 * ot_mapping_10.transform(Xs=X1) + w1 * X1

        X0_repaired = lmbd * barycenter_0 + (1 - lmbd) * X0
        X1_repaired = lmbd * barycenter_1 + (1 - lmbd) * X1
        return X0_repaired, X1_repaired

    # Define lambda values
    lambdas = np.linspace(0, 1, 100)
    DIs = []

    # Define coefficients for logistic regression
    beta0 = np.array([1, -1, -0.5, 1, -1])
    beta1 = np.array([1, -0.4, 1, -1, 1])

    for lmbd in lambdas:
        # Repair datasets
        X0r, X1r = geometric_repair(X0, X1, lmbd)

        # Compute outcomes using logistic regression
        y0 = 1 / (1 + np.exp(-X0r.dot(beta0)))
        y1 = 1 / (1 + np.exp(-X1r.dot(beta1)))

        # Format dataset
        X, Y = format_dataset(X0r, X1r, y0, y1)

        # Train logistic regression model
        clf = clf.fit(X[:, 1:], (Y > 0.5).astype(int))
        Y_pred = clf.predict(X[:, 1:])

        # Compute disparate impact
        DIs.append(disparate(X, Y_pred, 0)[1])

    return DIs

def random_repair(X0, X1, lmbd):
    # Compute linear transport mappings using LinearTransport
    ot_mapping_01 = ot.da.LinearTransport()
    ot_mapping_01.fit(Xs=X0, Xt=X1)  # Fit mapping from X0 to X1

    ot_mapping_10 = ot.da.LinearTransport()
    ot_mapping_10.fit(Xs=X1, Xt=X0)  # Fit mapping from X1 to X0

    # Compute weights
    w0, w1 = X0.shape[0], X1.shape[0]
    w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)

    # Compute barycenters using the transform method
    barycenter_0 = w1 * ot_mapping_01.transform(Xs=X0) + w0 * X0
    barycenter_1 = w0 * ot_mapping_10.transform(Xs=X1) + w1 * X1

    # Apply random repair
    ber0 = np.random.binomial(1, lmbd, size=(X0.shape[0], 1))
    ber1 = np.random.binomial(1, lmbd, size=(X1.shape[0], 1))

    X0_repaired = ber0 * barycenter_0 + (1 - ber0) * X0
    X1_repaired = ber1 * barycenter_1 + (1 - ber1) * X1

    return X0_repaired, X1_repaired

def DI_list_random_repair(X0, X1, clf):
    # Compute linear transport mappings using LinearTransport
    ot_mapping_01 = ot.da.LinearTransport()
    ot_mapping_01.fit(Xs=X0, Xt=X1)  # Fit mapping from X0 to X1

    ot_mapping_10 = ot.da.LinearTransport()
    ot_mapping_10.fit(Xs=X1, Xt=X0)  # Fit mapping from X1 to X0

    # Define the random repair function
    def random_repair(X0, X1, lmbd):
        w0, w1 = X0.shape[0], X1.shape[0]
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)

        # Compute barycenters using the transform method
        barycenter_0 = w1 * ot_mapping_01.transform(Xs=X0) + w0 * X0
        barycenter_1 = w0 * ot_mapping_10.transform(Xs=X1) + w1 * X1

        ber0 = np.random.binomial(1, lmbd, size=(X0.shape[0], 1))
        ber1 = np.random.binomial(1, lmbd, size=(X1.shape[0], 1))

        X0_repaired = ber0 * barycenter_0 + (1 - ber0) * X0
        X1_repaired = ber1 * barycenter_1 + (1 - ber1) * X1
        return X0_repaired, X1_repaired

    # Define lambda values
    lambdas = np.linspace(0, 1, 100)
    DIs = []

    # Define coefficients for logistic regression
    beta0 = np.array([1, -1, -0.5, 1, -1])
    beta1 = np.array([1, -0.4, 1, -1, 1])

    for lmbd in lambdas:
        # Repair datasets
        X0r, X1r = random_repair(X0, X1, lmbd)

        # Compute outcomes using logistic regression
        y0 = 1 / (1 + np.exp(-X0r.dot(beta0)))
        y1 = 1 / (1 + np.exp(-X1r.dot(beta1)))

        # Format dataset
        X, Y = format_dataset(X0r, X1r, y0, y1)

        # Train logistic regression model
        clf = clf.fit(X[:, 1:], (Y > 0.5).astype(int))
        Y_pred = clf.predict(X[:, 1:])

        # Compute disparate impact
        DIs.append(disparate(X, Y_pred, 0)[1])

    return DIs