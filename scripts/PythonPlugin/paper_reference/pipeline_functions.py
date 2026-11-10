import numpy as np

def polyfit_constrained_start(x, y, order, x0, y0):
    """
    Fit polinomiale di grado `order`, vincolato a passare da (x0, y0).
    Ritorna: trend_line, coeffs (inclusi i coef. vincolati, cioè con il termine costante = y0)
    """
    # Shift dei dati rispetto al vincolo
    x_shifted = x - x0
    y_shifted = y - y0

    # Costruzione della matrice di Vandermonde senza termine costante (che è y0)
    X = np.vander(x_shifted, N=order+1)[:, :-1]  # esclude il termine x^0

    # Fit dei coefficienti rimanenti
    coeffs_reduced = np.linalg.lstsq(X, y_shifted, rcond=None)[0]

    # Ricostruzione del polinomio completo: y0 + a₁(x−x₀) + a₂(x−x₀)² + ...
    trend_line = y0 + np.polyval(np.append(coeffs_reduced, 0), x_shifted)

    # Coefficienti del polinomio globale (per compatibilità)
    # NB: per avere anche il termine costante effettivo, ricostruiamo il polinomio globale
    full_poly = np.poly1d(np.append(coeffs_reduced, 0))  # append 0 per x^0
    poly_coeffs = full_poly.integ()(x - x0) * 0 + y0 + np.polyval(full_poly, x - x0)

    return trend_line, coeffs_reduced
