order_poly_offset = n_offset - 1
poly_coeffs_B = np.polyfit(x_offset, y_offset, order_poly_offset)
trend_line_B = np.polyval(poly_coeffs_B, x_offset)
