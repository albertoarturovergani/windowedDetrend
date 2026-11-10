def plot_exponential_black_half(ax, tau_rise=5, tau_decay=10, A=1.0, t_max=100, dt=0.1):
    """
    Disegna su 'ax' una funzione esponenziale smorzata:
    - salita tratteggiata nera con area tratteggiata nera
    - discesa nera piena (curva + area)
    - assi e griglia nascosti
    """
    t = np.arange(0, t_max, dt)
    t_peak = tau_rise * np.log(1 + tau_decay / tau_rise)
    y = A * (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)

    mask_rise = t <= t_peak
    mask_decay = t > t_peak

    # Salita: curva tratteggiata + area tratteggiata
    ax.plot(t[mask_rise], -y[mask_rise], linestyle='--', color='black', linewidth=2)
    ax.fill_between(t[mask_rise], -y[mask_rise], color='none', edgecolor='black', hatch='///')

    # Discesa: curva piena + area piena
    ax.plot(t[mask_decay], -y[mask_decay], linestyle='-', color='black', linewidth=2)
    ax.fill_between(t[mask_decay], -y[mask_decay], color='black', alpha=0.5)

    # Rimuovi assi
    ax.axis('off')


def fit_constrained_poly(t_decay, y_decay, V_peak, degree=5, t_peak=15):
    t_shifted = t_decay - t_peak
    y_shifted = y_decay - V_peak
    X = np.vstack([t_shifted**i for i in range(1, degree + 1)]).T
    coeffs = np.linalg.lstsq(X, y_shifted, rcond=None)[0]
    return V_peak + X @ coeffs

# Funzioni di base
def generate_damped_oscillator(t, A=5.0, f=50.0, lambda_=25, t0=10):
    y = np.zeros_like(t)
    mask = t >= t0
    t_shifted = t[mask] - t0
    y[mask] = A * np.exp(-lambda_ * t_shifted / 1000) * np.sin(2 * np.pi * f * t_shifted / 1000)
    return y

def capacitive_artifact(t, t0, t_peak, tauRise, tauDecay, scale=-100):
    V0 = 1.0
    signal = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t0:
            signal[i] = 0
        elif ti < t_peak:
            signal[i] = V0 * (1 - np.exp(-(ti - t0) / tauRise))
        else:
            V_peak = V0 * (1 - np.exp(-(t_peak - t0) / tauRise))
            signal[i] = V_peak * np.exp(-(ti - t_peak) / tauDecay)
    return scale * signal

# Funzione carica-scarica
def carica_scarica(t, t0, t_peak, tauRise, tauDecay):
    V0 = 1.0
    signal = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t0:
            signal[i] = 0
        elif ti < t_peak:
            signal[i] = V0 * (1 - np.exp(-(ti - t0) / tauRise))
        else:
            V_peak = V0 * (1 - np.exp(-(t_peak - t0) / tauRise))
            signal[i] = V_peak * np.exp(-(ti - t_peak) / tauDecay)
    return signal





#def biexp(t, A, tau1, tau2, t0):
#    t_adj = np.clip(np.asarray(t) - t0, 0, None)
#    return A * (1 - np.exp(-t_adj / tau1)) * np.exp(-t_adj / tau2)

def biexp(t, A, tau1, tau2, t0):
    t_adj = np.clip(np.asarray(t) - t0, 0, None)
    # Limita gli argomenti dell'esponenziale per evitare overflow
    exp1 = np.exp(-np.clip(t_adj / tau1, 0, 700))  # evita valori troppo grandi
    exp2 = np.exp(-np.clip(t_adj / tau2, 0, 700))
    return A * (1 - exp1) * exp2


def exp_rise(t, A, tau, t0):
    t_adj = np.clip(np.asarray(t) - t0, 0, None)
    return A * (1 - np.exp(-t_adj / tau))

def exp_decay_constrained(t, tau, t_peak, t_full, y_full):
    interp = interp1d(t_full, y_full, kind='linear', bounds_error=False, fill_value="extrapolate")
    V_peak = float(interp(t_peak))
    return V_peak * np.exp(-(t - t_peak) / tau)

def add_scale_symbol(ax, dx=0.1, dy=0.1):
    # Coordinate in assi (0-1)
    x0, y0 = 0.88, 0.08  # in basso a destra
    ax.plot([x0, x0 + dx], [y0, y0], color='black', lw=1, transform=ax.transAxes, clip_on=False)
    ax.plot([x0, x0], [y0, y0 + dy], color='black', lw=1, transform=ax.transAxes, clip_on=False)
    ax.text(x0 + dx/2, y0 - 0.02, '100 ms', ha='center', va='top', transform=ax.transAxes)
    ax.text(x0 - 0.01, y0 + dy/2, '1', ha='right', va='center', transform=ax.transAxes)

def add_scale_symbol(ax, dx_ms=100, dy_uv=1, x0=0.65, y0=0.06):
    # Ottieni limiti assi per conversione unità → percentuali
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Conversione: da ms → unità assi (x)
    dx = dx_ms / (xlim[1] - xlim[0])

    # Conversione: da μV → unità assi (y)
    dy = dy_uv / (ylim[1] - ylim[0])

    # Posizione in basso a destra (coordinate normalizzate 0-1)
    #x0, y0 = 0.65, 0.06

    # Disegna le barrette
    ax.plot([x0, x0 + dx], [y0, y0], color='black', lw=1, transform=ax.transAxes, clip_on=False)
    ax.plot([x0, x0], [y0, y0 + dy], color='black', lw=1, transform=ax.transAxes, clip_on=False)

    # Etichette
    ax.text(x0 + dx / 2, y0 - 0.02, r'$100\ \mathrm{ms}$',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(x0 - 0.01, y0 + dy / 2, r'$1\ \mu V$',
            ha='right', va='center', transform=ax.transAxes)


def base_axis(ax, t0=0):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(t0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
