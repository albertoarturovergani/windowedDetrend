# Unmodified TMSpathPipeline functions (GPL-3.0), preserved as test references.
# Gaussian and resample branches only; provenance in offset_correction.json.
def generate_noise_from_distribution(
    time_series,
    model="Gaussian",
    n_samples=1000,
    rng=None
):
    import numpy as np

    if rng is None:
        rng=np.random.default_rng(42)

    x=np.asarray(
        time_series,
        dtype=float
    ).ravel()

    x=x[
        np.isfinite(x)
    ]

    if x.size==0:
        return np.zeros(
            n_samples,
            dtype=float
        )

    mean=float(
        np.mean(x)
    )

    std=float(
        np.std(x)*0.5
    )

    median=float(
        np.median(x)
    )

    minimum=float(
        np.min(x)
    )

    maximum=float(
        np.max(x)
    )

    epsilon=1e-8
    std=max(
        std,
        epsilon
    )

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        minimum=mean-std
        maximum=mean+std

    if minimum==maximum:
        minimum-=std
        maximum+=std

    if minimum>maximum:
        minimum,maximum=maximum,minimum

    model_name=str(
        model
    ).strip().lower()

    if model_name=="gaussian":
        return rng.normal(
            loc=mean,
            scale=std,
            size=n_samples
        )

    if model_name=="exponential":
        return rng.exponential(
            scale=max(std,epsilon),
            size=n_samples
        )

    if model_name=="laplace":
        return rng.laplace(
            loc=median,
            scale=std,
            size=n_samples
        )

    if model_name=="poisson":
        return rng.poisson(
            lam=max(mean,0.0),
            size=n_samples
        )

    if model_name=="rayleigh":
        return rng.rayleigh(
            scale=std,
            size=n_samples
        )

    if model_name=="gamma":
        shape=(
            mean**2/std**2
            if std>epsilon
            else 1.0
        )

        shape=max(
            shape,
            epsilon
        )

        scale=(
            std**2
            /max(
                abs(mean),
                epsilon
            )
        )

        return rng.gamma(
            shape=shape,
            scale=scale,
            size=n_samples
        )

    if model_name in (
        "studentt",
        "student_t",
        "t"
    ):
        return (
            rng.standard_t(
                df=2.0,
                size=n_samples
            )
            *std
            +mean
        )

    if model_name=="uniform":
        return rng.uniform(
            low=minimum,
            high=maximum,
            size=n_samples
        )

    raise ValueError(
        f"Modello non supportato: {model}"
    )

def apply_offset_correction(
    tep_agg,
    tep,
    timeMask,
    correctMode,
    oddSamples,
    EPOCHS,
    supported_models,
    rng=None
):
    import numpy as np
    from scipy.signal import resample

    if rng is None:
        rng=np.random.default_rng(42)

    tep_agg=np.asarray(
        tep_agg,
        dtype=float
    ).copy()

    tep=np.asarray(
        tep,
        dtype=float
    ).reshape(-1)

    times=np.asarray(
        EPOCHS.times,
        dtype=float
    )

    odd_samples_ms=(
        0.0
        if oddSamples is None or oddSamples is False
        else float(oddSamples)
    )

    margin_seconds=(
        odd_samples_ms
        /1000.0
    )

    pre_indices=np.flatnonzero(
        timeMask[0]
    )

    offset_indices=np.flatnonzero(
        timeMask[1]
    )

    if (
        pre_indices.size==0
        or offset_indices.size==0
    ):
        return tep_agg

    pre_end_time=float(
        times[
            pre_indices[-1]
        ]
    )

    offset_start_time=float(
        times[
            offset_indices[0]
        ]
    )

    offset_end_time=float(
        times[
            offset_indices[-1]
        ]
    )

    precorrectionMask=(
        (times>=float(times[pre_indices[0]]))
        &(times<=pre_end_time-margin_seconds)
    )

    if not np.any(
        precorrectionMask
    ):
        precorrectionMask=timeMask[
            0
        ].copy()

    correctionMask=(
        (times>=offset_start_time-margin_seconds)
        &(times<=offset_end_time+margin_seconds)
    )

    num_samples=int(
        np.sum(
            correctionMask
        )
    )

    pre_series=tep[
        precorrectionMask
    ]

    pre_series=pre_series[
        np.isfinite(
            pre_series
        )
    ]

    if (
        num_samples==0
        or pre_series.size==0
    ):
        return tep_agg

    mode=(
        str(correctMode).strip()
        if correctMode is not False
        else ""
    )

    mode_lower=mode.lower()

    supported_lower={
        str(item).lower()
        for item in supported_models
    }

    if mode_lower=="moving_average":
        window_size=max(
            1,
            int(
                round(
                    odd_samples_ms
                )
            )
        )

        if pre_series.size>=window_size:
            value=float(
                np.mean(
                    pre_series[
                        -window_size:
                    ]
                )
            )
        else:
            value=float(
                np.mean(
                    pre_series
                )
            )

        new_samples=np.full(
            num_samples,
            value,
            dtype=float
        )

    elif mode_lower=="median":
        new_samples=np.full(
            num_samples,
            float(
                np.median(
                    pre_series
                )
            ),
            dtype=float
        )

    elif mode_lower=="zeros":
        new_samples=np.zeros(
            num_samples,
            dtype=float
        )

    elif mode_lower=="resample":
        new_samples=resample(
            pre_series,
            num=num_samples
        )

    elif mode_lower in supported_lower:
        new_samples=generate_noise_from_distribution(
            time_series=pre_series,
            model=mode,
            n_samples=num_samples,
            rng=rng
        )

    else:
        raise ValueError(
            f"Metodo di correzione offset "
            f"non riconosciuto: {correctMode}"
        )

    tep_agg[
        correctionMask
    ]=new_samples

    return tep_agg
