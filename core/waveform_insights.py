def generate_waveform_insight(metrics: dict) -> str:
    if not metrics:
        return ""

    rms = metrics.get("rms", 0)
    cf = metrics.get("crest_factor", 0)
    kurt = metrics.get("kurtosis", 0)
    skew = metrics.get("skewness", 0)

    insights = []

    # Impactos / transitorios
    if kurt > 4:
        insights.append("Se detecta comportamiento impulsivo en la señal (kurtosis elevada), posible presencia de impactos o defectos localizados.")

    # Crest factor alto
    if cf > 3:
        insights.append("El factor de cresta es elevado, indicando presencia de picos transitorios sobre una base RMS baja.")

    # Sesgo
    if abs(skew) > 0.5:
        direction = "positivo" if skew > 0 else "negativo"
        insights.append(f"La señal presenta asimetría ({direction}), lo que puede indicar carga no balanceada o componente direccional.")

    # Señal limpia
    if not insights:
        insights.append("La señal presenta comportamiento estable, sin evidencia clara de impactos o asimetrías significativas.")

    return " ".join(insights)


def generate_batch_insights(metrics_dict: dict) -> dict:
    results = {}

    for name, metrics in metrics_dict.items():
        try:
            results[name] = generate_waveform_insight(metrics)
        except Exception:
            results[name] = ""

    return results
