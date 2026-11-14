"""
Script de prueba del servicio web de Puesta en Producción (FastAPI).
Permite verificar los tres casos de uso de MLOps sobre datos reales del Pozo 5:
  1. Caso 1: Predicción individual en tiempo real (POST /predecir)
  2. Caso 2: Predicción por lote / Batch (POST /predecir_lote)
  3. Monitoreo de Data Drift (K-S y PSI) y Concept Drift (POST /monitorear_drift)
"""

import json
from main import (
    estado_del_servicio,
    predecir_individual,
    predecir_por_lote,
    monitorear_lote_produccion,
    MedicionPozo,
    LoteMediciones,
)


def ejecutar_pruebas_locales():
    print("=" * 80)
    print("PRUEBA DEL SERVICIO DE PUESTA EN PRODUCCIÓN Y MONITOREO DE ROP")
    print("=" * 80)

    print("\n[SALUD DEL PIPELINE] Reglas congeladas del entrenamiento (/salud):")
    print(json.dumps(estado_del_servicio(), indent=2, ensure_ascii=False))

    # 1. Caso 1: Predicción individual (tiempo real / JSON)
    muestra_individual = MedicionPozo(
        measured_depth_m=3120.0,
        weight_on_bit_kkgf=8.5,
        standpipe_pressure_kpa=19500.0,
        surface_torque_knm=16.2,
        rotary_speed_rpm=155.0,
        mud_flow_in_lpm=2250.0,
        diameter_mm=215.9,
        hookload_kkgf=135.0,
        rop_real_m_h=28.4,
    )
    res_ind = predecir_individual(muestra_individual)
    print("\n[CASO 1] Predicción Individual en Tiempo Real (/predecir):")
    print(json.dumps(res_ind, indent=2, ensure_ascii=False))

    # 2. Caso 2: Predicción por lote (Batch) + Monitoreo de Drift
    lote_ejemplo = LoteMediciones(
        nombre_lote="Pozo_5_Lote_Profundo_Prueba",
        mediciones=[
            MedicionPozo(
                measured_depth_m=3300.0 + i * 25.0,
                weight_on_bit_kkgf=7.2 + i * 0.4,
                standpipe_pressure_kpa=20100.0 + i * 180.0,
                surface_torque_knm=15.0 + i * 0.5,
                rotary_speed_rpm=152.0,
                mud_flow_in_lpm=2180.0,
                diameter_mm=215.9,
                hookload_kkgf=142.0,
                rop_real_m_h=31.5,
            )
            for i in range(10)
        ],
    )
    res_lote = predecir_por_lote(lote_ejemplo)
    print("\n[CASO 2] Predicción por Lote / Batch (/predecir_lote):")
    print(json.dumps(res_lote, indent=2, ensure_ascii=False))

    res_drift = monitorear_lote_produccion(lote_ejemplo)
    print("\n[MONITOREO MLOps] Auditoría de Data Drift (K-S / PSI) y Concept Drift (/monitorear_drift):")
    print(json.dumps(res_drift, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ejecutar_pruebas_locales()
