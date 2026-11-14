import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import List, Optional, Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(f"HTTP {status_code}: {detail}")
    class HTMLResponse(str):
        pass
    def Field(default=None, **kwargs):
        return default
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in self.__class__.__dict__.items():
                if not k.startswith("_") and not callable(v):
                    setattr(self, k, v)
            for k, v in kwargs.items():
                setattr(self, k, v)
    class FastAPI:
        def __init__(self, **kwargs):
            self.meta = kwargs
        def get(self, *args, **kwargs):
            return lambda fn: fn
        def post(self, *args, **kwargs):
            return lambda fn: fn

# ==============================================================================
# CONFIGURACIÓN DE LA API DE PUESTA EN PRODUCCIÓN Y MONITOREO (FASTAPI)
# ==============================================================================
app = FastAPI(
    title="YPF ROP Predictor - API de Producción y Monitoreo MLOps",
    description=(
        "Servicio web productivo para la predicción de la Tasa de Penetración (Rate of Penetration m/h) "
        "en pozos de perforación del bloque Volve (Mar del Norte, Noruega). Empaqueta el preprocesamiento "
        "(StandardScaler / RobustScaler) y el estimador final dentro de un Pipeline sellado de scikit-learn, "
        "permitiendo predicción individual en tiempo real (/predecir), predicción por lote (/predecir_lote) "
        "y monitoreo continuo de Data Drift y Concept Drift (/monitorear_drift) con K-S y PSI."
    ),
    version="1.1.0",
)

FEATURES = [
    "Weight on Bit kkgf",
    "Average Standpipe Pressure kPa",
    "Average Surface Torque kN.m",
    "Average Rotary Speed rpm",
    "Mud Flow In L/min",
    "Diameter mm",
    "Average Hookload kkgf",
]

SHORT_LABELS = [
    "WOB (kkgf)",
    "Presión (kPa)",
    "Torque (kN·m)",
    "RPM",
    "Caudal (L/min)",
    "Diámetro (mm)",
    "Hookload (kkgf)",
]

FIELD_TO_COL = {
    "weight_on_bit_kkgf": "Weight on Bit kkgf",
    "standpipe_pressure_kpa": "Average Standpipe Pressure kPa",
    "surface_torque_knm": "Average Surface Torque kN.m",
    "rotary_speed_rpm": "Average Rotary Speed rpm",
    "mud_flow_in_lpm": "Mud Flow In L/min",
    "diameter_mm": "Diameter mm",
    "hookload_kkgf": "Average Hookload kkgf",
}

MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "modelo_rop_pipeline.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "modelo_rop_pipeline.joblib"),
]

BUNDLE: Dict[str, Any] = {}
for path in MODEL_PATHS:
    if os.path.exists(path):
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "pipeline_xgb" in loaded:
            BUNDLE = loaded
        else:
            BUNDLE = {"pipeline_xgb": loaded}
        break


def calcular_psi(referencia: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calcula el Population Stability Index (PSI) entre el entrenamiento y el lote productivo."""
    cuantiles = np.linspace(0, 100, bins + 1)
    cortes = np.unique(np.percentile(referencia, cuantiles))
    if len(cortes) < 3:
        return 0.0
    cortes[0] = -np.inf
    cortes[-1] = np.inf
    prop_ref = np.histogram(referencia, bins=cortes)[0] / len(referencia)
    prop_act = np.histogram(actual, bins=cortes)[0] / len(actual)
    prop_ref = np.clip(prop_ref, 1e-4, None)
    prop_act = np.clip(prop_act, 1e-4, None)
    return float(np.sum((prop_act - prop_ref) * np.log(prop_act / prop_ref)))


# ==============================================================================
# ESQUEMAS DE DATOS DE ENTRADA Y SALIDA (PYDANTIC / JSON)
# ==============================================================================
class MedicionPozo(BaseModel):
    measured_depth_m: Optional[float] = Field(
        3120.0, description="Profundidad medida referencial (Measured Depth m, informativa)"
    )
    weight_on_bit_kkgf: float = Field(8.5, description="Peso sobre la mecha (Weight on Bit kkgf)")
    standpipe_pressure_kpa: float = Field(19500.0, description="Presión de bombeo (Average Standpipe Pressure kPa)")
    surface_torque_knm: float = Field(16.2, description="Torque en superficie (Average Surface Torque kN.m)")
    rotary_speed_rpm: float = Field(155.0, description="Velocidad de rotación (Average Rotary Speed rpm)")
    mud_flow_in_lpm: float = Field(2250.0, description="Caudal de ingreso de lodo (Mud Flow In L/min)")
    diameter_mm: float = Field(215.9, description="Diámetro de mecha (Diameter mm, ej. 215.9 mm = 8.5 in)")
    hookload_kkgf: float = Field(135.0, description="Carga en el gancho (Average Hookload kkgf)")
    rop_real_m_h: Optional[float] = Field(
        None,
        description="(Opcional) ROP real medida en campo (Rate of Penetration m/h), usada en monitoreo",
    )

    def to_feature_dict(self) -> Dict[str, float]:
        return {col: float(getattr(self, field)) for field, col in FIELD_TO_COL.items()}


class LoteMediciones(BaseModel):
    nombre_lote: str = Field("Pozo_5_Lote_Produccion", description="Identificador del pozo o lote operativo")
    mediciones: List[MedicionPozo]


# ==============================================================================
# ENDPOINTS PRODUCTIVOS
# ==============================================================================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def pagina_inicio():
    """Interfaz web interactiva con gráficos dinámicos en vivo para Hugging Face Spaces y pruebas locales."""
    return """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YPF ROP Predictor — Servicio Web en Producción (FastAPI + MLOps)</title>
<style>
  :root {
    --bg:#0b1120; --card:#111827; --border:#1f2937;
    --accent:#38bdf8; --emerald:#10b981; --amber:#f59e0b; --rose:#f43f5e; --purple:#a855f7;
    --text:#f3f4f6; --muted:#9ca3af;
  }
  * { box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',system-ui,sans-serif; }
  body { background:var(--bg); color:var(--text); padding:22px 26px 48px; line-height:1.45; }
  .container { max-width:1440px; margin:0 auto; }
  .header {
    display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;
    gap:16px; padding-bottom:18px; border-bottom:1px solid var(--border); margin-bottom:20px;
  }
  .badge {
    background:rgba(16,185,129,0.15); color:var(--emerald);
    border:1px solid rgba(16,185,129,0.4); padding:5px 12px;
    border-radius:999px; font-size:12.5px; font-weight:700;
  }
  .btn-docs {
    background:var(--accent); color:#0f172a; text-decoration:none;
    padding:10px 16px; border-radius:8px; font-weight:700; font-size:13.5px;
  }
  .top-grid { display:grid; grid-template-columns:420px 1fr; gap:20px; margin-bottom:20px; }
  .charts-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }
  @media(max-width:1080px){ .top-grid, .charts-grid { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px 20px; }
  h2 { font-size:16.5px; margin-bottom:6px; color:var(--accent); }
  .sub { font-size:12.5px; color:var(--muted); margin-bottom:14px; }
  .presets { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px; }
  .preset-btn {
    background:#1e293b; color:#cbd5e1; border:1px solid #334155;
    padding:5px 10px; border-radius:6px; font-size:11.5px; font-weight:600; cursor:pointer;
  }
  .preset-btn:hover, .preset-btn.active { background:var(--accent); color:#0f172a; border-color:var(--accent); }
  .form-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .field-box { background:#030712; border:1px solid #1f2937; border-radius:8px; padding:8px 10px; }
  .field-head { display:flex; justify-content:space-between; font-size:11.5px; color:var(--muted); margin-bottom:4px; }
  .field-head b { color:#fff; font-size:12.5px; }
  input[type=range] { width:100%; accent-color:var(--accent); cursor:pointer; }
  input[type=number] {
    width:100%; padding:5px 8px; border-radius:5px; border:1px solid #374151;
    background:#090d16; color:#fff; font-size:13px; margin-top:4px;
  }
  .btn-main {
    margin-top:14px; width:100%; padding:11px; background:var(--emerald);
    color:#052e16; border:none; border-radius:8px; font-weight:800; font-size:14px; cursor:pointer;
  }
  .btn-secondary {
    padding:7px 12px; background:#1e293b; color:#e2e8f0; border:1px solid #334155;
    border-radius:6px; font-size:12px; font-weight:600; cursor:pointer;
  }
  .btn-secondary:hover { background:#334155; }
  .kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:14px; }
  @media(max-width:700px){ .kpi-row { grid-template-columns:1fr 1fr; } }
  .kpi { background:#030712; border:1px solid #1f2937; border-radius:8px; padding:10px; text-align:center; }
  .kpi .val { font-size:21px; font-weight:800; color:var(--accent); }
  .kpi .lbl { font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-top:2px; }
  canvas { width:100%; background:#060a14; border:1px solid #1e293b; border-radius:8px; display:block; }
  pre {
    background:#030712; border:1px solid #1f2937; padding:12px;
    border-radius:8px; overflow-x:auto; font-size:12px; color:#a7f3d0; max-height:250px;
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <span class="badge">● PIPELINE EN PRODUCCIÓN ACTIVO (FASTAPI + DOCKER)</span>
      <h1 style="margin-top:6px;font-size:24px;">Predicción de ROP y Monitoreo Dinámico en Tiempo Real — Pozos Noruega</h1>
      <p style="color:var(--muted);font-size:13.5px;">Mové cualquier sensor o elegí un escenario: los gráficos y el JSON se actualizan en vivo consultando el Pipeline empaquetado (<code>modelo_rop_pipeline.joblib</code>)</p>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
      <button class="btn-secondary" onclick="limpiarHistorial()">Reiniciar Historial</button>
      <a class="btn-docs" href="/docs" target="_blank">Abrir Swagger UI (/docs) →</a>
    </div>
  </div>

  <div class="top-grid">
    <!-- Panel Izquierdo: Controles Interactivos -->
    <div class="card">
      <h2>1. Parámetros Operativos de Entrada</h2>
      <div class="sub">Escenarios rápidos de prueba o ajuste fino con sliders:</div>
      <div class="presets">
        <button class="preset-btn active" onclick="cargarPreset('p5_std', this)">Pozo 5 Estándar (8.5")</button>
        <button class="preset-btn" onclick="cargarPreset('somero', this)">Pozo Somero Rápido (12.25")</button>
        <button class="preset-btn" onclick="cargarPreset('alta_energia', this)">Alta Energía Mecánica</button>
        <button class="preset-btn" onclick="cargarPreset('drift', this)">Simular Anomalía / Drift</button>
      </div>

      <div class="form-grid">
        <div class="field-box">
          <div class="field-head"><span>WOB (kkgf)</span><b id="lbl_wob">8.5</b></div>
          <input type="range" id="rng_wob" min="0.5" max="32" step="0.5" value="8.5" oninput="syncInput('wob', this.value)">
          <input type="number" id="wob" step="0.5" value="8.5" oninput="syncInput('wob', this.value)">
        </div>
        <div class="field-box">
          <div class="field-head"><span>Presión (kPa)</span><b id="lbl_spp">19500</b></div>
          <input type="range" id="rng_spp" min="5000" max="36000" step="250" value="19500" oninput="syncInput('spp', this.value)">
          <input type="number" id="spp" step="250" value="19500" oninput="syncInput('spp', this.value)">
        </div>
        <div class="field-box">
          <div class="field-head"><span>Torque (kN·m)</span><b id="lbl_trq">16.2</b></div>
          <input type="range" id="rng_trq" min="0.5" max="38" step="0.5" value="16.2" oninput="syncInput('trq', this.value)">
          <input type="number" id="trq" step="0.5" value="16.2" oninput="syncInput('trq', this.value)">
        </div>
        <div class="field-box">
          <div class="field-head"><span>Rotación (RPM)</span><b id="lbl_rpm">155</b></div>
          <input type="range" id="rng_rpm" min="30" max="240" step="5" value="155" oninput="syncInput('rpm', this.value)">
          <input type="number" id="rpm" step="5" value="155" oninput="syncInput('rpm', this.value)">
        </div>
        <div class="field-box">
          <div class="field-head"><span>Caudal (L/min)</span><b id="lbl_flow">2250</b></div>
          <input type="range" id="rng_flow" min="800" max="4800" step="50" value="2250" oninput="syncInput('flow', this.value)">
          <input type="number" id="flow" step="50" value="2250" oninput="syncInput('flow', this.value)">
        </div>
        <div class="field-box">
          <div class="field-head"><span>Diámetro (mm)</span><b id="lbl_diam">215.9</b></div>
          <input type="range" id="rng_diam" min="215.9" max="444.5" step="1.0" value="215.9" oninput="syncInput('diam', this.value)">
          <input type="number" id="diam" step="0.1" value="215.9" oninput="syncInput('diam', this.value)">
        </div>
        <div class="field-box">
          <div class="field-head"><span>Hookload (kkgf)</span><b id="lbl_hkld">135.0</b></div>
          <input type="range" id="rng_hkld" min="75" max="200" step="1" value="135.0" oninput="syncInput('hkld', this.value)">
          <input type="number" id="hkld" step="1" value="135.0" oninput="syncInput('hkld', this.value)">
        </div>
        <div class="field-box">
          <div class="field-head"><span>Profundidad (m)</span><b id="lbl_md">3120</b></div>
          <input type="range" id="rng_md" min="450" max="3800" step="15" value="3120" oninput="syncInput('md', this.value)">
          <input type="number" id="md" step="15" value="3120" oninput="syncInput('md', this.value)">
        </div>
      </div>

      <button class="btn-main" onclick="predecir(true)">⚡ Ejecutar Predicción y Registrar Punto en el Gráfico</button>
    </div>

    <!-- Panel Derecho: KPIs + Gráfico 1 (Comparativa de Modelos y Sensibilidad WOB) + Gráfico 2 (Z-scores Drift) -->
    <div class="card">
      <h2>2. Respuesta Dinámica del Pipeline en Producción</h2>
      <div class="sub">Resultados devueltos en tiempo real por el endpoint <code>POST /predecir</code>:</div>
      <div class="kpi-row">
        <div class="kpi"><div class="val" id="rop_xgb">--</div><div class="lbl">Pipeline XGBoost (m/h)</div></div>
        <div class="kpi"><div class="val" id="rop_rf" style="color:var(--emerald);">--</div><div class="lbl">Pipeline Random Forest</div></div>
        <div class="kpi"><div class="val" id="rop_mlp" style="color:var(--amber);">--</div><div class="lbl">Pipeline Red Neuronal</div></div>
        <div class="kpi"><div class="val" id="regimen" style="font-size:12.5px;color:#e2e8f0;padding-top:4px;">--</div><div class="lbl">Régimen K-Means</div></div>
      </div>

      <div class="charts-grid" style="margin-bottom:0;">
        <div>
          <div style="font-size:12.5px;font-weight:700;color:#cbd5e1;margin-bottom:6px;">
            Gráfico A: Comparación de Modelos y Curva de Respuesta ROP vs. WOB
          </div>
          <canvas id="canvasModelos" width="460" height="270"></canvas>
        </div>
        <div>
          <div style="font-size:12.5px;font-weight:700;color:#cbd5e1;margin-bottom:6px;">
            Gráfico B: Monitoreo de Sensores en Vivo (Z-Score vs. Entrenamiento)
          </div>
          <canvas id="canvasZScores" width="460" height="270"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Fila Inferior: Historial Dinámico de Predicciones + Respuesta JSON -->
  <div class="charts-grid">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div>
          <h2>3. Registro Gráfico en Vivo de tus Predicciones (Historial Paso a Paso)</h2>
          <div class="sub" style="margin-bottom:0;">Cada cambio de parámetros agrega un nuevo punto a la curva para comparar cómo responden los 3 modelos en tiempo real.</div>
        </div>
      </div>
      <canvas id="canvasHistorial" width="660" height="250"></canvas>
    </div>

    <div class="card">
      <h2>4. Payload JSON Devuelto por FastAPI (POST /predecir)</h2>
      <div class="sub">Estructura exacta consumible por cualquier aplicación externa, dashboard o script en Python:</div>
      <pre id="json_out">Consultando endpoint POST /predecir...</pre>
    </div>
  </div>
</div>

<script>
const HISTORIAL = [];
let debounceTimer = null;

function syncInput(id, val) {
  document.getElementById(id).value = val;
  document.getElementById('rng_' + id).value = val;
  document.getElementById('lbl_' + id).textContent = val;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => predecir(true), 90);
}

function cargarPreset(nombre, btn) {
  if (btn) {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  const presets = {
    p5_std: { wob: 8.5, spp: 19500, trq: 16.2, rpm: 155, flow: 2250, diam: 215.9, hkld: 135, md: 3120 },
    somero: { wob: 5.5, spp: 11200, trq: 4.8, rpm: 138, flow: 3450, diam: 311.15, hkld: 96, md: 820 },
    alta_energia: { wob: 19.0, spp: 24200, trq: 25.0, rpm: 185, flow: 2380, diam: 215.9, hkld: 148, md: 3280 },
    drift: { wob: 30.5, spp: 34500, trq: 36.5, rpm: 65, flow: 1150, diam: 215.9, hkld: 192, md: 3460 }
  };
  const p = presets[nombre] || presets.p5_std;
  Object.keys(p).forEach(k => {
    document.getElementById(k).value = p[k];
    document.getElementById('rng_' + k).value = p[k];
    document.getElementById('lbl_' + k).textContent = p[k];
  });
  predecir(true);
}

function limpiarHistorial() {
  HISTORIAL.length = 0;
  predecir(true);
}

async function predecir(agregarAlHistorial = true) {
  const payload = {
    measured_depth_m: parseFloat(document.getElementById('md').value),
    weight_on_bit_kkgf: parseFloat(document.getElementById('wob').value),
    standpipe_pressure_kpa: parseFloat(document.getElementById('spp').value),
    surface_torque_knm: parseFloat(document.getElementById('trq').value),
    rotary_speed_rpm: parseFloat(document.getElementById('rpm').value),
    mud_flow_in_lpm: parseFloat(document.getElementById('flow').value),
    diameter_mm: parseFloat(document.getElementById('diam').value),
    hookload_kkgf: parseFloat(document.getElementById('hkld').value)
  };
  const res = await fetch('/predecir', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();

  document.getElementById('rop_xgb').textContent = data.rop_predicha_principal_m_h.toFixed(2) + ' m/h';
  document.getElementById('rop_rf').textContent = data.rop_predicha_rf_m_h.toFixed(2) + ' m/h';
  document.getElementById('rop_mlp').textContent = (data.rop_predicha_mlp_m_h ?? 0).toFixed(2) + ' m/h';
  document.getElementById('regimen').textContent = data.regimen_operativo_kmeans || 'Normal';
  document.getElementById('json_out').textContent = JSON.stringify(data, null, 2);

  if (agregarAlHistorial) {
    HISTORIAL.push({
      paso: HISTORIAL.length + 1,
      md: payload.measured_depth_m,
      wob: payload.weight_on_bit_kkgf,
      xgb: data.rop_predicha_principal_m_h,
      rf: data.rop_predicha_rf_m_h,
      mlp: data.rop_predicha_mlp_m_h || data.rop_predicha_principal_m_h
    });
    if (HISTORIAL.length > 25) HISTORIAL.shift();
  }

  dibujarGraficoModelos(data, payload.weight_on_bit_kkgf);
  dibujarGraficoZScores(data.z_scores_sensores || []);
  dibujarGraficoHistorial();
}

function dibujarGraficoModelos(data, wobActual) {
  const c = document.getElementById('canvasModelos');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);

  // Mitad izquierda: Barras comparativas de los 3 modelos
  const modelos = [
    { nombre: 'XGBoost', val: data.rop_predicha_principal_m_h, col: '#38bdf8' },
    { nombre: 'RandomF', val: data.rop_predicha_rf_m_h, col: '#10b981' },
    { nombre: 'Red MLP', val: data.rop_predicha_mlp_m_h || 0, col: '#f59e0b' }
  ];
  const maxVal = Math.max(55, ...modelos.map(m => m.val * 1.15));
  const padT = 32, padB = 34, ph = H - padT - padB;

  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 11.5px Segoe UI';
  ctx.fillText('Predicción por Modelo (m/h)', 14, 18);

  modelos.forEach((m, i) => {
    const bx = 22 + i * 56;
    const bh = Math.max(4, (m.val / maxVal) * ph);
    const by = padT + ph - bh;
    ctx.fillStyle = m.col;
    ctx.beginPath();
    ctx.roundRect(bx, by, 38, bh, 5);
    ctx.fill();

    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 11.5px Segoe UI';
    ctx.fillText(m.val.toFixed(1), bx + 2, by - 5);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px Segoe UI';
    ctx.fillText(m.nombre, bx - 2, H - 12);
  });

  // Mitad derecha: Curva de sensibilidad ROP vs WOB en las condiciones actuales
  const curva = data.curva_sensibilidad_wob || [];
  if (curva.length > 1) {
    const x0 = 215, x1 = W - 16, cw = x1 - x0;
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 11.5px Segoe UI';
    ctx.fillText('Curva Física ROP vs. WOB actual', x0, 18);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const gy = padT + (ph * g / 3);
      ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x1, gy); ctx.stroke();
    }

    const wMin = curva[0].wob, wMax = curva[curva.length - 1].wob;
    // Curva XGBoost
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    curva.forEach((pt, idx) => {
      const px = x0 + ((pt.wob - wMin) / (wMax - wMin)) * cw;
      const py = padT + ph - (pt.rop_xgb / maxVal) * ph;
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Curva MLP
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2.0;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    curva.forEach((pt, idx) => {
      const px = x0 + ((pt.wob - wMin) / (wMax - wMin)) * cw;
      const py = padT + ph - (pt.rop_mlp / maxVal) * ph;
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Punto operativo actual
    const pxAct = x0 + Math.min(1, Math.max(0, (wobActual - wMin) / (wMax - wMin))) * cw;
    const pyAct = padT + ph - (data.rop_predicha_principal_m_h / maxVal) * ph;
    ctx.fillStyle = '#f43f5e';
    ctx.beginPath(); ctx.arc(pxAct, pyAct, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10.5px Segoe UI';
    ctx.fillText('WOB ' + wMin + ' a ' + wMax + ' kkgf (Punto rojo = WOB actual)', x0, H - 12);
  }
}

function dibujarGraficoZScores(zList) {
  const c = document.getElementById('canvasZScores');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 112, padR = 38, padT = 24, padB = 26;
  const pw = W - padL - padR, ph = H - padT - padB;
  const zMax = 4.5;
  const zeroX = padL + pw / 2;

  // Líneas de umbral (-3, -2, 0, +2, +3)
  [-3, -2, 0, 2, 3].forEach(zv => {
    const gx = zeroX + (zv / zMax) * (pw / 2);
    ctx.strokeStyle = zv === 0 ? '#64748b' : (Math.abs(zv) === 3 ? 'rgba(244,63,94,0.45)' : '#1e293b');
    ctx.setLineDash(Math.abs(zv) === 3 ? [4, 3] : []);
    ctx.beginPath(); ctx.moveTo(gx, padT); ctx.lineTo(gx, H - padB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Segoe UI';
    ctx.fillText((zv > 0 ? '+' : '') + zv + 'σ', gx - 8, H - 8);
  });

  const rowH = ph / Math.max(1, zList.length);
  zList.forEach((item, i) => {
    const cy = padT + i * rowH + rowH * 0.5;
    const zClamped = Math.max(-zMax, Math.min(zMax, item.z_score));
    const bx = zeroX + (zClamped / zMax) * (pw / 2);
    const absZ = Math.abs(item.z_score);
    const barColor = absZ > 3.0 ? '#f43f5e' : (absZ > 2.0 ? '#f59e0b' : '#10b981');

    ctx.fillStyle = '#cbd5e1';
    ctx.font = '11.5px Segoe UI';
    ctx.fillText(item.etiqueta, 10, cy + 4);

    ctx.fillStyle = barColor;
    const leftX = Math.min(zeroX, bx);
    const widthX = Math.max(3, Math.abs(bx - zeroX));
    ctx.fillRect(leftX, cy - 7, widthX, 14);

    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 10.5px Segoe UI';
    ctx.fillText((item.z_score >= 0 ? '+' : '') + item.z_score.toFixed(2) + 'σ', W - padR + 4, cy + 4);
  });
}

function dibujarGraficoHistorial() {
  const c = document.getElementById('canvasHistorial');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 48, padR = 24, padT = 30, padB = 34;
  const pw = W - padL - padR, ph = H - padT - padB;

  const allVals = HISTORIAL.flatMap(h => [h.xgb, h.rf, h.mlp]);
  const yMin = Math.max(0, Math.min(10, ...allVals) - 4);
  const yMax = Math.max(50, ...allVals) + 5;

  ctx.strokeStyle = '#1e293b';
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px Segoe UI';
  for (let i = 0; i <= 4; i++) {
    const gy = padT + (ph * i / 4);
    const yv = yMax - (yMax - yMin) * (i / 4);
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(W - padR, gy); ctx.stroke();
    ctx.fillText(yv.toFixed(0) + ' m/h', 6, gy + 4);
  }

  if (HISTORIAL.length === 0) return;

  function trazarSerie(key, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.3;
    ctx.beginPath();
    HISTORIAL.forEach((h, idx) => {
      const px = padL + (HISTORIAL.length === 1 ? pw / 2 : (idx / (HISTORIAL.length - 1)) * pw);
      const py = padT + ((yMax - h[key]) / (yMax - yMin)) * ph;
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    HISTORIAL.forEach((h, idx) => {
      const px = padL + (HISTORIAL.length === 1 ? pw / 2 : (idx / (HISTORIAL.length - 1)) * pw);
      const py = padT + ((yMax - h[key]) / (yMax - yMin)) * ph;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(px, py, 3.8, 0, Math.PI * 2); ctx.fill();
    });
  }

  trazarSerie('xgb', '#38bdf8');
  trazarSerie('rf', '#10b981');
  trazarSerie('mlp', '#f59e0b');

  // Leyenda superior
  ctx.font = 'bold 11.5px Segoe UI';
  ctx.fillStyle = '#38bdf8'; ctx.fillText('● XGBoost', padL, 18);
  ctx.fillStyle = '#10b981'; ctx.fillText('● Random Forest', padL + 95, 18);
  ctx.fillStyle = '#f59e0b'; ctx.fillText('● Red Neuronal MLP', padL + 220, 18);
  ctx.fillStyle = '#94a3b8'; ctx.fillText('(' + HISTORIAL.length + ' simulaciones registradas)', padL + 370, 18);
}

window.onload = () => {
  cargarPreset('somero');
  setTimeout(() => cargarPreset('alta_energia'), 120);
  setTimeout(() => cargarPreset('p5_std'), 240);
};
</script>
</body>
</html>"""


@app.get("/salud")
def estado_del_servicio():
    """Verifica la salud del pipeline en producción y expone las estadísticas congeladas del entrenamiento."""
    if not BUNDLE:
        raise HTTPException(status_code=503, detail="El archivo modelo_rop_pipeline.joblib no fue encontrado.")
    pipe_mlp = BUNDLE.get("pipeline_mlp")
    scaler = pipe_mlp.named_steps.get("scaler") if pipe_mlp is not None else None
    return {
        "estado": "OPERATIVO",
        "version_modelo": BUNDLE.get("version", "1.1.0"),
        "modelos_empaquetados": [k for k in BUNDLE.keys() if k.startswith("pipeline_")],
        "features_operativas": FEATURES,
        "reglas_congeladas_standard_scaler": {
            col: {
                "media_entrenamiento": round(float(scaler.mean_[i]), 3),
                "desvio_entrenamiento": round(float(scaler.scale_[i]), 3),
            }
            for i, col in enumerate(FEATURES)
        }
        if scaler is not None
        else {},
    }


@app.post("/predecir")
def predecir_individual(medicion: MedicionPozo):
    """
    Caso 1: Predicción individual en tiempo real (API Web).
    Recibe un JSON de una única muestra de perforación, aplica automáticamente las transformaciones
    congeladas del Pipeline y devuelve el ROP estimado en m/h, el régimen K-Means, los z-scores
    por sensor y la curva de respuesta dinámica ROP vs. WOB.
    """
    if not BUNDLE:
        raise HTTPException(status_code=503, detail="Pipeline no disponible.")

    base_dict = medicion.to_feature_dict()
    df_nuevo = pd.DataFrame([base_dict])[FEATURES]

    pipe_main = BUNDLE.get("pipeline_xgb") or BUNDLE.get("pipeline_rf")
    pred_main = float(pipe_main.predict(df_nuevo)[0])
    pred_rf = float(BUNDLE["pipeline_rf"].predict(df_nuevo)[0]) if "pipeline_rf" in BUNDLE else pred_main
    pred_mlp = float(BUNDLE["pipeline_mlp"].predict(df_nuevo)[0]) if "pipeline_mlp" in BUNDLE else None

    regimen_txt = "Régimen Operativo Estándar"
    if "pipeline_kmeans" in BUNDLE and "nombres_clusters" in BUNDLE:
        cluster_id = int(BUNDLE["pipeline_kmeans"].predict(df_nuevo)[0])
        regimen_txt = BUNDLE["nombres_clusters"].get(cluster_id, f"Cluster {cluster_id}")

    # Monitoreo individual de z-score respecto a la distribución de entrenamiento
    variables_fuera_rango = []
    z_scores_sensores = []
    if "pipeline_mlp" in BUNDLE:
        scaler = BUNDLE["pipeline_mlp"].named_steps.get("scaler")
        if scaler is not None:
            z_vals = (df_nuevo.values[0] - scaler.mean_) / scaler.scale_
            for col, lbl, z in zip(FEATURES, SHORT_LABELS, z_vals):
                zf = round(float(z), 2)
                z_scores_sensores.append({"variable": col, "etiqueta": lbl, "z_score": zf})
                if abs(zf) > 3.0:
                    variables_fuera_rango.append({"variable": col, "z_score": zf})

    # Curva rápida de sensibilidad ROP vs WOB para las condiciones operativas actuales
    wob_grid = np.linspace(1.0, 28.0, 12)
    df_curva = pd.DataFrame([
        {**base_dict, "Weight on Bit kkgf": float(w)} for w in wob_grid
    ])[FEATURES]
    curva_xgb = pipe_main.predict(df_curva)
    curva_mlp = BUNDLE["pipeline_mlp"].predict(df_curva) if "pipeline_mlp" in BUNDLE else curva_xgb
    curva_sensibilidad = [
        {"wob": round(float(w), 1), "rop_xgb": round(float(rx), 2), "rop_mlp": round(float(rm), 2)}
        for w, rx, rm in zip(wob_grid, curva_xgb, curva_mlp)
    ]

    respuesta = {
        "rop_predicha_principal_m_h": round(pred_main, 2),
        "rop_predicha_rf_m_h": round(pred_rf, 2),
        "rop_predicha_mlp_m_h": round(pred_mlp, 2) if pred_mlp is not None else None,
        "regimen_operativo_kmeans": regimen_txt,
        "alerta_operativa": (
            "ATENCIÓN: Lectura de sensores fuera del rango habitual (|z| > 3)"
            if variables_fuera_rango
            else "OK: Parámetros dentro de la distribución de entrenamiento"
        ),
        "variables_atipicas": variables_fuera_rango,
        "z_scores_sensores": z_scores_sensores,
        "curva_sensibilidad_wob": curva_sensibilidad,
    }
    if medicion.rop_real_m_h is not None:
        respuesta["rop_real_m_h"] = round(float(medicion.rop_real_m_h), 2)
        respuesta["error_absoluto_m_h"] = round(abs(float(medicion.rop_real_m_h) - pred_main), 2)
    return respuesta


@app.post("/predecir_lote")
def predecir_por_lote(lote: LoteMediciones):
    """
    Caso 2: Predicción por lote (Batch / conjunto de datos).
    Procesa un lote completo de registros nuevos de perforación en una sola llamada vectorizada.
    """
    if not BUNDLE:
        raise HTTPException(status_code=503, detail="Pipeline no disponible.")
    if not lote.mediciones:
        raise HTTPException(status_code=400, detail="El lote de mediciones está vacío.")

    df_lote = pd.DataFrame([m.to_feature_dict() for m in lote.mediciones])[FEATURES]
    pipe_main = BUNDLE.get("pipeline_xgb") or BUNDLE.get("pipeline_rf")
    preds_main = pipe_main.predict(df_lote)
    preds_mlp = BUNDLE["pipeline_mlp"].predict(df_lote) if "pipeline_mlp" in BUNDLE else preds_main

    return {
        "nombre_lote": lote.nombre_lote,
        "cantidad_registros": len(df_lote),
        "resumen_predicciones_principal": {
            "rop_media_m_h": round(float(np.mean(preds_main)), 2),
            "rop_mediana_m_h": round(float(np.median(preds_main)), 2),
            "rop_min_m_h": round(float(np.min(preds_main)), 2),
            "rop_max_m_h": round(float(np.max(preds_main)), 2),
        },
        "resumen_predicciones_mlp": {
            "rop_media_m_h": round(float(np.mean(preds_mlp)), 2),
            "rop_min_m_h": round(float(np.min(preds_mlp)), 2),
            "rop_max_m_h": round(float(np.max(preds_mlp)), 2),
        },
        "predicciones_m_h": [round(float(x), 2) for x in preds_main],
    }


@app.post("/monitorear_drift")
def monitorear_lote_produccion(lote: LoteMediciones):
    """
    Monitoreo de Datos Entrantes (Data Drift con K-S y PSI) y Desempeño del Modelo (Concept Drift).
    Compara las distribuciones del lote productivo frente a la muestra de referencia del entrenamiento.
    """
    if not BUNDLE or "referencia_train" not in BUNDLE:
        raise HTTPException(status_code=503, detail="Datos de referencia no disponibles en el bundle.")
    if len(lote.mediciones) < 5:
        raise HTTPException(status_code=400, detail="Se requieren al menos 5 mediciones para evaluar drift estadístico.")

    df_ref = pd.DataFrame(BUNDLE["referencia_train"])[FEATURES]
    df_lote = pd.DataFrame([m.to_feature_dict() for m in lote.mediciones])[FEATURES]

    reporte_variables = []
    vars_con_ks = 0
    vars_con_psi_severo = 0

    for col in FEATURES:
        ref_vals = df_ref[col].values
        act_vals = df_lote[col].values
        ks_stat, p_val = ks_2samp(ref_vals, act_vals)
        psi_val = calcular_psi(ref_vals, act_vals)

        hay_ks = bool(p_val < 0.05)
        if hay_ks:
            vars_con_ks += 1
        if psi_val >= 0.25:
            vars_con_psi_severo += 1

        estado_psi = (
            "ESTABLE (<0.10)"
            if psi_val < 0.10
            else ("CAMBIO MODERADO (0.10-0.25)" if psi_val < 0.25 else "DRIFT SEVERO (>=0.25)")
        )
        reporte_variables.append(
            {
                "variable": col,
                "media_train": round(float(np.mean(ref_vals)), 2),
                "media_lote": round(float(np.mean(act_vals)), 2),
                "ks_stat": round(float(ks_stat), 3),
                "ks_p_valor": round(float(p_val), 5),
                "alerta_ks": hay_ks,
                "psi": round(float(psi_val), 3),
                "diagnostico_psi": estado_psi,
            }
        )

    valores_reales = [m.rop_real_m_h for m in lote.mediciones if m.rop_real_m_h is not None]
    desempeno = None
    if len(valores_reales) == len(lote.mediciones):
        y_true = np.array(valores_reales, dtype=float)
        pipe_main = BUNDLE.get("pipeline_xgb") or BUNDLE.get("pipeline_rf")
        y_pred = pipe_main.predict(df_lote)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        desempeno = {
            "r2_lote": round(r2, 4),
            "mae_m_h": round(mae, 3),
            "rmse_m_h": round(rmse, 3),
            "concept_drift_detectado": bool(mae > 13.5),
        }

    semaforo = "ESTABLE"
    recomendacion = "Mantener la versión actual del pipeline en producción."
    if vars_con_psi_severo >= 3 or (desempeno and desempeno["concept_drift_detectado"]):
        semaforo = "DRIFT CRÍTICO - REENTRENAR MODELO"
        recomendacion = (
            "Se detectó deriva severa en las variables de entrada (PSI >= 0.25) y/o degradación de desempeño. "
            "Ejecutar actualización incremental (partial_fit) o reentrenamiento periódico del Pipeline."
        )
    elif vars_con_ks >= 3 or vars_con_psi_severo >= 1:
        semaforo = "ALERTA MODERADA"
        recomendacion = "Monitorear de cerca el próximo tramo de perforación y auditar calibración de sensores."

    return {
        "nombre_lote": lote.nombre_lote,
        "cantidad_registros": len(df_lote),
        "semaforo_monitoreo": semaforo,
        "recomendacion_mlops": recomendacion,
        "variables_con_ks_significativo": f"{vars_con_ks}/{len(FEATURES)}",
        "variables_con_psi_severo": f"{vars_con_psi_severo}/{len(FEATURES)}",
        "desempeno_lote": desempeno,
        "detalle_data_drift_por_variable": reporte_variables,
    }
