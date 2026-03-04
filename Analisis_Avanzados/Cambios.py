"""
=============================================================================
ESTUDIO CRONOBIOLOGICO — ANALISIS DE DERIVA BIOMETRICA v2
=============================================================================
Mejoras sobre la version original:
  · 8 metricas analizadas (vs 4 anteriores)
  · Coeficiente de Solapamiento OVL entre distribuciones
  · Coeficiente de Convergencia lambda por metrica
  · Violin + Box plots (se ve la FORMA de la distribucion, no solo la caja)
  · Radar comparativo de los tres perfiles
  · Heatmap de estadisticos por metrica
  · Resumen de convergencia ordenado por impacto
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob, warnings
from scipy.stats import mannwhitneyu, gaussian_kde
from math import sqrt

try:
    import matplotlib; matplotlib.use('TkAgg')
except: pass

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURACION ANONIMIZADA
# =============================================================================
ARCHIVO_SUJETO_REAL      = "victor"
ARCHIVO_SUJETO_CONFUSION = "leo"
CODIGO_REAL              = "P11"
CODIGO_CONFUSION         = "P5"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Estudio_Cronobiologico_Anonimo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# Nombres SIN tildes (evita encoding issues en Windows)
COND_MAN = f"{CODIGO_REAL} (Manana/Control)"
COND_TAR = f"{CODIGO_REAL} (Tarde/Fatiga)"
COND_CON = f"{CODIGO_CONFUSION} (Sujeto Similar)"
ORDEN    = [COND_MAN, COND_TAR, COND_CON]
PALETTE  = {COND_MAN: "#2ecc71", COND_TAR: "#e74c3c", COND_CON: "#95a5a6"}

METRICAS = {
    'Pupil_Mean'      : 'Diametro Pupilar Prom. (px)',
    'Pupil_CV'        : 'Coef. Variacion Pupilar',
    'Pupil_Vel_Max'   : 'Vel. Max. Pupila (px/s)',
    'Vel_Mean'        : 'Velocidad Sacadica Prom.',
    'Jerk_Mean'       : 'Jerk Ocular Medio',
    'Jerk_Max'        : 'Jerk Ocular Maximo',
    'Main_Seq_Slope'  : 'Pendiente Main Sequence',
    'Fractal_Dim'     : 'Dimension Fractal (HFD)',
}

# =============================================================================
#  FUNCIONES
# =============================================================================

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    pooled = sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else np.nan


def ovl_coefficient(x, y, n=500):
    """
    Coeficiente de Solapamiento (OVL).
    Integral del minimo de las dos KDE sobre el soporte comun.
    OVL=1 -> identicas | OVL=0 -> sin solapamiento.
    """
    if len(x) < 5 or len(y) < 5: return np.nan
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    grid   = np.linspace(lo, hi, n)
    try:
        k1 = gaussian_kde(x)(grid)
        k2 = gaussian_kde(y)(grid)
        return float(np.clip(np.trapz(np.minimum(k1, k2), grid), 0, 1))
    except: return np.nan


def lambda_conv(mu_man, mu_tar, mu_con):
    """
    lambda = (mu_tarde - mu_manana) / (mu_confusion - mu_manana)
    Fraccion del camino hacia P5 que recorre P11 con la fatiga.
    lambda=0: sin cambio | lambda=1: convergencia total.
    """
    gap = mu_con - mu_man
    return float((mu_tar - mu_man) / gap) if abs(gap) > 1e-9 else np.nan


def cohen_label(d):
    if np.isnan(d): return "N/A"
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "pequeno"
    if a < 0.8: return "mediano"
    return "GRANDE"


# =============================================================================
#  1. CARGA DE DATOS
# =============================================================================

def cargar_datos():
    print("=" * 70)
    print(f"  ESTUDIO CRONOBIOLOGICO: {CODIGO_REAL} (AM) vs {CODIGO_REAL} (PM)")
    print("=" * 70)

    files = glob.glob(INPUT_PATH, recursive=True)
    dfs   = []
    print("Clasificando archivos...")

    for f in files:
        fn = os.path.basename(f).lower()
        try:
            df = pd.read_csv(f)
            if 'Pupil_Mean' in df.columns:
                df = df[df['Pupil_Mean'] > 0]

            if   "test" in fn:                      df['Condicion'] = COND_TAR
            elif ARCHIVO_SUJETO_REAL      in fn:    df['Condicion'] = COND_MAN
            elif ARCHIVO_SUJETO_CONFUSION in fn:    df['Condicion'] = COND_CON
            else:                                   continue

            dfs.append(df)
        except Exception as e:
            print(f"  Error en {f}: {e}")

    if not dfs:
        print("No se encontraron datos.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Dataset: {len(df)} muestras")
    for c in ORDEN:
        print(f"  {c}: {(df['Condicion']==c).sum()} muestras")
    return df


# =============================================================================
#  2. ANALISIS ESTADISTICO
# =============================================================================

def analizar_metricas(df):
    resultados = []
    print("\n" + "-" * 60)
    print("  ANALISIS DE DERIVA BIOMETRICA")
    print("-" * 60)

    for metrica, etiqueta in METRICAS.items():
        if metrica not in df.columns: continue

        x_man = df[df['Condicion'] == COND_MAN][metrica].dropna().values
        x_tar = df[df['Condicion'] == COND_TAR][metrica].dropna().values
        x_con = df[df['Condicion'] == COND_CON][metrica].dropna().values

        if len(x_man) == 0 or len(x_tar) == 0: continue

        mu_man = np.mean(x_man); sd_man = np.std(x_man, ddof=1)
        mu_tar = np.mean(x_tar); sd_tar = np.std(x_tar, ddof=1)
        mu_con = np.mean(x_con) if len(x_con) > 0 else np.nan
        sd_con = np.std(x_con, ddof=1) if len(x_con) > 0 else np.nan

        pct_mt = (mu_tar - mu_man) / mu_man * 100 if mu_man != 0 else np.nan

        _, p_mt = mannwhitneyu(x_man, x_tar, alternative='two-sided')
        d_mt    = cohen_d(x_man, x_tar)
        ovl_mt  = ovl_coefficient(x_man, x_tar)

        p_tc = d_tc = ovl_tc = np.nan
        if len(x_con) > 0:
            _, p_tc = mannwhitneyu(x_tar, x_con, alternative='two-sided')
            d_tc    = cohen_d(x_tar, x_con)
            ovl_tc  = ovl_coefficient(x_tar, x_con)

        lam = lambda_conv(mu_man, mu_tar, mu_con) if not np.isnan(mu_con) else np.nan

        resultados.append({
            'Metrica'      : metrica, 'Etiqueta': etiqueta,
            'mu_man': mu_man, 'sd_man': sd_man,
            'mu_tar': mu_tar, 'sd_tar': sd_tar,
            'mu_con': mu_con, 'sd_con': sd_con,
            'Pct_cambio_MT': pct_mt,
            'p_MT' : p_mt,  'd_MT': d_mt,  'OVL_MT': ovl_mt,
            'p_TC' : p_tc,  'd_TC': d_tc,  'OVL_TC': ovl_tc,
            'Lambda': lam,
        })

        print(f"\n  {metrica}")
        print(f"    Manana : {mu_man:.3f} +- {sd_man:.3f}")
        print(f"    Tarde  : {mu_tar:.3f} +- {sd_tar:.3f}   Delta%={pct_mt:+.2f}%")
        if not np.isnan(mu_con):
            print(f"    {CODIGO_CONFUSION}     : {mu_con:.3f} +- {sd_con:.3f}")
        print(f"    M->T   d={d_mt:.3f} [{cohen_label(d_mt)}]  p={p_mt:.4f}  OVL={ovl_mt:.3f}")
        if not np.isnan(d_tc):
            print(f"    T->P5  d={d_tc:.3f} [{cohen_label(d_tc)}]  p={p_tc:.4f}  OVL={ovl_tc:.3f}")
        if not np.isnan(lam):
            print(f"    Lambda = {lam:.3f}  ({lam*100:.1f}% del camino hacia {CODIGO_CONFUSION})")

    df_res = pd.DataFrame(resultados)
    ruta   = os.path.join(OUTPUT_DIR, "Resultados_Estadisticos_Deriva_v2.csv")
    df_res.to_csv(ruta, index=False)
    print(f"\n  CSV guardado: {ruta}")
    return df_res


# =============================================================================
#  3. FIGURA 1 — VIOLIN + BOX (reemplaza el boxplot original)
# =============================================================================

def figura_violin_box(df, df_res):
    metricas_disp = [m for m in METRICAS if m in df.columns]
    n = len(metricas_disp)
    cols, rows = 4, (n + 3) // 4

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    fig.suptitle(
        f'Deriva Biometrica: {CODIGO_REAL} (Manana vs Tarde) vs {CODIGO_CONFUSION}\n'
        'Verde=Control  Rojo=Fatiga  Gris=Sujeto similar',
        fontsize=14, fontweight='bold'
    )

    orden_disp = [c for c in ORDEN if c in df['Condicion'].unique()]

    for i, metrica in enumerate(metricas_disp):
        ax = axes[i]

        # Violin (forma de la distribución completa)
        sns.violinplot(data=df, x='Condicion', y=metrica, ax=ax,
                       order=orden_disp, palette=PALETTE,
                       inner=None, alpha=0.40, linewidth=0)
        # Box delgado encima
        sns.boxplot(data=df, x='Condicion', y=metrica, ax=ax,
                    order=orden_disp, palette=PALETTE,
                    width=0.22, showfliers=False, linewidth=1.2,
                    boxprops=dict(alpha=0.85))

        fila = df_res[df_res['Metrica'] == metrica]
        if len(fila) > 0:
            r      = fila.iloc[0]
            sig    = ('***' if r['p_MT'] < 0.001 else '**' if r['p_MT'] < 0.01
                      else '*' if r['p_MT'] < 0.05 else 'ns')
            l_str  = f"  lam={r['Lambda']:.2f}" if not np.isnan(r['Lambda']) else ""
            o_str  = f"  OVL={r['OVL_TC']:.2f}" if not np.isnan(r['OVL_TC']) else ""
            titulo = f"{METRICAS[metrica]}\nD%={r['Pct_cambio_MT']:+.1f}%{l_str}{o_str}  {sig}"
            ax.set_title(titulo, fontsize=8.5, fontweight='bold')
        else:
            ax.set_title(METRICAS[metrica], fontsize=9)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([c.split('(')[1].rstrip(')') for c in orden_disp],
                            rotation=20, ha='right', fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "Comparativa_Fatiga_Violin.png")
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Violin+Box: {ruta}")


# =============================================================================
#  4. FIGURA 2 — BARRAS DE CONVERGENCIA lambda y OVL
# =============================================================================

def figura_convergencia(df_res):
    df_plot = df_res.dropna(subset=['Lambda']).sort_values('Lambda', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Convergencia Biometrica: {CODIGO_REAL} Tarde -> {CODIGO_CONFUSION}\n'
        'lambda = fraccion del espacio cubierto por la fatiga'
        '  |  OVL = solapamiento de distribuciones',
        fontsize=13, fontweight='bold'
    )

    # Panel A — lambda
    ax = axes[0]
    col_l = ['#c0392b' if abs(v) > 0.5 else '#e67e22' if abs(v) > 0.2 else '#f1c40f'
              for v in df_plot['Lambda']]
    bars = ax.barh(df_plot['Metrica'], df_plot['Lambda'],
                   color=col_l, edgecolor='black', linewidth=0.5)
    ax.axvline(0,   color='black',  linewidth=0.8)
    ax.axvline(0.5, color='orange', linewidth=1.5, linestyle='--', alpha=0.7,
               label='lambda=0.5')
    ax.axvline(1.0, color='red',    linewidth=1.5, linestyle='--', alpha=0.7,
               label='lambda=1.0 (convergencia total)')
    for bar, val in zip(bars, df_plot['Lambda']):
        ax.text(val + (0.03 if val >= 0 else -0.03),
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center',
                ha='left' if val >= 0 else 'right', fontsize=9)
    ax.set_xlabel('Coeficiente de convergencia lambda', fontsize=11)
    ax.set_title('Panel A — lambda por metrica\n(rojo = fatiga acerca P11 a P5)', fontsize=11)
    ax.legend(fontsize=9); ax.set_xlim(-0.7, 1.4)

    # Panel B — OVL
    df_ovl = df_res.dropna(subset=['OVL_TC']).sort_values('OVL_TC', ascending=False)
    col_o  = ['#c0392b' if v > 0.7 else '#e67e22' if v > 0.4 else '#27ae60'
               for v in df_ovl['OVL_TC']]
    bars2  = axes[1].barh(df_ovl['Metrica'], df_ovl['OVL_TC'],
                           color=col_o, edgecolor='black', linewidth=0.5)
    axes[1].axvline(0.7, color='red',    linewidth=1.5, linestyle='--', alpha=0.7,
                    label='OVL=0.7 (alto solapamiento)')
    axes[1].axvline(0.4, color='orange', linewidth=1.5, linestyle='--', alpha=0.7,
                    label='OVL=0.4 (moderado)')
    for bar, val in zip(bars2, df_ovl['OVL_TC']):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9)
    axes[1].set_xlabel('OVL (Tarde vs P5)', fontsize=11)
    axes[1].set_title('Panel B — Solapamiento tarde vs P5\n(rojo = distribuciones indistinguibles)',
                       fontsize=11)
    axes[1].legend(fontsize=9); axes[1].set_xlim(0, 1.15)

    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "Convergencia_Lambda_OVL.png")
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Convergencia lambda+OVL: {ruta}")


# =============================================================================
#  5. FIGURA 3 — RADAR COMPARATIVO
# =============================================================================

def figura_radar(df, df_res):
    metricas_disp = [m for m in METRICAS
                     if m in df.columns and m in df_res['Metrica'].values]
    if len(metricas_disp) < 3: return

    medias = {}
    for cond in ORDEN:
        sub = df[df['Condicion'] == cond]
        if len(sub) > 0:
            medias[cond] = {m: sub[m].mean() for m in metricas_disp if m in sub.columns}

    if len(medias) < 2: return

    mins = {m: min(v.get(m, np.nan) for v in medias.values()) for m in metricas_disp}
    maxs = {m: max(v.get(m, np.nan) for v in medias.values()) for m in metricas_disp}

    def norm(vals):
        out = []
        for m in metricas_disp:
            rng = maxs[m] - mins[m]
            out.append((vals.get(m, mins[m]) - mins[m]) / rng if rng > 0 else 0.5)
        return out

    N      = len(metricas_disp)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    etiq   = [METRICAS[m].split('(')[0].strip()[:16] for m in metricas_disp]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle(
        f'Perfil Biometrico Comparativo\n'
        f'Verde={CODIGO_REAL} Manana  |  Rojo={CODIGO_REAL} Tarde  |  Gris={CODIGO_CONFUSION}',
        fontsize=13, fontweight='bold'
    )

    for cond, color, ls, lw, alpha in [
        (COND_MAN, '#2ecc71', '-',  2.5, 0.20),
        (COND_TAR, '#e74c3c', '-',  2.5, 0.20),
        (COND_CON, '#7f8c8d', '--', 1.8, 0.10),
    ]:
        if cond not in medias: continue
        v = norm(medias[cond]) + [norm(medias[cond])[0]]
        ax.plot(angles, v, color=color, linestyle=ls, linewidth=lw, label=cond)
        ax.fill(angles, v, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(etiq, size=9, weight='bold')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], size=7, color='gray')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.15), fontsize=9)

    ruta = os.path.join(OUTPUT_DIR, "Radar_Tres_Perfiles.png")
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Radar: {ruta}")


# =============================================================================
#  6. FIGURA 4 — HEATMAP DE ESTADISTICOS
# =============================================================================

def figura_heatmap(df_res):
    cols   = ['Pct_cambio_MT', 'd_MT', 'OVL_MT', 'OVL_TC', 'Lambda']
    labels = ['Delta% M->T', 'd Cohen M->T', 'OVL M/T', 'OVL T/P5', 'lambda']

    df_hm = df_res[['Metrica'] + cols].dropna(subset=['d_MT']).copy()
    df_hm = df_hm.set_index('Metrica')[cols].astype(float)
    df_hm.columns = labels

    fig, ax = plt.subplots(figsize=(11, max(4, len(df_hm) * 0.65 + 1.5)))
    sns.heatmap(df_hm, annot=True, fmt='.3f', cmap='RdYlGn_r',
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Magnitud del efecto'})
    ax.set_title(
        'Resumen de Deriva Biometrica\n'
        '(rojo=alto impacto/convergencia | verde=estable)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, "Heatmap_Estadisticos_Deriva.png")
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap: {ruta}")


# =============================================================================
#  RESUMEN CONSOLA
# =============================================================================

def imprimir_resumen(df_res):
    print("\n" + "=" * 70)
    print("  RESUMEN DE CONVERGENCIA BIOMETRICA")
    print("=" * 70)
    for _, r in df_res.sort_values('Lambda', ascending=False, key=abs).iterrows():
        lam = r['Lambda']
        if np.isnan(lam): continue
        nivel = "ALTO  " if abs(lam) > 0.5 else ("MEDIO " if abs(lam) > 0.2 else "BAJO  ")
        ovl_s = f"OVL={r['OVL_TC']:.3f}" if not np.isnan(r['OVL_TC']) else "OVL=N/A"
        print(f"  [{nivel}]  {r['Metrica']:<22}  lambda={lam:+.3f}  {ovl_s}"
              f"  D%={r['Pct_cambio_MT']:+.1f}%  d={r['d_MT']:.3f} [{cohen_label(r['d_MT'])}]")

    print("\n  CLAVES:")
    print("  lambda>0 = fatiga acerca P11 a P5 | lambda<0 = los aleja")
    print("  OVL>0.7  = distribuciones practicamente indistinguibles")
    print("  d >= 0.8 = efecto GRANDE | 0.5 = mediano | 0.2 = pequeno | <0.2 = negligible")
    print("\n  PARA LA TESIS:")
    print("  - p<0.05 en M->T: la fatiga altera significativamente la metrica.")
    print("  - OVL(Tarde,P5)>0.7: el clasificador no puede separar las distribuciones.")
    print("  - lambda cuantifica la direccion y magnitud de la convergencia.")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    df = cargar_datos()
    if df is None: exit(1)

    df_res = analizar_metricas(df)

    print("\n" + "-" * 60)
    print("  GENERANDO FIGURAS")
    print("-" * 60)

    figura_violin_box(df, df_res)
    figura_convergencia(df_res)
    figura_radar(df, df_res)
    figura_heatmap(df_res)
    imprimir_resumen(df_res)

    print(f"\n  Archivos en: {OUTPUT_DIR}")
    print("  1. Comparativa_Fatiga_Violin.png    — violin+box de las 8 metricas")
    print("  2. Convergencia_Lambda_OVL.png      — barras de lambda y solapamiento OVL")
    print("  3. Radar_Tres_Perfiles.png          — radar de los 3 perfiles normalizados")
    print("  4. Heatmap_Estadisticos_Deriva.png  — vision de conjunto en color")
    print("  5. Resultados_Estadisticos_Deriva_v2.csv")