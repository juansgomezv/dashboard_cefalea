"""
GRID SEARCH PARA OPTIMIZACIÓN DE UMAP + HDBSCAN
=================================================

Este script implementa búsqueda exhaustiva de hiperparámetros para UMAP y HDBSCAN
en el contexto de clustering estratificado por nivel de IndiceDolor.

MÉTRICAS DE EVALUACIÓN:
- DBCV (Density-Based Cluster Validation) - métrica interna de HDBSCAN
- Silhouette Score con distancias Gower
- Número de clusters encontrados
- Porcentaje de outliers
- Estabilidad de clusters (cluster persistence)

ESTRATEGIA:
Grid search independiente para cada nivel de IndiceDolor, ya que los tamaños
de muestra y estructuras son diferentes.
"""

import pandas as pd
import numpy as np
import json
import umap
import hdbscan
import gower
import itertools
from sklearn.metrics import silhouette_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_json_serializable(obj):
    """Convierte tipos numpy/pandas a tipos nativos Python."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


def load_config_and_data():
    """Carga configuración y dataset."""
    with open("config_transfer.json", "r") as f:
        config = json.load(f)
    
    df = pd.read_csv("dataset_procesado.csv").astype(str).fillna("NA")
    
    if "IndiceDolor" not in df.columns:
        raise ValueError("IndiceDolor no encontrado en dataset")
    
    return df, config


def prepare_features_and_weights(df, config, strategy="pesos_crudos", threshold=0.35):
    """Extrae features y pesos según configuración."""
    if strategy == "pesos_crudos":
        dict_pesos = config["pesos_crudos"]
    elif strategy == "pesos_norm":
        dict_pesos = config["pesos_norm"]
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")
    
    features_all = [f for f in config["features"] if f != "IndiceDolor"]
    weights_all = np.array([dict_pesos.get(f, 0.0) for f in features_all])
    
    weights_filtered = np.where(weights_all >= threshold, weights_all, 0.0)
    active_mask = weights_filtered > 0
    
    features_active = [f for f, active in zip(features_all, active_mask) if active]
    weights_active = weights_filtered[active_mask]
    
    df_features = df[features_active]
    
    return features_active, weights_active, df_features


# ═══════════════════════════════════════════════════════════════════════════
# GRID SEARCH PARA UN NIVEL DE IndiceDolor
# ═══════════════════════════════════════════════════════════════════════════

def grid_search_single_level(
    X, 
    D_precomputed,
    nivel_nombre,
    umap_params_grid,
    hdbscan_params_grid,
    n_jobs=1,
    verbose=True
):
    """
    Grid search exhaustivo para un nivel específico de IndiceDolor.
    
    Parameters
    ----------
    X : DataFrame
        Features del subset (para calcular métricas)
    D_precomputed : np.ndarray
        Matriz de distancias Gower precalculada
    nivel_nombre : str
        Nombre del nivel (para logging)
    umap_params_grid : dict
        Grilla de hiperparámetros UMAP
    hdbscan_params_grid : dict
        Grilla de hiperparámetros HDBSCAN
    n_jobs : int
        Número de procesos paralelos (solo para HDBSCAN)
    verbose : bool
        Mostrar progreso
    
    Returns
    -------
    results_df : DataFrame
        Resultados de todas las combinaciones probadas
    best_params : dict
        Mejores parámetros encontrados
    """
    n_samples = len(X)
    
    if verbose:
        print(f"\n{'='*75}")
        print(f"GRID SEARCH: {nivel_nombre}")
        print(f"{'='*75}")
        print(f"N muestras: {n_samples}")
    
    # Validar parámetros según tamaño de muestra
    max_n_components = min(umap_params_grid.get('n_components', [5])[0], n_samples - 2)
    max_n_neighbors = min(max(umap_params_grid.get('n_neighbors', [15])), n_samples - 1)
    
    if max_n_neighbors < 2:
        if verbose:
            print(f"⚠️  Muy pocas muestras ({n_samples}) para grid search significativo")
        return None, None
    
    # Generar todas las combinaciones
    umap_keys = list(umap_params_grid.keys())
    umap_values = [umap_params_grid[k] for k in umap_keys]
    
    hdbscan_keys = list(hdbscan_params_grid.keys())
    hdbscan_values = [hdbscan_params_grid[k] for k in hdbscan_keys]
    
    umap_combinations = list(itertools.product(*umap_values))
    hdbscan_combinations = list(itertools.product(*hdbscan_values))
    
    total_combinations = len(umap_combinations) * len(hdbscan_combinations)
    
    if verbose:
        print(f"Combinaciones UMAP: {len(umap_combinations)}")
        print(f"Combinaciones HDBSCAN: {len(hdbscan_combinations)}")
        print(f"Total combinaciones: {total_combinations}")
    
    results = []
    
    # Barra de progreso
    iterator = tqdm(umap_combinations, desc=f"{nivel_nombre}", 
                   disable=not verbose) if verbose else umap_combinations
    
    for umap_combo in iterator:
        # Construir parámetros UMAP
        umap_params = dict(zip(umap_keys, umap_combo))
        
        # Ajustar parámetros si exceden límites
        actual_n_comp = umap_params.get('n_components', 5)
        if actual_n_comp > max_n_components:
            continue  # Salta esta combinación si no cabe en la muestra
        umap_params['n_neighbors'] = min(umap_params.get('n_neighbors', 15), max_n_neighbors)
        
        try:
            # UMAP
            reducer = umap.UMAP(
                metric='precomputed',
                random_state=42,
                **umap_params
            )
            embedding = reducer.fit_transform(D_precomputed)
            
            # Probar cada combinación de HDBSCAN
            for hdbscan_combo in hdbscan_combinations:
                hdbscan_params = dict(zip(hdbscan_keys, hdbscan_combo))
                
                # Validar min_cluster_size no exceda tamaño de muestra
                hdbscan_params['min_cluster_size'] = min(
                    hdbscan_params.get('min_cluster_size', 10),
                    n_samples // 2
                )
                
                try:
                    # HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        metric='euclidean',
                        core_dist_n_jobs=n_jobs,
                        gen_min_span_tree=True,
                        **hdbscan_params
                    )
                    labels = clusterer.fit_predict(embedding)
                    
                    # Métricas
                    n_clusters = len(np.unique(labels[labels >= 0]))
                    n_outliers = (labels == -1).sum()
                    outlier_pct = n_outliers / len(labels) * 100
                    
                    # DBCV (Density-Based Cluster Validation) - métrica interna de HDBSCAN
                    dbcv = clusterer.relative_validity_ if hasattr(clusterer, 'relative_validity_') else np.nan
                    
                    # Silhouette (solo si hay al menos 2 clusters)
                    if n_clusters >= 2 and n_outliers < len(labels):
                        mask_valid = labels >= 0
                        try:
                            sil = silhouette_score(
                                D_precomputed[np.ix_(mask_valid, mask_valid)],
                                labels[mask_valid],
                                metric='precomputed'
                            )
                        except:
                            sil = np.nan
                    else:
                        sil = np.nan
                    
                    # Cluster persistence (estabilidad) - de HDBSCAN
                    if hasattr(clusterer, 'cluster_persistence_'):
                        persistence = clusterer.cluster_persistence_
                        avg_persistence = float(np.mean(persistence)) if len(persistence) > 0 else 0.0
                    else:
                        avg_persistence = np.nan
                    
                    # Guardar resultados
                    result = {
                        **{f'umap_{k}': v for k, v in umap_params.items()},
                        **{f'hdbscan_{k}': v for k, v in hdbscan_params.items()},
                        'n_clusters': n_clusters,
                        'n_outliers': n_outliers,
                        'outlier_pct': outlier_pct,
                        'silhouette': sil,
                        'dbcv': dbcv,
                        'avg_persistence': avg_persistence
                    }
                    
                    results.append(result)
                
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️  HDBSCAN falló: {e}")
                    continue
        
        except Exception as e:
            if verbose:
                print(f"\n⚠️  UMAP falló: {e}")
            continue
    
    if len(results) == 0:
        if verbose:
            print("❌ No se pudieron generar resultados válidos")
        return None, None
    
    results_df = pd.DataFrame(results)
    
    # Seleccionar mejores parámetros
    # Criterio: maximizar DBCV (o Silhouette si DBCV no disponible)
    # Penalizar % de outliers alto y número de clusters extremos
    
    results_df['score'] = 0.0
    
    # Componente 1: DBCV (0.0 a 1.0, mayor es mejor)
    if not results_df['dbcv'].isna().all():
        results_df['score'] += results_df['dbcv'].fillna(0) * 0.4
    
    # Componente 2: Silhouette (-1 a 1, mayor es mejor)
    if not results_df['silhouette'].isna().all():
        results_df['score'] += (results_df['silhouette'].fillna(0) + 1) / 2 * 0.3
    
    # Componente 3: Penalizar outliers (0 a 100%, menor es mejor)
    results_df['score'] += (1 - results_df['outlier_pct'] / 100) * 0.2
    
    # Componente 4: Preferir 2-5 clusters (penalizar extremos)
    ideal_clusters = 3
    cluster_penalty = np.abs(results_df['n_clusters'] - ideal_clusters) / 5
    results_df['score'] += np.clip(1 - cluster_penalty, 0, 1) * 0.1
    
    # Ordenar por score
    results_df = results_df.sort_values('score', ascending=False)
    
    # Extraer mejores parámetros
    best_row = results_df.iloc[0]
    
    best_params = {
        'umap': {k.replace('umap_', ''): v for k, v in best_row.items() 
                if k.startswith('umap_')},
        'hdbscan': {k.replace('hdbscan_', ''): v for k, v in best_row.items() 
                   if k.startswith('hdbscan_')},
        'metrics': {
            'n_clusters': int(best_row['n_clusters']),
            'n_outliers': int(best_row['n_outliers']),
            'outlier_pct': float(best_row['outlier_pct']),
            'silhouette': float(best_row['silhouette']) if not np.isnan(best_row['silhouette']) else None,
            'dbcv': float(best_row['dbcv']) if not np.isnan(best_row['dbcv']) else None,
            'score': float(best_row['score'])
        }
    }
    
    if verbose:
        print(f"\n{'='*75}")
        print("MEJORES PARÁMETROS:")
        print(f"{'='*75}")
        print("\nUMAP:")
        for k, v in best_params['umap'].items():
            print(f"  {k}: {v}")
        print("\nHDBSCAN:")
        for k, v in best_params['hdbscan'].items():
            print(f"  {k}: {v}")
        print("\nMétricas:")
        for k, v in best_params['metrics'].items():
            print(f"  {k}: {v}")
    
    return results_df, best_params


# ═══════════════════════════════════════════════════════════════════════════
# GRID SEARCH COMPLETO (TODOS LOS NIVELES)
# ═══════════════════════════════════════════════════════════════════════════

def run_complete_grid_search(
    strategy="pesos_crudos",
    threshold=0.35,
    output_dir="grid_search_results",
    preset="balanceado",
    n_jobs=1
):
    """
    Ejecuta grid search para todos los niveles de IndiceDolor.
    
    Parameters
    ----------
    strategy : str
        "pesos_crudos" o "pesos_norm"
    threshold : float
        Threshold de Cramér V para selección de features
    output_dir : str
        Directorio para guardar resultados
    preset : str
        "rapido", "balanceado", o "exhaustivo"
    n_jobs : int
        Número de procesos paralelos (HDBSCAN)
    """
    print("="*75)
    print("GRID SEARCH COMPLETO: UMAP + HDBSCAN")
    print("="*75)
    print(f"Preset: {preset}")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Cargar datos
    df, config = load_config_and_data()
    features_list, weights_array, df_features = prepare_features_and_weights(
        df, config, strategy, threshold
    )
    
    print(f"\nFeatures activos: {len(features_list)}")
    print(f"Estrategia: {strategy}")
    print(f"Threshold: {threshold}")
    
    # 2. Obtener grillas de hiperparámetros según preset
    if preset not in GRID_PRESETS:
        raise ValueError(f"Preset desconocido: {preset}. Usa 'rapido', 'balanceado', o 'exhaustivo'.")
    
    umap_params_grid = GRID_PRESETS[preset]['umap']
    hdbscan_params_grid = GRID_PRESETS[preset]['hdbscan']
    
    print(f"\nGrilla UMAP:")
    for k, v in umap_params_grid.items():
        print(f"  {k}: {v}")
    
    print(f"\nGrilla HDBSCAN:")
    for k, v in hdbscan_params_grid.items():
        print(f"  {k}: {v}")
    
    # Calcular número total de combinaciones
    n_umap_combos = np.prod([len(v) for v in umap_params_grid.values()])
    n_hdbscan_combos = np.prod([len(v) for v in hdbscan_params_grid.values()])
    total_combos = n_umap_combos * n_hdbscan_combos
    
    print(f"\nCombinaciones UMAP: {n_umap_combos}")
    print(f"Combinaciones HDBSCAN: {n_hdbscan_combos}")
    print(f"Total combinaciones por nivel: {total_combos}")
    print(f"Tiempo estimado por nivel: ~{total_combos * 0.5 / 60:.1f}-{total_combos * 1.5 / 60:.1f} min")
    
    # 3. Iterar por nivel de IndiceDolor
    y = df['IndiceDolor']
    niveles = sorted(y.unique())
    
    all_results = {}
    all_best_params = {}
    
    for nivel in niveles:
        mask_nivel = y == nivel
        X_nivel = df_features[mask_nivel]
        n_pacientes = len(X_nivel)
        
        print(f"\n{'#'*75}")
        print(f"NIVEL: {nivel} (n={n_pacientes})")
        print(f"{'#'*75}")
        
        if n_pacientes < 20:
            print(f"⚠️  Muy pocos pacientes para grid search ({n_pacientes} < 20)")
            continue
        
        # Calcular distancias
        print("Calculando matriz de distancias Gower...")
        D_nivel = gower.gower_matrix(X_nivel,
                                     cat_features=[True]*len(features_list),
                                     weight=weights_array)
        
        # Grid search
        results_df, best_params = grid_search_single_level(
            X_nivel,
            D_nivel,
            nivel,
            umap_params_grid,
            hdbscan_params_grid,
            n_jobs=n_jobs,
            verbose=True
        )
        
        if results_df is not None:
            all_results[nivel] = results_df
            all_best_params[nivel] = best_params
            
            # Guardar resultados detallados
            results_path = f"{output_dir}/grid_search_{nivel}.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\n✓ Resultados guardados: {results_path}")
    
    # 4. Guardar mejores parámetros de todos los niveles
    best_params_path = f"{output_dir}/best_params_all_levels.json"
    with open(best_params_path, 'w') as f:
        json.dump(convert_to_json_serializable(all_best_params), f, indent=2)
    print(f"\n✓ Mejores parámetros guardados: {best_params_path}")
    
    # 5. Crear visualizaciones
    create_grid_search_visualizations(all_results, output_dir)
    
    # 6. Resumen final
    print("\n" + "="*75)
    print("RESUMEN FINAL")
    print("="*75)
    
    for nivel, params in all_best_params.items():
        metrics = params['metrics']
        print(f"\n{nivel}:")
        print(f"  Clusters: {metrics['n_clusters']}")
        print(f"  Outliers: {metrics['outlier_pct']:.1f}%")
        if metrics['dbcv'] is not None:
            print(f"  DBCV: {metrics['dbcv']:.3f}")
        if metrics['silhouette'] is not None:
            print(f"  Silhouette: {metrics['silhouette']:.3f}")
        print(f"  Score total: {metrics['score']:.3f}")
    
    return all_best_params

def create_grid_search_visualizations(all_results, output_dir):
    """Crea visualizaciones de los resultados del grid search."""
    
    for nivel, results_df in all_results.items():
        if len(results_df) == 0:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Grid Search: {nivel}', fontsize=16, fontweight='bold')
        
        # Plot 1: Score vs n_components (UMAP)
        ax = axes[0, 0]
        
        # Agrupar por n_components y calcular estadísticas
        grouped = results_df.groupby('umap_n_components')['score']
        positions = sorted(results_df['umap_n_components'].unique())
        
        # Crear boxplot manualmente para cada grupo
        box_data = [results_df[results_df['umap_n_components'] == pos]['score'].values 
                   for pos in positions]
        
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='blue', markersize=4, alpha=0.5))
        
        # Añadir puntos individuales con jitter
        for i, pos in enumerate(positions):
            y_data = box_data[i]
            x_data = np.random.normal(pos, 0.04, size=len(y_data))  # jitter
            ax.scatter(x_data, y_data, alpha=0.4, s=20, color='darkblue', zorder=3)
        
        ax.set_xlabel('n_components (UMAP)', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Score vs UMAP n_components', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(p)) for p in positions])
        
        # Plot 2: Score vs min_cluster_size (HDBSCAN)
        ax = axes[0, 1]
        
        grouped2 = results_df.groupby('hdbscan_min_cluster_size')['score']
        positions2 = sorted(results_df['hdbscan_min_cluster_size'].unique())
        
        box_data2 = [results_df[results_df['hdbscan_min_cluster_size'] == pos]['score'].values 
                    for pos in positions2]
        
        bp2 = ax.boxplot(box_data2, positions=range(len(positions2)), widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='green', markersize=4, alpha=0.5))
        
        # Añadir puntos individuales
        for i, pos in enumerate(positions2):
            y_data = box_data2[i]
            x_data = np.random.normal(i, 0.04, size=len(y_data))
            ax.scatter(x_data, y_data, alpha=0.4, s=20, color='darkgreen', zorder=3)
        
        ax.set_xlabel('min_cluster_size (HDBSCAN)', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Score vs HDBSCAN min_cluster_size', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(len(positions2)))
        ax.set_xticklabels([str(int(p)) for p in positions2])
        
        # Plot 3: n_clusters vs outlier_pct (CORREGIDO: variable dependiente en Y)
        ax = axes[1, 0]
        
        # n_clusters es la variable independiente (elegida por parámetros)
        # outlier_pct es variable dependiente (resultado)
        scatter = ax.scatter(results_df['n_clusters'],       # X: independiente
                           results_df['outlier_pct'],       # Y: dependiente
                           c=results_df['score'],
                           cmap='viridis',
                           alpha=0.6,
                           s=50,
                           edgecolors='black',
                           linewidth=0.5)
        
        ax.set_xlabel('Número de Clusters', fontsize=10)
        ax.set_ylabel('% Outliers', fontsize=10)
        ax.set_title('Outliers vs Clusters (color = score)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Añadir colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Score')
        cbar.ax.tick_params(labelsize=9)
        
        # Plot 4: Top 10 configuraciones
        ax = axes[1, 1]
        top10 = results_df.head(10).copy()
        
        # Crear labels más informativos
        top10['config'] = top10.apply(
            lambda r: f"U:{int(r['umap_n_components'])},H:{int(r['hdbscan_min_cluster_size'])}", 
            axis=1
        )
        
        # Barplot horizontal para mejor legibilidad
        colors = plt.cm.viridis(top10['score'] / top10['score'].max())
        bars = ax.barh(range(len(top10)), top10['score'], color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10['config'], fontsize=9)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Config (U:umap_comp, H:hdbscan_size)', fontsize=9)
        ax.set_title('Top 10 Configuraciones', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # Mejor ranking arriba
        
        # Añadir valores en las barras
        for i, (bar, score) in enumerate(zip(bars, top10['score'])):
            ax.text(score, i, f' {score:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plot_path = f"{output_dir}/grid_search_viz_{nivel}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualización guardada: {plot_path}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# GRILLA SIMPLIFICADA (RÁPIDA) vs EXHAUSTIVA (COMPLETA)
# ═══════════════════════════════════════════════════════════════════════════

GRID_PRESETS = {
    'rapido': {
        'umap': {
            'n_components': [5],
            'n_neighbors': [15],
            'min_dist': [0.0, 0.1]
        },
        'hdbscan': {
            'min_cluster_size': [10, 15],
            'min_samples': [5],
            'cluster_selection_epsilon': [0.0]
        }
    },
    'balanceado': {
        'umap': {
            'n_components': [2, 3],
            'n_neighbors': [5, 10, 15],
            'min_dist': [0.0, 0.1, 0.3]
        },
        'hdbscan': {
            'min_cluster_size': [5, 10, 15, 20],
            'min_samples': [3, 5, 10],
            'cluster_selection_epsilon': [0.0, 0.1]
        }
    },
    'exhaustivo': {
        'umap': {
            'n_components': [2, 3, 5, 8, 10],
            'n_neighbors': [5, 10, 15, 20, 30],
            'min_dist': [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
        },
        'hdbscan': {
            'min_cluster_size': [5, 8, 10, 12, 15, 20, 25],
            'min_samples': [1, 3, 5, 7, 10],
            'cluster_selection_epsilon': [0.0, 0.05, 0.1, 0.2, 0.3]
        }
    }
}



# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT: CONTROL DE EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # --- OPCIÓN 1: EJECUCIÓN INDIVIDUAL (Para pruebas rápidas) ---
    # Usa esta sección para probar los cambios en el preset y n_components
    EJECUCION_INDIVIDUAL = True  # Cambia a False para usar la batería completa

    if EJECUCION_INDIVIDUAL:
        # CONFIGURA AQUÍ TU PRUEBA:
        mi_estrategia = 'pesos_crudos'    # 'pesos_crudos' o 'pesos_norm'
        mi_threshold  = 0.04        # El valor corregido que mencionaste
        mi_preset     = 'exhaustivo'    # 'rapido', 'balanceado' o 'exhaustivo'
        mi_output     = 'prueba_gs_04_crudo_exh' # Nombre de la carpeta de prueba

        print(f"\n🧪 INICIANDO PRUEBA INDIVIDUAL")
        print(f"⚙️  Config: {mi_estrategia} | TH: {mi_threshold} | Preset: {mi_preset}")
        
        run_complete_grid_search(
            strategy=mi_estrategia,
            threshold=mi_threshold,
            output_dir=mi_output,
            preset=mi_preset,
            n_jobs=-1
        )
        print(f"\n✅ Prueba finalizada. Revisa la carpeta: {mi_output}/")


    # --- OPCIÓN 2: BATERÍA AUTOMATIZADA (24 EXPERIMENTOS) ---
    # Para activarla, pon EJECUCION_INDIVIDUAL = False arriba
    else:
        thresholds_crudos = [0.25, 0.30, 0.35, 0.40]
        thresholds_norm = [0.025, 0.030, 0.035, 0.040]
        estrategias = ['pesos_crudos', 'pesos_norm']
        presets = ['rapido', 'balanceado', 'exhaustivo']

        print(f"\n🚀 Iniciando batería automatizada de 24 experimentos...")
        print(f"⚙️  Configuración: n_jobs=-1")

        for strategy in estrategias:
            lista_th = thresholds_crudos if strategy == 'pesos_crudos' else thresholds_norm
            strat_tag = 'crudo' if strategy == 'pesos_crudos' else 'norm'

            for th in lista_th:
                for preset in presets:
                    pre_tag = {'rapido': 'rap', 'balanceado': 'bal', 'exhaustivo': 'exh'}[preset]
                    th_tag = str(th).replace("0.", "0")
                    folder_name = f"gs_{th_tag}_{strat_tag}_{pre_tag}"

                    print(f"\n" + "═"*75)
                    print(f"🔍 PROCESANDO: {folder_name}")
                    print("═"*75)

                    try:
                        run_complete_grid_search(
                            strategy=strategy,
                            threshold=th,
                            output_dir=folder_name,
                            preset=preset,
                            n_jobs=-1 
                        )
                    except Exception as e:
                        print(f"\n❌ Error en {folder_name}: {str(e)}")
                        continue

        print("\n" + "═"*75)
        print("✨ BATERÍA DE 24 EXPERIMENTOS FINALIZADA ✨")
        print("═"*75)