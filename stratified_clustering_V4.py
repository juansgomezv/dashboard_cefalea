"""
IMPLEMENTACIÓN COMPLETA: STRATIFIED CLUSTERING OPTIMIZADO
========================================================================

Este script implementa el ESCENARIO B utilizando hiperparámetros 
dinámicos y específicos para cada nivel de dolor.
Guarda los resultados en la raíz del proyecto e incluye las coordenadas UMAP.
"""

import pandas as pd
import numpy as np
import json
import umap
import hdbscan
import gower
import matplotlib.pyplot as plt
from math import pi
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# LIBRO DE RECETAS (HIPERPARÁMETROS GANADORES DEL GRID SEARCH)
# ═══════════════════════════════════════════════════════════════════════════
CONFIGURACIONES_GANADORAS = {
    "0": {
        "estrategia": "pesos_crudos",
        "threshold": 0.30,
        "umap": {"n_components": 2, "n_neighbors": 30, "min_dist": 0.3},
        "hdbscan": {"min_cluster_size": 8, "min_samples": 1, "cluster_selection_epsilon": 0.0}
    },
    "1": {
        "estrategia": "pesos_crudos",
        "threshold": 0.40,
        "umap": {"n_components": 5, "n_neighbors": 15, "min_dist": 0.1},
        "hdbscan": {"min_cluster_size": 15, "min_samples": 5, "cluster_selection_epsilon": 0.0}
    },
    "2": {
        "estrategia": "pesos_crudos",
        "threshold": 0.35,
        "umap": {"n_components": 3, "n_neighbors": 15, "min_dist": 0.3},
        "hdbscan": {"min_cluster_size": 10, "min_samples": 3, "cluster_selection_epsilon": 0.0}
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES Y PREPARACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_json_serializable(obj):
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
    print("--- CARGANDO CONFIGURACIÓN Y DATOS ---")
    try:
        with open("config_transfer.json", "r") as f:
            config = json.load(f)
        df = pd.read_csv("dataset_procesado.csv").astype(str).fillna("NA")
        if "IndiceDolor" not in df.columns:
            raise ValueError("IndiceDolor no encontrado.")
        return df, config
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise SystemExit(1)

def prepare_features_and_weights(df, config, strategy, threshold):
    dict_pesos = config.get(strategy, {})
    features_all = [f for f in config["features"] if f != "IndiceDolor"]
    weights_all = np.array([dict_pesos.get(f, 0.0) for f in features_all])
    
    weights_filtered = np.where(weights_all >= threshold, weights_all, 0.0)
    active_mask = weights_filtered > 0
    
    features_active = [f for f, active in zip(features_all, active_mask) if active]
    weights_active = weights_filtered[active_mask]
    
    return features_active, weights_active

# ═══════════════════════════════════════════════════════════════════════════
# ESCENARIO B: CLUSTERING ESTRATIFICADO CON RECETAS DINÁMICAS
# ═══════════════════════════════════════════════════════════════════════════

def scenario_b_stratified_clustering(df, config):
    print("\n" + "="*75)
    print("EJECUTANDO CLUSTERING CON HIPERPARÁMETROS OPTIMIZADOS (GRID SEARCH)")
    print("="*75)
    
    y = df['IndiceDolor']
    niveles = sorted(y.unique())
    all_results = {}
    
    # Arreglos para guardar etiquetas y coordenadas
    all_labels = np.full(len(df), -1, dtype=int)
    all_umap_x = np.full(len(df), np.nan)
    all_umap_y = np.full(len(df), np.nan)
    
    cluster_counter = 0

    for nivel in niveles:
        print("\n" + "-"*75)
        print(f"PROCESANDO NIVEL DE DOLOR: {nivel}")
        
        nivel_str = str(nivel)
        if nivel_str not in CONFIGURACIONES_GANADORAS:
            print(f"⚠️  No hay configuración para el nivel {nivel}. Saltando...")
            continue
            
        receta = CONFIGURACIONES_GANADORAS[nivel_str]
        print(f"⚙️  Aplicando Receta: Threshold={receta['threshold']} | UMAP={receta['umap']} | HDBSCAN={receta['hdbscan']}")
        
        features_list, weights_array = prepare_features_and_weights(
            df, config, receta["estrategia"], receta["threshold"]
        )
        print(f"✓ Features activos tras aplicar threshold: {len(features_list)}")

        mask_nivel = y == nivel
        indices_nivel = df[mask_nivel].index
        X_nivel = df.loc[mask_nivel, features_list]
        n_pacientes = len(X_nivel)
        
        if n_pacientes < 15:
            print(f"⚠️  Muy pocos pacientes ({n_pacientes}) para clustering robusto.")
            all_results[nivel_str] = {'n_pacientes': n_pacientes, 'n_subgrupos': 0, 'n_outliers': 0}
            continue
        
        # 3. Matriz de distancias Gower
        D_nivel = gower.gower_matrix(X_nivel, cat_features=[True]*len(features_list), weight=weights_array)
        
        # 4. UMAP
        u_params = receta["umap"]
        reducer = umap.UMAP(
            n_components=int(u_params["n_components"]), 
            n_neighbors=int(u_params["n_neighbors"]), 
            min_dist=float(u_params["min_dist"]), 
            metric='precomputed', 
            random_state=42
        )
        embedding = reducer.fit_transform(D_nivel)
        
        # --- NUEVO: Guardar las primeras 2 componentes para la visualización 2D en Streamlit ---
        all_umap_x[indices_nivel] = embedding[:, 0]
        if embedding.shape[1] >= 2:
            all_umap_y[indices_nivel] = embedding[:, 1]
        else:
            all_umap_y[indices_nivel] = 0.0 # Por si algún UMAP fuera de 1 sola dimensión
        
        # 5. HDBSCAN
        h_params = receta["hdbscan"]
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(h_params["min_cluster_size"]), 
            min_samples=int(h_params["min_samples"]), 
            cluster_selection_epsilon=float(h_params["cluster_selection_epsilon"]), 
            metric='euclidean'
        )
        labels_nivel = clusterer.fit_predict(embedding)
        
        # 6. Extracción de resultados
        subgrupos = np.unique(labels_nivel[labels_nivel >= 0])
        n_subgrupos = len(subgrupos)
        n_outliers = (labels_nivel == -1).sum()
        
        print(f"📊 Resultados: {n_subgrupos} subgrupos encontrados. Outliers: {n_outliers}")
        
        if n_subgrupos == 0:
            all_results[nivel_str] = {'n_pacientes': n_pacientes, 'n_subgrupos': 0, 'n_outliers': n_outliers}
            continue
        
        subgrupo_profiles = []
        for subgrupo_id in subgrupos:
            mask_sub = labels_nivel == subgrupo_id
            n_sub = mask_sub.sum()
            
            perfil = {}
            for feat in features_list:
                mode_val = X_nivel.loc[mask_sub, feat].mode()
                if len(mode_val) > 0:
                    mode_val = mode_val[0]
                    pct = (X_nivel.loc[mask_sub, feat] == mode_val).mean()
                    
                    mode_val_native = mode_val.item() if hasattr(mode_val, 'item') else str(mode_val)
                    pct_native = float(pct) if hasattr(pct, 'item') else float(pct)
                    perfil[feat] = {'valor': mode_val_native, 'frecuencia': pct_native}
            
            subgrupo_profiles.append({
                'id': int(cluster_counter),
                'n': int(n_sub),
                'porcentaje': float(n_sub/n_pacientes*100),
                'perfil': perfil
            })
            
            global_indices = indices_nivel[mask_sub]
            all_labels[global_indices] = cluster_counter
            cluster_counter += 1
        
        all_results[nivel_str] = {
            'n_pacientes': int(n_pacientes),
            'n_subgrupos': int(n_subgrupos),
            'n_outliers': int(n_outliers),
            'subgrupos': subgrupo_profiles
        }
        
        # --- GRÁFICOS DE RADAR ---
        if n_subgrupos > 0:
            radar_features = features_list[:8] 
            N = len(radar_features)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            plt.xticks(angles[:-1], radar_features, color='grey', size=8)
            ax_radar.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=7)
            plt.ylim(0, 1.1)
            
            for sg_prof in subgrupo_profiles:
                values = [sg_prof['perfil'].get(feat, {'frecuencia':0})['frecuencia'] for feat in radar_features]
                values += values[:1]
                ax_radar.plot(angles, values, linewidth=2, linestyle='solid', label=f"Subgrupo {sg_prof['id']} (n={sg_prof['n']})")
                ax_radar.fill(angles, values, alpha=0.1)
                
            plt.title(f'Frecuencia de Síntomas - Nivel {nivel}', size=14, y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            radar_path = f"radar_nivel_{nivel}.png" # Guardado en raíz
            plt.savefig(radar_path, dpi=150, bbox_inches='tight')
            plt.close()

    # --- DIAGRAMA DE SANKEY GLOBAL ---
    print("\nGenerando Diagrama de Sankey Global...")
    sankey_labels = ["Total Pacientes"]
    source, target, value = [], [], []
    
    nivel_start_idx = 1
    for i, nivel in enumerate(niveles):
        nivel_str = str(nivel)
        if nivel_str in all_results:
            sankey_labels.append(f"Dolor Nivel {nivel}")
            source.append(0) 
            target.append(nivel_start_idx + i)
            value.append(all_results[nivel_str]['n_pacientes'])

    current_node_idx = len(sankey_labels)
    for i, nivel in enumerate(niveles):
        nivel_str = str(nivel)
        if nivel_str in all_results:
            nivel_idx = nivel_start_idx + i
            res = all_results[nivel_str]
            
            if res.get('n_outliers', 0) > 0:
                sankey_labels.append(f"Outliers (Nivel {nivel})")
                source.append(nivel_idx)
                target.append(current_node_idx)
                value.append(res['n_outliers'])
                current_node_idx += 1
                
            if 'subgrupos' in res:
                for sg in res['subgrupos']:
                    sankey_labels.append(f"Subgrupo {sg['id']}")
                    source.append(nivel_idx)
                    target.append(current_node_idx)
                    value.append(sg['n'])
                    current_node_idx += 1

    fig_sankey = go.Figure(data=[go.Sankey(
        node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=sankey_labels),
        link = dict(source=source, target=target, value=value)
    )])
    fig_sankey.update_layout(title_text="Flujo de Estudiantes: Total -> Índice de Dolor -> Subgrupos")
    sankey_path = "flujo_pacientes_sankey.html" # Guardado en raíz
    fig_sankey.write_html(sankey_path)

    # --- GUARDAR RESULTADOS EN LA RAÍZ ---
    with open("resultados_finales.json", 'w') as f:
        json.dump(convert_to_json_serializable(all_results), f, indent=2, ensure_ascii=False)
    
    df_output = df.copy()
    df_output['Subgrupo_Global'] = all_labels
    
    # Añadimos las nuevas columnas de coordenadas
    df_output['umap_x'] = all_umap_x
    df_output['umap_y'] = all_umap_y
    
    df_output.to_csv("dataset_con_subgrupos.csv", index=False)
    print(f"✓ ¡Análisis completado! Archivos generados en el directorio actual.")

if __name__ == "__main__":
    # Cargamos datos y la configuración de pesos general
    df_principal, config_transfer = load_config_and_data()
    
    # Ejecutamos el pipeline (se guardará todo en la carpeta actual)
    scenario_b_stratified_clustering(df_principal, config_transfer)