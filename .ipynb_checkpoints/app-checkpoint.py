import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

IMG_PATH = "IMAGE11.jpg"  # Change to your image filename
IMG_WIDTH, IMG_HEIGHT = 718, 967

ZONE_Y = {"A": 30, "B": 92, "C": 175}
PARTITIONS = ['Rack', 'Shelf', 'Bin']
PART_X = {'Rack': 40, 'Shelf': 340, 'Bin': 650}
PART_COLOR = {'Rack': '#228b22', 'Shelf': '#4682b4', 'Bin': '#ff8c00'}
ZONE_COLOR = {'A': '#ff2323', 'B': '#ffac42', 'C': '#47c956'}
XYZ_COLOR = {'X': '#1565C0', 'Y': '#FFEB3B', 'Z': '#8E24AA'}
XYZ_LABELS = {'X': 'Low Variability (X)', 'Y':'Medium (Y)', 'Z': 'High (Z)'}

st.set_page_config(page_title="Warehouse Classification & Forecasting", layout="wide")
st.title("Warehouse ABC/XYZ/Weight Classification & SKU Demand Forecaster")

uploaded = st.file_uploader("Upload your warehouse CSV", type=["csv", "xlsx"])
if uploaded:
    # LOAD AND CLEAN COLUMN NAMES
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower().str.replace("(", "").str.replace(")", "")
    df = df.rename(columns={
        "dimensions_cm_lxwxh":"dimensions",
        "weight_kg":"weight",
        "storage_type":"storage",
        "average_daily_orders":"avg_orders",
        "product_category":"category"
    })
    st.write("Loaded columns:", df.columns.tolist())
    tab1, tab2, tab3 = st.tabs(["Classification & Layout", "SKU Demand Forecaster", "AGV Routing"])
    with tab1:
        st.header("ABC Classification")
        a_pct = st.slider("ABC 'A' cutoff (%)", 0.05, 0.90, 0.3)
        b_pct = st.slider("ABC 'B' cutoff (%)", a_pct+0.01, 0.99, 0.7)
        df_sorted = df.sort_values("avg_orders", ascending=False).reset_index(drop=True)
        n_skus = len(df_sorted)
        a_cut, b_cut = int(a_pct * n_skus), int(b_pct * n_skus)
        df_sorted["abc"] = ["A"]*a_cut + ["B"]*(b_cut-a_cut) + ["C"]*(n_skus-b_cut)
        st.write(f"A: Top {a_cut}, B: Next {b_cut-a_cut}, C: {n_skus-b_cut}")
        st.dataframe(df_sorted[["sku_id","avg_orders","abc"]])

        st.header("XYZ Classification")
        mean_orders = df["avg_orders"].mean()
        std_orders = df["avg_orders"].std()
        x_cv = st.slider("XYZ 'X' cutoff (fraction of mean)", 0.3, 1.3, 0.7)
        y_cv = st.slider("XYZ 'Y' cutoff (fraction of mean)", x_cv+0.01, 3.0, 1.3)
        df_sorted["cv"] = df_sorted["avg_orders"] / (mean_orders + 1e-9)
        df_sorted["xyz"] = df_sorted["cv"].apply(lambda v: "X" if v <= x_cv else ("Y" if v <= y_cv else "Z"))
        st.dataframe(df_sorted[["sku_id","avg_orders","xyz"]])

        st.header("Weight Bin Classification")
        w_edges = st.slider("Weight bin edges (kg)", 0.1, 10.0, (1.0,3.0))
        weight_bins = pd.cut(df_sorted["weight"], bins=[0,w_edges[0],w_edges[1],df_sorted["weight"].max()+1], labels=["Light","Medium","Heavy"])
        df_sorted["weight_bin"] = weight_bins.astype(str).fillna("")
        st.dataframe(df_sorted[["sku_id","weight","weight_bin"]])
        csv_classification = df_sorted.to_csv(index=False)
        st.download_button(
            label="Download Classification CSV",
            data=csv_classification,
            file_name='warehouse_classification.csv',
            mime='text/csv'
        )

        st.header("Class Distribution Visualization")
        fig1, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].pie(df_sorted['abc'].value_counts(), labels=df_sorted['abc'].value_counts().index, autopct='%1.1f%%', colors=[ZONE_COLOR[c] for c in ['A','B','C']])
        axs[0].set_title("ABC Classes")
        axs[1].pie(df_sorted['xyz'].value_counts(), labels=df_sorted['xyz'].value_counts().index, autopct='%1.1f%%', colors=[XYZ_COLOR[x] for x in ['X','Y','Z']])
        axs[1].set_title("XYZ Classes")
        axs[2].pie(df_sorted['weight_bin'].value_counts(), labels=df_sorted['weight_bin'].value_counts().index, autopct='%1.1f%%', colors=["#a0e7e5","#b4a0e7","#e7a0a0"])
        axs[2].set_title("Weight Bins")
        st.pyplot(fig1)

        st.header("Warehouse Map Visualization")
        img = Image.open(IMG_PATH).resize((IMG_WIDTH, IMG_HEIGHT))
        fig, ax = plt.subplots(figsize=(12,3.5))
        ax.imshow(img)
        for zone in ["A", "B", "C"]:
            zone_y = ZONE_Y[zone]
            ax.text(IMG_WIDTH//2, zone_y-18, f"ZONE {zone}", fontsize=16, color=ZONE_COLOR[zone], weight="bold", ha="center")
            for part in PARTITIONS:
                x0 = PART_X[part]
                ax.text(x0, zone_y-7, part, fontsize=12, color=PART_COLOR[part], ha="left", weight="bold")
                ax.add_patch(Rectangle((x0, zone_y), 180, 35, color=PART_COLOR[part], alpha=0.05))
                for idx, xyz in enumerate(["X","Y","Z"]):
                    x_band = x0+7+idx*58
                    ax.text(x_band+28, zone_y+3, XYZ_LABELS[xyz], fontsize=9, color=XYZ_COLOR[xyz], ha="center")
                    skus = df_sorted[(df_sorted['abc']==zone)
                                    & (df_sorted['storage'].str.lower()==part.lower())
                                    & (df_sorted['xyz']==xyz)]
                    for k, row in enumerate(skus.itertuples()):
                        sku_id = getattr(row, "sku_id", "")
                        wbin = getattr(row, "weight_bin", "")
                        if wbin is None: wbin = ""
                        rx = x_band + k*55
                        ry = zone_y + 16 + idx*14
                        label = f"{sku_id}\n{wbin}"
                        box = Rectangle((rx, ry), 46, 13, color=XYZ_COLOR[xyz], ec=ZONE_COLOR[zone], lw=1.5, alpha=0.8)
                        ax.add_patch(box)
                        ax.text(rx+23, ry+7, label, ha="center", va="center", fontsize=7, color="black")
        ax.axis('off')
        st.pyplot(fig)

    with tab2:
        st.header("SKU Demand Forecaster")
        st.write("Forecast SKU demand with exponential and linear options.")
        horizon_days = st.slider("Forecast horizon (days)", 7, 180, 30)
        forecast_type = st.selectbox("Forecast Model", ["Linear (avg*d)", "Exponential Growth (avg*d*1.02^d)"])
        if forecast_type.startswith("Linear"):
            df["forecast_orders"] = df["avg_orders"] * horizon_days
        else:
            df["forecast_orders"] = df["avg_orders"] * horizon_days * (1.02 ** horizon_days)
        st.write(f"Forecast period: Next {horizon_days} days; Model: {forecast_type}")
        st.dataframe(df[["sku_id","avg_orders","forecast_orders","storage","weight"]])

        # Forecast Classification
        df_forecast = df.sort_values("forecast_orders", ascending=False).reset_index(drop=True)
        n_skus = len(df_forecast)
        af_pct = st.slider("Forecast ABC 'A' cutoff (%)", 0.05, 0.9, 0.3)
        bf_pct = st.slider("Forecast ABC 'B' cutoff (%)", af_pct+0.01, 0.99, 0.7)
        af_cut, bf_cut = int(af_pct * n_skus), int(bf_pct * n_skus)
        df_forecast["abc_forecast"] = ["A"]*af_cut + ["B"]*(bf_cut-af_cut) + ["C"]*(n_skus-bf_cut)
        mean_forecast = df_forecast["forecast_orders"].mean()
        std_forecast = df_forecast["forecast_orders"].std()
        xf_cv = st.slider("Forecast XYZ 'X' cutoff", 0.3, 1.3, 0.8)
        yf_cv = st.slider("Forecast XYZ 'Y' cutoff", xf_cv+0.01, 3.0, 1.3)
        df_forecast["cv_forecast"] = df_forecast["forecast_orders"] / (mean_forecast + 1e-9)
        df_forecast["xyz_forecast"] = df_forecast["cv_forecast"].apply(lambda v: "X" if v <= xf_cv else ("Y" if v <= yf_cv else "Z"))
        bins = [0, 1.0, 3.0, df_forecast["weight"].max()+1]
        df_forecast["weight_bin_forecast"] = pd.cut(df_forecast["weight"], bins=bins, labels=["Light","Medium","Heavy"]).astype(str).fillna("")
        st.dataframe(df_forecast[["sku_id","forecast_orders","abc_forecast","xyz_forecast","weight_bin_forecast","storage"]])
        
        csv_forecast = df_forecast.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv_forecast,
            file_name='warehouse_forecast.csv',
            mime='text/csv'
        )
        
        st.header("Forecasted Demand Bar Chart")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_forecast['sku_id'], df_forecast['forecast_orders'], color="#4e79a7")
        ax.set_ylabel("Forecasted Orders")
        ax.set_xlabel("SKU ID")
        ax.set_title(f"Forecasted SKU Demand ({forecast_type})")
        ax.set_xticklabels(df_forecast['sku_id'], fontsize=8, rotation=30, ha='right')

        st.pyplot(fig)

        st.header("Forecasted Class Distribution")
        fig2, axs2 = plt.subplots(1,3,figsize=(12,4))
        axs2[0].pie(df_forecast['abc_forecast'].value_counts(), labels=df_forecast['abc_forecast'].value_counts().index, autopct='%1.1f%%', colors=[ZONE_COLOR[c] for c in ['A','B','C']])
        axs2[0].set_title("ABC (Forecast)")
        axs2[1].pie(df_forecast['xyz_forecast'].value_counts(), labels=df_forecast['xyz_forecast'].value_counts().index, autopct='%1.1f%%', colors=[XYZ_COLOR[x] for x in ['X','Y','Z']])
        axs2[1].set_title("XYZ (Forecast)")
        axs2[2].pie(df_forecast['weight_bin_forecast'].value_counts(), labels=df_forecast['weight_bin_forecast'].value_counts().index, autopct='%1.1f%%', colors=["#a0e7e5","#b4a0e7","#e7a0a0"])
        axs2[2].set_title("Weight Bins (Forecast)")
        st.pyplot(fig2)

    with tab3:
        st.header("AGV Routing & Path Optimization")
        st.write("Visualize AGV paths between warehouse zones using shortest-path algorithms.")
        
        # Define nodes (locations in warehouse)
        nodes = {
            "Receiving": (50, 50),
            "Zone_A_Rack": (200, 150),
            "Zone_A_Shelf": (400, 150),
            "Zone_B_Rack": (200, 400),
            "Zone_B_Shelf": (400, 400),
            "Zone_C_Rack": (200, 650),
            "Zone_C_Shelf": (400, 650),
            "Shipping": (650, 50),
            "Outbound": (650, 400)
        }
        
        # Distance matrix (Euclidean or real path distances)
        import numpy as np
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        node_names = list(nodes.keys())
        n = len(node_names)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = euclidean_distance(nodes[node_names[i]], nodes[node_names[j]])
        
        # Display distance matrix
        st.subheader("Distance Matrix (Euclidean)")
        df_dist = pd.DataFrame(distance_matrix, index=node_names, columns=node_names)
        st.dataframe(df_dist.style.format("{:.1f}"))
        
        # Path planning with Dijkstra
        from scipy.sparse.csgraph import dijkstra
        st.subheader("Shortest Path Finder")
        start_node = st.selectbox("Start Location", node_names, index=0)
        end_node = st.selectbox("End Location", node_names, index=7)
        
        dist_matrix_sparse = distance_matrix.copy()
        distances, predecessors = dijkstra(dist_matrix_sparse, return_predecessors=True, indices=node_names.index(start_node))
        
        end_idx = node_names.index(end_node)
        shortest_distance = distances[end_idx]
        st.write(f"Shortest distance from {start_node} to {end_node}: **{shortest_distance:.2f} units**")
        
        # Reconstruct path
        path_indices = []
        current = end_idx
        while current != -9999:
            path_indices.insert(0, current)
            current = predecessors[current]
            if current == -9999 or current == node_names.index(start_node):
                path_indices.insert(0, node_names.index(start_node))
                break
        
        path_names = [node_names[i] for i in path_indices]
        st.write(f"Path: {' → '.join(path_names)}")
        
        # Visualize AGV route on warehouse map
        st.subheader("AGV Route Visualization")
        img = Image.open(IMG_PATH).resize((IMG_WIDTH, IMG_HEIGHT))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img, extent=[0, IMG_WIDTH, IMG_HEIGHT, 0])
        
        # Plot all nodes
        for name, (x, y) in nodes.items():
            ax.scatter(x, y, color='blue', s=100, zorder=5)
            ax.text(x+10, y-10, name, fontsize=8, color='black', weight='bold')
        
        # Plot AGV path
        path_coords = [nodes[name] for name in path_names]
        if len(path_coords) > 1:
            xs, ys = zip(*path_coords)
            ax.plot(xs, ys, color='red', linewidth=3, marker='o', markersize=8, label='AGV Route', zorder=10)
        
        ax.set_xlim(0, IMG_WIDTH)
        ax.set_ylim(IMG_HEIGHT, 0)
        ax.legend()
        ax.axis('off')
        st.pyplot(fig)


else:
    st.info("Upload a CSV with columns SKU_ID, Product_Category, Dimensions_CM(LxWxH), Weight_KG, Storage_Type, Average_Daily_Orders")
