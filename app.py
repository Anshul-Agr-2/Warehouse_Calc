import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

@st.cache_data
def load_image(path):
    return Image.open(path)

@st.cache_data
def load_and_process_csv(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower().str.replace("(", "").str.replace(")", "")
    df = df.rename(columns={
        "dimensions_cm_lxwxh":"dimensions",
        "weight_kg":"weight",
        "storage_type":"storage",
        "average_daily_orders":"avg_orders",
        "product_category":"category"
    })
    return df

IMG_PATH = "IMAGE111.jpg"  # Change to your image filename
IMG_WIDTH, IMG_HEIGHT = 718, 967
# IMG_WIDTH, IMG_HEIGHT = 718, 967

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
    df = load_and_process_csv(uploaded)
    # Track active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Create tabs - Streamlit doesn't support programmatic selection yet, so we use a workaround
    tab_names = ["Classification & Layout", "SKU Demand Forecaster", "AGV Routing"]
    selected_tab = st.radio("Navigate:", tab_names, index=st.session_state.active_tab, horizontal=True, label_visibility="collapsed")
    st.session_state.active_tab = tab_names.index(selected_tab)
    
    if selected_tab == "Classification & Layout":
        st.header("📦 ABC Classification Analysis")
        st.markdown("**Pareto-Based Inventory Segmentation**")
        a_pct = st.slider("Class A Threshold (High-Value Items)", 0.05, 0.90, 0.3, help="Top % of SKUs by order frequency")
        b_pct = st.slider("Class B Threshold (Medium-Value Items)", a_pct+0.01, 0.99, 0.7, help="Next % of SKUs by order frequency")

        df_sorted = df.sort_values("avg_orders", ascending=False).reset_index(drop=True)
        n_skus = len(df_sorted)
        a_cut, b_cut = int(a_pct * n_skus), int(b_pct * n_skus)
        df_sorted["abc"] = ["A"]*a_cut + ["B"]*(b_cut-a_cut) + ["C"]*(n_skus-b_cut)
        st.info(f"**Class Distribution:** A = {a_cut} SKUs | B = {b_cut-a_cut} SKUs | C = {n_skus-b_cut} SKUs")
        st.dataframe(df_sorted[["sku_id","avg_orders","abc"]], use_container_width=True)

        st.header("📊 XYZ Variability Analysis")
        st.markdown("**Demand Stability Classification**")
        mean_orders = df["avg_orders"].mean()
        std_orders = df["avg_orders"].std()
        x_cv = st.slider("Class X Threshold (Low Variability)", 0.3, 1.3, 0.7, help="Coefficient of variation relative to mean")
        y_cv = st.slider("Class Y Threshold (Medium Variability)", x_cv+0.01, 3.0, 1.3, help="Coefficient of variation relative to mean")
        df_sorted["cv"] = df_sorted["avg_orders"] / (mean_orders + 1e-9)
        df_sorted["xyz"] = df_sorted["cv"].apply(lambda v: "X" if v <= x_cv else ("Y" if v <= y_cv else "Z"))
        st.dataframe(df_sorted[["sku_id","avg_orders","xyz"]], use_container_width=True)

        st.header("⚖️ Weight-Based Classification")
        st.markdown("**Physical Handling Requirements**")
        w_edges = st.slider("Weight Category Boundaries (kg)", 0.1, 10.0, (1.0,3.0), help="Define Light/Medium/Heavy thresholds")
        weight_bins = pd.cut(df_sorted["weight"], bins=[0,w_edges[0],w_edges[1],df_sorted["weight"].max()+1], labels=["Light","Medium","Heavy"])

        df_sorted["weight_bin"] = weight_bins.astype(str).fillna("")
        st.dataframe(df_sorted[["sku_id","weight","weight_bin"]], use_container_width=True)
        csv_classification = df_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Export Classification Data (CSV)",
            data=csv_classification,
            file_name='warehouse_classification.csv',
            mime='text/csv'
        )

        st.header("📊 Class Distribution Analytics")
        fig1, axs = plt.subplots(1,3,figsize=(12,4), facecolor='white')
        
        # ABC Pie Chart - Professional gradient
        abc_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        axs[0].pie(df_sorted['abc'].value_counts(), labels=df_sorted['abc'].value_counts().index, 
                   autopct='%1.1f%%', colors=abc_colors, startangle=90,
                   wedgeprops={'edgecolor': 'white', 'linewidth': 2}, 
                   textprops={'fontsize': 10, 'weight': 'bold'})
        axs[0].set_title("ABC Classification", fontsize=12, weight='bold', pad=15)
        
        # XYZ Pie Chart - Cool tones
        xyz_colors = ['#FFA07A', '#98D8C8', '#F7DC6F']
        axs[1].pie(df_sorted['xyz'].value_counts(), labels=df_sorted['xyz'].value_counts().index, 
                   autopct='%1.1f%%', colors=xyz_colors, startangle=90,
                   wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                   textprops={'fontsize': 10, 'weight': 'bold'})
        axs[1].set_title("Variability (XYZ)", fontsize=12, weight='bold', pad=15)
        
        # Weight Pie Chart - Earthly tones
        weight_colors = ['#BB8FCE', '#85C1E2', '#F8B739']
        axs[2].pie(df_sorted['weight_bin'].value_counts(), labels=df_sorted['weight_bin'].value_counts().index, 
                   autopct='%1.1f%%', colors=weight_colors, startangle=90,
                   wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                   textprops={'fontsize': 10, 'weight': 'bold'})
        axs[2].set_title("Weight Categories", fontsize=12, weight='bold', pad=15)
        
        plt.tight_layout()
        st.pyplot(fig1)


        st.header("Warehouse Map Visualization")
        img = load_image(IMG_PATH)
        IMG_WIDTH, IMG_HEIGHT = img.size
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img)

        
        # Define exact coordinates from your input
        STORAGE_COORDS = {
            "A": {"Rack": (215, 200), "Shelf": (400, 200), "Bin": (555, 200)},
            "B": {"Rack": (215, 365), "Shelf": (400, 365), "Bin": (555, 365)},
            "C": {"Rack": (215, 530), "Shelf": (400, 530), "Bin": (555, 530)}
        }
        
        # Plot storage type headers (Rack, Shelf, Bin) - ZONE headers removed
        # Plot storage type headers (Rack, Shelf, Bin) - ZONE headers removed
        # Define custom y-offsets for each zone
        zone_offsets = {"A": -10, "B": -10, "C": -20}  # Adjust these values as needed
        
        for zone, coords_dict in STORAGE_COORDS.items():
            for storage_type, (x, y) in coords_dict.items():
                y_offset = zone_offsets[zone]  # Get zone-specific offset
                ax.text(x, y + y_offset, storage_type, fontsize=10, 
                        color='white', weight="bold", ha="center",
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#2E4057', edgecolor='white', linewidth=1.5, alpha=0.95))


        
        # Plot SKUs as dots with labels (vertical column layout)
        # Plot SKUs as dots with labels (vertical column layout)
        offset_y = 15  # Vertical spacing between SKUs
        zone_sku_offsets = {"A": 10, "B": 10, "C": 0}  # Adjust starting Y offset per zone
        
        for zone in ["A", "B", "C"]:
            for storage_type in ["Rack", "Shelf", "Bin"]:
                # Filter SKUs by ABC zone and storage type
                skus = df_sorted[(df_sorted['abc'] == zone) & 
                                (df_sorted['storage'].str.lower() == storage_type.lower())]
                
                if len(skus) == 0:
                    continue
                
                # Get base coordinates
                base_x, base_y = STORAGE_COORDS[zone][storage_type]
                
                # Plot each SKU vertically
                for i, row in enumerate(skus.itertuples()):
                    sku_id = getattr(row, "sku_id", "")
                    wbin = getattr(row, "weight_bin", "")
                    
                    # Calculate position with zone-specific offset
                    x = base_x
                    y = base_y + i * offset_y + zone_sku_offsets[zone]
                    
                    # Plot smaller dot
                    ax.scatter(x, y, color=ZONE_COLOR[zone], s=30, edgecolors='black', linewidths=1.0, zorder=13)
                    
                    # Plot label (SKU ID) - smaller font
                    ax.text(x + 5, y, f"{sku_id}", fontsize=7, color='black', ha="left", va="center")
        
        ax.axis('off')
        st.pyplot(fig)


    elif selected_tab == "SKU Demand Forecaster":
        st.header("📈 Demand Forecasting Module")
        st.markdown("**Predictive Analytics for Inventory Planning**")
        horizon_days = st.slider("Forecast Time Horizon (Days)", 7, 180, 30, help="Projection period for demand estimation")
        forecast_type = st.selectbox("Forecasting Model", ["Linear Trend (avg × days)", "Exponential Growth (avg × days × 1.02^days)"], help="Select demand projection methodology")
        if "Linear" in forecast_type:
            df["forecast_orders"] = df["avg_orders"] * horizon_days
            model_desc = "Linear Extrapolation"
        else:
            df["forecast_orders"] = df["avg_orders"] * horizon_days * (1.02 ** horizon_days)
            model_desc = "Exponential Growth Model"
        st.info(f"**Projection Period:** {horizon_days} days | **Model:** {model_desc}")
        st.dataframe(df[["sku_id","avg_orders","forecast_orders","storage","weight"]], use_container_width=True)


        # Forecast Classification
        df_forecast = df.sort_values("forecast_orders", ascending=False).reset_index(drop=True)
        n_skus = len(df_forecast)
        st.markdown("---")
        st.subheader("📊 Forecasted Inventory Segmentation")
        af_pct = st.slider("Projected Class A Threshold", 0.05, 0.9, 0.3, help="High-demand items in forecast period")
        bf_pct = st.slider("Projected Class B Threshold", af_pct+0.01, 0.99, 0.7, help="Medium-demand items in forecast period")

        af_cut, bf_cut = int(af_pct * n_skus), int(bf_pct * n_skus)
        df_forecast["abc_forecast"] = ["A"]*af_cut + ["B"]*(bf_cut-af_cut) + ["C"]*(n_skus-bf_cut)
        mean_forecast = df_forecast["forecast_orders"].mean()
        std_forecast = df_forecast["forecast_orders"].std()
        xf_cv = st.slider("Projected Class X Threshold (Variability)", 0.3, 1.3, 0.8, help="Low variability in forecast")
        yf_cv = st.slider("Projected Class Y Threshold (Variability)", xf_cv+0.01, 3.0, 1.3, help="Medium variability in forecast")

        df_forecast["cv_forecast"] = df_forecast["forecast_orders"] / (mean_forecast + 1e-9)
        df_forecast["xyz_forecast"] = df_forecast["cv_forecast"].apply(lambda v: "X" if v <= xf_cv else ("Y" if v <= yf_cv else "Z"))
        bins = [0, 1.0, 3.0, df_forecast["weight"].max()+1]
        df_forecast["weight_bin_forecast"] = pd.cut(df_forecast["weight"], bins=bins, labels=["Light","Medium","Heavy"]).astype(str).fillna("")
        st.dataframe(df_forecast[["sku_id","forecast_orders","abc_forecast","xyz_forecast","weight_bin_forecast","storage"]], use_container_width=True)
        
        csv_forecast = df_forecast.to_csv(index=False)
        st.download_button(
            label="📥 Export Forecast Results (CSV)",
            data=csv_forecast,
            file_name='warehouse_forecast.csv',
            mime='text/csv'
        )
        
        st.header("📊Forecasted Demand Bar Chart")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_forecast['sku_id'], df_forecast['forecast_orders'], color="#4e79a7")
        ax.set_ylabel("Forecasted Orders")
        ax.set_xlabel("SKU ID")
        ax.set_title(f"Forecasted SKU Demand ({forecast_type})")
        ax.set_xticklabels(df_forecast['sku_id'], fontsize=8, rotation=30, ha='right')

        st.pyplot(fig)

        st.header("📈 Forecasted Distribution Analytics")
        fig2, axs2 = plt.subplots(1,3,figsize=(12,4), facecolor='white')
        
        # ABC Forecast - Professional gradient
        abc_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        axs2[0].pie(df_forecast['abc_forecast'].value_counts(), labels=df_forecast['abc_forecast'].value_counts().index, 
                    autopct='%1.1f%%', colors=abc_colors, startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                    textprops={'fontsize': 10, 'weight': 'bold'})
        axs2[0].set_title("ABC Classification", fontsize=12, weight='bold', pad=15)
        
        # XYZ Forecast
        xyz_colors = ['#FFA07A', '#98D8C8', '#F7DC6F']
        axs2[1].pie(df_forecast['xyz_forecast'].value_counts(), labels=df_forecast['xyz_forecast'].value_counts().index, 
                    autopct='%1.1f%%', colors=xyz_colors, startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                    textprops={'fontsize': 10, 'weight': 'bold'})
        axs2[1].set_title("Variability (XYZ)", fontsize=12, weight='bold', pad=15)
        
        # Weight Forecast
        weight_colors = ['#BB8FCE', '#85C1E2', '#F8B739']
        axs2[2].pie(df_forecast['weight_bin_forecast'].value_counts(), labels=df_forecast['weight_bin_forecast'].value_counts().index, 
                    autopct='%1.1f%%', colors=weight_colors, startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                    textprops={'fontsize': 10, 'weight': 'bold'})
        axs2[2].set_title("Weight Categories", fontsize=12, weight='bold', pad=15)
        
        plt.tight_layout()
        st.pyplot(fig2)

        
    elif selected_tab == "AGV Routing":
        st.header("🚛 Automated Guided Vehicle (AGV) Route Optimization")
        st.markdown("**Multi-Agent Path Planning & Fleet Management**")

        
        # Define nodes
        nodes = {
            "Receiving": (50, 85), "Inbound": (128, 85),
            "Zone_A_Rack": (215, 200), "Zone_A_Shelf": (400, 200), "Zone_A_Bin": (555, 200),
            "Zone_B_Rack": (215, 365), "Zone_B_Shelf": (400, 365), "Zone_B_Bin": (555, 365),
            "Zone_C_Rack": (215, 530), "Zone_C_Shelf": (400, 530), "Zone_C_Bin": (555, 530),
            "Outbound": (589, 85), "Shipping": (655, 85)
        }
        
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        node_names = list(nodes.keys())
        
        # AGV colors
        AGV_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
        
        # Distance matrix (cached)
        @st.cache_data
        def compute_distance_matrix():
            n = len(node_names)
            dist_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_mat[i, j] = euclidean_distance(nodes[node_names[i]], nodes[node_names[j]])
            return dist_mat
        
        distance_matrix = compute_distance_matrix()
        
        with st.expander("📊 View Distance Matrix", expanded=True):
            df_dist = pd.DataFrame(distance_matrix, index=node_names, columns=node_names)
            st.dataframe(df_dist.style.format("{:.1f}"))
        
        st.markdown("---")
        
        # Initialize session state with DEMO DATA
        if 'num_agvs' not in st.session_state:
            st.session_state.num_agvs = 2
        if 'agv_routes' not in st.session_state:
            # Pre-loaded demo routes
            st.session_state.agv_routes = {
                0: ["Receiving", "Zone_A_Rack", "Zone_B_Shelf", "Shipping"],
                1: ["Inbound", "Zone_C_Bin", "Zone_A_Shelf", "Outbound"]
            }
        if 'show_results' not in st.session_state:
            st.session_state.show_results = True  # Show demo on load
        
        # Number of AGVs selector
        num_agvs = st.number_input("Fleet Size (Number of AGVs)", min_value=1, max_value=5, value=st.session_state.num_agvs, step=1, help="Configure simultaneous vehicle operations")

        
        if num_agvs != st.session_state.num_agvs:
            st.session_state.num_agvs = num_agvs
            # Add default routes for new AGVs
            for i in range(num_agvs):
                if i not in st.session_state.agv_routes:
                    st.session_state.agv_routes[i] = ["Receiving", "Shipping"]
            # Remove extra AGVs
            keys_to_remove = [k for k in st.session_state.agv_routes.keys() if k >= num_agvs]
            for k in keys_to_remove:
                del st.session_state.agv_routes[k]
            st.session_state.show_results = False
        
        st.markdown("---")
        
        # Build routes for each AGV
        for agv_id in range(st.session_state.num_agvs):
            with st.expander(f"🤖 AGV {agv_id + 1} Route Builder", expanded=True):
                st.markdown(f"<span style='color:{AGV_COLORS[agv_id]};font-weight:bold;'>AGV {agv_id + 1}</span>", unsafe_allow_html=True)
                st.write(f"**Current Route:** {' → '.join(st.session_state.agv_routes[agv_id])}")
                
                # Display route
                for idx in range(len(st.session_state.agv_routes[agv_id])):
                    col1, col2, col3 = st.columns([0.5, 3, 1.5])
                    
                    with col1:
                        st.markdown(f"**`{idx + 1}`**")
                    with col2:
                        st.write(st.session_state.agv_routes[agv_id][idx])
                    with col3:
                        btn_cols = st.columns(3)
                        with btn_cols[0]:
                            if idx > 0:
                                if st.button("↑", key=f"agv{agv_id}_up_{idx}", use_container_width=True):
                                    temp = st.session_state.agv_routes[agv_id][idx]
                                    st.session_state.agv_routes[agv_id][idx] = st.session_state.agv_routes[agv_id][idx-1]
                                    st.session_state.agv_routes[agv_id][idx-1] = temp
                                    st.session_state.show_results = False
                                    st.rerun()
                        with btn_cols[1]:
                            if idx < len(st.session_state.agv_routes[agv_id]) - 1:
                                if st.button("↓", key=f"agv{agv_id}_down_{idx}", use_container_width=True):
                                    temp = st.session_state.agv_routes[agv_id][idx]
                                    st.session_state.agv_routes[agv_id][idx] = st.session_state.agv_routes[agv_id][idx+1]
                                    st.session_state.agv_routes[agv_id][idx+1] = temp
                                    st.session_state.show_results = False
                                    st.rerun()
                        with btn_cols[2]:
                            if len(st.session_state.agv_routes[agv_id]) > 2:
                                if st.button("✕", key=f"agv{agv_id}_del_{idx}", use_container_width=True):
                                    st.session_state.agv_routes[agv_id].pop(idx)
                                    st.session_state.show_results = False
                                    st.rerun()
                
                # Add stop
                col_add1, col_add2 = st.columns([3, 1])
                with col_add1:
                    new_loc = st.selectbox(f"Add stop to AGV {agv_id+1}:", node_names, key=f"agv{agv_id}_add_select")
                with col_add2:
                    st.markdown("<div style='padding-top: 26px;'></div>", unsafe_allow_html=True)
                    if st.button("➕", use_container_width=True, key=f"agv{agv_id}_add_btn"):
                        st.session_state.agv_routes[agv_id].append(new_loc)
                        st.session_state.show_results = False
                        st.rerun()
                
                # Quick actions
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("🔄 Reset", key=f"agv{agv_id}_reset", use_container_width=True):
                        st.session_state.agv_routes[agv_id] = ["Receiving", "Shipping"]
                        st.session_state.show_results = False
                with col_b2:
                    if st.button("⇄ Reverse", key=f"agv{agv_id}_reverse", use_container_width=True):
                        st.session_state.agv_routes[agv_id] = list(reversed(st.session_state.agv_routes[agv_id]))
                        st.session_state.show_results = False
        
        st.markdown("---")
        
        # Calculate button
        if st.button("🚀 Calculate All Routes & Visualize", type="primary", use_container_width=True):
            st.session_state.show_results = True
        
        # Results
        if st.session_state.show_results:
            st.markdown("---")
            st.subheader("📊 Fleet Analysis")
            
            # Calculate distances for all AGVs
            all_stats = []
            for agv_id in range(st.session_state.num_agvs):
                total_dist = 0
                segments = []
                route = st.session_state.agv_routes[agv_id]
                for i in range(len(route) - 1):
                    dist = euclidean_distance(nodes[route[i]], nodes[route[i+1]])
                    total_dist += dist
                    segments.append((route[i], route[i+1], dist))
                all_stats.append({
                    'agv_id': agv_id,
                    'stops': len(route),
                    'distance': total_dist,
                    'segments': segments
                })
            
            # Summary metrics
            cols = st.columns(st.session_state.num_agvs)
            for idx, stat in enumerate(all_stats):
                with cols[idx]:
                    st.markdown(f"<span style='color:{AGV_COLORS[idx]};font-weight:bold;'>AGV {idx+1}</span>", unsafe_allow_html=True)
                    st.metric("Stops", stat['stops'])
                    st.metric("Distance", f"{stat['distance']:.1f} px")
            
            st.markdown("---")
            
            # Detailed breakdown
            for stat in all_stats:
                with st.expander(f"AGV {stat['agv_id']+1} Route Breakdown", expanded=True):
                    for idx, (start, end, dist) in enumerate(stat['segments'], 1):
                        st.write(f"**{idx}.** {start} → {end} : `{dist:.1f} px`")
            
            st.markdown("---")
            st.subheader("🗺️ Fleet Visualization")
            
            @st.cache_data
            def load_warehouse_image():
                return Image.open(IMG_PATH)
            
            img = load_warehouse_image()
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(img)
            
            # Plot all nodes
            for name, (x, y) in nodes.items():
                ax.scatter(x, y, color='#2E4057', s=70, edgecolors='white', linewidths=2, zorder=5, alpha=0.9)
                ax.text(x+14, y+6, name, fontsize=6, color='#2E4057', weight='bold', 
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2E4057', linewidth=1.5, alpha=0.95))
            
            # Plot all AGV routes
            for agv_id, stat in enumerate(all_stats):
                route = st.session_state.agv_routes[agv_id]
                route_coords = [nodes[stop] for stop in route]
                color = AGV_COLORS[agv_id]
                
                if len(route_coords) > 1:
                    xs, ys = zip(*route_coords)
                    ax.plot(xs, ys, color=color, linewidth=2.5, marker='o', markersize=5, 
                           label=f'AGV {agv_id+1}: {stat["distance"]:.1f}px', zorder=10, alpha=0.7, linestyle='--')
                    
                    # Numbering
                    for idx, (x, y) in enumerate(route_coords, 1):
                        ax.text(x-12-agv_id*8, y-12-agv_id*8, str(idx), fontsize=7, color='white', weight='bold',
                               bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='white', linewidth=1.5))
            
            ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
            ax.axis('off')
            st.pyplot(fig)

else:
    st.info("Upload a CSV with columns SKU_ID, Product_Category, Dimensions_CM(LxWxH), Weight_KG, Storage_Type, Average_Daily_Orders")
