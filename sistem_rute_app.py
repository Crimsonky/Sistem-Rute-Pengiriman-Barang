import streamlit as st
import pandas as pd
import networkx as nx
import folium
from folium import Marker
from streamlit_folium import st_folium
import heapq
import time
import tracemalloc
import matplotlib.pyplot as plt

# ===== Load Data =====
nodes_df = pd.read_csv('transportation_nodes.csv')
edges_df = pd.read_csv('transportation_edges_augmented.csv')

G = nx.DiGraph()
id_to_label = {}
label_to_id = {}

for _, row in nodes_df.iterrows():
    G.add_node(row['id'], label=row['name'], **row.to_dict())
    id_to_label[row['id']] = row['name']
    label_to_id[row['name']] = row['id']

def parse_congestion(cong):
    if pd.isna(cong):
        return 1.0
    try:
        return float(cong)
    except (ValueError, TypeError):
        pass

    s = str(cong).strip().lower()
    mapping = {
        'rendah': 1,
        'sedang': 2,
        'tinggi': 3
    }
    return mapping.get(s, 1.0)

def compute_weight(distance_km, avg_speed_kmh, congestion, direction):
    speed = avg_speed_kmh if avg_speed_kmh and avg_speed_kmh > 0 else 1.0
    cong_val = parse_congestion(congestion)
    time_hours = distance_km / speed
    cong_factor = 1 + cong_val
    dir_factor = 1.5 if direction == 1 else 1.0
    return time_hours * cong_factor * dir_factor

for _, row in edges_df.iterrows():
    weight = compute_weight(
        distance_km=row['distance_km'],
        avg_speed_kmh=row['avg_speed_kmh'],
        congestion=row['congestion'],
        direction=row['direction']
    )
    G.add_edge(
        row['from'],
        row['to'],
        weight=weight,
        label=round(weight, 4),
        **row.to_dict()
    )

def dijkstra(graph, start, end):
    distances = {n: float('inf') for n in graph.nodes}
    distances[start] = 0
    predecessors = {}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        if current_node == end:
            path = []
            while current_node in predecessors:
                path.append(current_node)
                current_node = predecessors[current_node]
            path.append(start)
            path.reverse()
            return path, distances[end]

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return None, float('inf')

def a_star(graph, start, end, heuristic):
    open_set = [(0, start)]
    came_from = {}
    g = {n: float('inf') for n in graph.nodes}
    f = {n: float('inf') for n in graph.nodes}
    g[start] = 0
    f[start] = heuristic(start, end)

    while open_set:
        _, node = heapq.heappop(open_set)
        if node == end:
            path = []
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return list(reversed(path)), g[end]

        for neighbor in graph.neighbors(node):
            tentative_g = g[node] + graph[node][neighbor]['weight']
            if tentative_g < g[neighbor]:
                came_from[neighbor] = node
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f[neighbor], neighbor))
    return None, float('inf')

def heuristic_distance(n1, n2):
    try:
        from math import radians
        from sklearn.metrics.pairwise import haversine_distances
        lat1, lon1 = radians(G.nodes[n1]['latitude']), radians(G.nodes[n1]['longitude'])
        lat2, lon2 = radians(G.nodes[n2]['latitude']), radians(G.nodes[n2]['longitude'])
        return haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1] * 6371
    except:
        return 0

def multi_vehicle_routing(graph, vehicle_data, algorithm):
    routes = {}
    deliveries = sorted(enumerate(vehicle_data), key=lambda x: x[1]['priority'])  
    for idx, (vehicle_idx, v) in enumerate(deliveries):
        if v['weight'] <= v['capacity']:
            path, cost = (dijkstra(graph, v['start'], v['end']) if algorithm == 'dijkstra'
                          else a_star(graph, v['start'], v['end'], heuristic_distance))
            if path:
                distance = sum(graph[path[j]][path[j+1]].get('distance_km', 0) for j in range(len(path)-1))
                routes[idx] = {
                    'vehicle': vehicle_idx, 
                    'start': v['start'],
                    'end': v['end'],
                    'path': path,
                    'cost': cost,
                    'distance': distance
                }
    return routes

# ===== Streamlit App =====
st.set_page_config(layout="wide")
st.title("🚚 Sistem Rute Pengiriman Barang - Tebet, Jakarta Selatan")

st.sidebar.header("🛠️ Panel Kontrol")

depot_nodes_df = nodes_df[nodes_df['name'].str.lower().str.contains("depot")]
ruko_gedung_nodes_df = nodes_df[nodes_df['name'].str.lower().str.contains("ruko|gedung")]

max_vehicles = 15
num_vehicles = st.sidebar.slider("Jumlah Kendaraan", min_value=1, max_value=max_vehicles, value=1)

vehicle_data = []
st.sidebar.subheader("🚗 Detail Kendaraan")
for i in range(num_vehicles):
    with st.sidebar.expander(f"Kendaraan {i+1}"):
        start = st.selectbox(f"Start (Depot) - Kendaraan {i+1}", depot_nodes_df['name'], key=f"start_{i}")
        end = st.selectbox(f"End (Ruko/Gedung) - Kendaraan {i+1}", ruko_gedung_nodes_df['name'], key=f"end_{i}")
        weight = st.number_input(f"Berat Pengiriman (Ton) - Kendaraan {i+1}", min_value=0.0, max_value=15.0, value=1.0, step=0.1, key=f"weight_{i}")
        priority = st.number_input(f"Prioritas - Kendaraan {i+1}", min_value=1, max_value=15, value=1, key=f"priority_{i}")
        capacity = st.number_input(f"Kapasitas Kendaraan (Ton) - Kendaraan {i+1}", min_value=0.0, max_value=15.0, value=2.0, step=0.1, key=f"capacity_{i}")
        vehicle_data.append({
            'start': label_to_id[start],
            'end': label_to_id[end],
            'weight': weight,
            'priority': priority,
            'capacity': capacity
        })

algorithm = st.sidebar.selectbox("Algoritma", ['dijkstra', 'a_star'])

show_performance = st.sidebar.checkbox("Visualisasi Performa Komputasi")
visualization_option = st.sidebar.selectbox("Jenis Visualisasi", ["Peta Folium", "Graph (NetworkX)"])

# ===== Performance Measurement Dinamis =====
start_time = time.time()
tracemalloc.start()

routes = multi_vehicle_routing(G, vehicle_data, algorithm)

current, peak = tracemalloc.get_traced_memory()
end_time = time.time()
tracemalloc.stop()

execution_time = (end_time - start_time)
memory_usage = peak / 10**6

# ===== Visualisasi =====
if visualization_option == "Peta Folium":
    st.subheader("🗺️ Peta Visualisasi Rute")
    m = folium.Map(location=[-6.2297, 106.8532], zoom_start=15)

    for node in G.nodes:
        lat, lon = G.nodes[node].get('latitude'), G.nodes[node].get('longitude')
        label = G.nodes[node].get('label', str(node))
        if pd.notna(lat) and pd.notna(lon):
            Marker([lat, lon], tooltip=label, icon=folium.Icon(color="gray", icon="circle")).add_to(m)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black', 'pink']
    for idx, route in routes.items():
        coords = [[G.nodes[n]['latitude'], G.nodes[n]['longitude']] for n in route['path'] if pd.notna(G.nodes[n]['latitude'])]
        if coords:
            folium.PolyLine(coords, color=colors[idx % len(colors)], weight=5,
                            tooltip=f"Rute Kendaraan {route['vehicle']+1}").add_to(m)

            start_label = id_to_label.get(route['start'], str(route['start']))
            end_label = id_to_label.get(route['end'], str(route['end']))

            folium.Marker(coords[0], icon=folium.Icon(color="green"),
                          tooltip=f"Start ({start_label})").add_to(m)
            folium.Marker(coords[-1], icon=folium.Icon(color="red"),
                          tooltip=f"End ({end_label})").add_to(m)

    st_data = st_folium(m, width=900, height=500)

elif visualization_option == "Graph (NetworkX)":
    st.subheader("🔗 Visualisasi Graph Jaringan Jalan")
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) for node in G.nodes 
           if pd.notna(G.nodes[node].get('latitude')) and pd.notna(G.nodes[node].get('longitude'))}

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue', ax=ax)
    nx.draw_networkx_edges(G, pos, width=1, edge_color='lightgray', alpha=0.7, ax=ax)

    # Tambahkan label nama lokasi di setiap node
    labels = {node: G.nodes[node]['label'] for node in G.nodes if 'label' in G.nodes[node]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # Highlight rute kendaraan
    edge_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black', 'pink']
    for idx, route in routes.items():
        path_edges = list(zip(route['path'][:-1], route['path'][1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color=edge_colors[idx % len(edge_colors)], ax=ax)

    plt.title("Graph Transportation Network")
    plt.axis('off')
    st.pyplot(fig)


# ===== Dashboard Performa & Biaya =====
st.subheader("📊 Metrik Performa & Biaya Pengiriman")
for idx, (key, route) in enumerate(routes.items(), start=1):
    s_label = id_to_label.get(route['start'], str(route['start']))
    e_label = id_to_label.get(route['end'], str(route['end']))
    path_labels = [id_to_label.get(n, str(n)) for n in route['path']]
    route_str = " ➔ ".join(path_labels)

    st.markdown(f"**[Urutan ke-{idx}] - Kendaraan {route['vehicle']+1} rute dari '{s_label}' ke '{e_label}':**")
    st.write(route_str)
    st.write(f"• Total Jarak: **{route['distance']:.2f} km**")
    st.write(f"• Total Waktu (estimasi): **{route['cost'] * 60:.0f} menit**")

    biaya_per_km = 5000  # Rp per km
    total_biaya = route['distance'] * biaya_per_km

    st.write(f"• Total Biaya Pengiriman: **Rp {total_biaya:,.0f}**")

# ===== Grafik Analisis Kinerja =====
if show_performance:
    st.subheader(f"⏱️ Analisis Kinerja Algoritma: {algorithm.upper()}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Waktu Komputasi", value=f"{execution_time:.4f} detik")

    with col2:
        st.metric(label="Penggunaan Memori", value=f"{memory_usage:.4f} MB")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].bar(['Waktu Komputasi'], [execution_time], color='skyblue')
    ax[0].set_title('Waktu Komputasi (detik)')
    ax[0].set_ylabel('Detik')

    ax[1].bar(['Penggunaan Memori'], [memory_usage], color='lightgreen')
    ax[1].set_title('Penggunaan Memori (MB)')
    ax[1].set_ylabel('MB')
    st.pyplot(fig)
