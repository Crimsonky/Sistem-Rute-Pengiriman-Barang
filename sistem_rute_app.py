import streamlit as st
import pandas as pd
import networkx as nx
import folium
from folium import Marker, PolyLine
from streamlit_folium import st_folium
import heapq

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
        'Rendah':  0.2,
        'Sedang':  0.5,
        'Tinggi':  0.8
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

# ===== Algoritma =====
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



# def a_star(graph, start, end, heuristic):
#     open_set = [(0, start)]
#     came_from = {}
#     g_score = {n: float('inf') for n in graph.nodes}
#     g_score[start] = 0
#     f_score = {n: float('inf') for n in graph.nodes}
#     f_score[start] = heuristic(start, end)

#     while open_set:
#         current_node = heapq.heappop(open_set)

#         if current_node == end:
#             path = []
#             while current_node in came_from:
#                 path.append(current_node)
#                 current_node = came_from[current_node]
#             path.append(start)
#             path.reverse()
#             return path, g_score[end]

#         for neighbor in graph.neighbors(current_node):
#             tentative_g_score = g_score[current_node] + graph[current_node][neighbor]['weight']
#             if tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current_node
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
#                 if (f_score[neighbor], neighbor) not in open_set:
#                     heapq.heappush(open_set, (f_score[neighbor], neighbor))

#     return None, float('inf')

def heuristic_distance(n1, n2):
    try:
        from math import radians
        from sklearn.metrics.pairwise import haversine_distances
        lat1, lon1 = radians(G.nodes[n1]['latitude']), radians(G.nodes[n1]['longitude'])
        lat2, lon2 = radians(G.nodes[n2]['latitude']), radians(G.nodes[n2]['longitude'])
        return haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1] * 6371
    except:
        return 0

# def heuristic_distance(n1, n2):
#   try:
#       node1 = G.nodes[n1]
#       node2 = G.nodes[n2]
#       if pd.notna(node1['latitude']) and pd.notna(node1['longitude']) and pd.notna(node2['latitude']) and pd.notna(node2['longitude']):
#           from sklearn.metrics.pairwise import haversine_distances
#           from math import radians
#           node1_loc = (radians(node1['latitude']), radians(node1['longitude']))
#           node2_loc = (radians(node2['latitude']), radians(node2['longitude']))

#           return haversine_distances([node1_loc, node2_loc])[0,1] * 6371
#       else:
#           return 0
#   except KeyError:
#         return 0
      
# ===== Streamlit UI =====
st.set_page_config(layout="wide")
st.title("🚚 Sistem Rute Pengiriman Barang - Tebet, Jakarta Selatan")

# ===== Sidebar Panel =====
st.sidebar.header("🛠️ Panel Kontrol")

# --- Filter Berdasarkan Nama Lokasi ---
depot_nodes_df = nodes_df[nodes_df['name'].str.lower().str.contains("depot")]
ruko_gedung_nodes_df = nodes_df[nodes_df['name'].str.lower().str.contains("ruko|gedung")]

depot_labels = list(depot_nodes_df['name'])
ruko_gedung_labels = list(ruko_gedung_nodes_df['name'])

start_input = st.sidebar.text_input("Start Nodes (Depot/Gudang, pisahkan dengan koma)", ",".join(depot_labels[:1]))
start_labels = [s.strip() for s in start_input.split(',') if s.strip() in label_to_id]
end_labels = st.sidebar.multiselect("End Nodes (Ruko/Gedung)", ruko_gedung_labels, default=ruko_gedung_labels[:1])

priorities = st.sidebar.text_input("Prioritas (dipisah koma)", "1,2")
demands = st.sidebar.text_input("Berat Pengiriman (Kg) (dipisah koma)", "10,20")
capacities = st.sidebar.text_input("Kapasitas Kendaraan (Kg) (dipisah koma)", "30,30")
algorithm = st.sidebar.selectbox("Algoritma", ['dijkstra', 'a_star'])

# --- Konversi Input ---
priorities = list(map(int, priorities.split(',')))
demands = list(map(int, demands.split(',')))
capacities = list(map(int, capacities.split(',')))
start_nodes = [label_to_id[l] for l in start_labels]
end_nodes = [label_to_id[l] for l in end_labels]

# ===== Routing =====
def multi_vehicle_routing(graph, start_nodes, end_nodes, priorities, demands, capacities, algorithm):
    routes = {}
    remaining = capacities.copy()
    deliveries = sorted(zip(start_nodes, end_nodes, priorities, demands), key=lambda x: x[2])
    for s, e, p, d in deliveries:
        for i, cap in enumerate(remaining):
            if d <= cap:
                path, cost = (dijkstra(graph, s, e) if algorithm == 'dijkstra' else a_star(graph, s, e, heuristic_distance))
                if path:
                    distance = sum(graph[path[j]][path[j+1]].get('distance_km', 0) for j in range(len(path)-1))
                    routes[(s, e)] = {'vehicle': i, 'path': path, 'cost': cost, 'distance': distance}
                    remaining[i] -= d
                break
    return routes

routes = multi_vehicle_routing(G, start_nodes, end_nodes, priorities, demands, capacities, algorithm)

# ===== Folium Map =====
st.subheader("🗺️ Peta Visualisasi Rute")
m = folium.Map(location=[-6.2297, 106.8532], zoom_start=15)

for node in G.nodes:
    lat, lon = G.nodes[node].get('latitude'), G.nodes[node].get('longitude')
    label = G.nodes[node].get('label', str(node))
    if pd.notna(lat) and pd.notna(lon):
        Marker([lat, lon], tooltip=label, icon=folium.Icon(color="gray", icon="circle")).add_to(m)

colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx, ((start, end), route) in enumerate(routes.items()):
    coords = [[G.nodes[n]['latitude'], G.nodes[n]['longitude']] for n in route['path'] if pd.notna(G.nodes[n]['latitude'])]
    if coords:
        folium.PolyLine(coords, color=colors[idx % len(colors)], weight=5,
                        tooltip=f"Rute Kendaraan {route['vehicle']}").add_to(m)

        start_label = id_to_label.get(start, str(start))
        end_label = id_to_label.get(end, str(end))

        folium.Marker(coords[0], icon=folium.Icon(color="green"),
                      tooltip=f"Start ({start_label})").add_to(m)
        folium.Marker(coords[-1], icon=folium.Icon(color="red"),
                      tooltip=f"End ({end_label})").add_to(m)

st_data = st_folium(m, width=900, height=500)

# ===== Dashboard =====
st.subheader("📊 Metrik Performa Rute")
for (start, end), route in routes.items():
    s_label = id_to_label[start]
    e_label = id_to_label[end]
    path_labels = [id_to_label.get(n, str(n)) for n in route['path']]

    route_str = " ➔ ".join(path_labels)

    st.markdown(f"**🚗 [Kendaraan {route['vehicle']}] Rute dari '{s_label}' ke '{e_label}':**")
    st.write(route_str)
    st.write(f"• Total Jarak: **{route['distance']:.2f} km**")
    st.write(f"• Total Waktu (estimasi): **{route['cost']:.2f} jam**")

