import json
import networkx as nx
from pyvis.network import Network

# loading JSON data
with open('data/mystery_graph_2.json', 'r', encoding='utf-8') as f:
    triplets = json.load(f)

# creating an empty Directed Graph
G = nx.DiGraph()

# adding nodes and edges
for item in triplets:
    subject = item.get("subject")
    relation = item.get("relation")
    obj = item.get("object")

    G.add_edge(subject, obj, title=relation, label=relation)

# visualizing the network using PyVis
net = Network(notebook=False, directed=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(G)

# physics settings to push nodes apart for better readability
net.repulsion(node_distance=300, spring_length=250)
net.show_buttons(filter_=['physics'])

# saving and exporting as HTML
output_file = "knowledge_graph.html"
net.save_graph(output_file)
print(f"Graph successfully generated! You can open {output_file} in any web browser.")