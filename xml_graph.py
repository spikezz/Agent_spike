import click
import networkx as nx
import matplotlib.pyplot as plt

@click.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='Path to the GraphML file')
def visualize_graph(file):
    # Read the GraphML file
    graph = nx.read_graphml(file)

    # Draw the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph)  # positions for all nodes

    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_size=700)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=12)

    # Show the plot
    plt.title("Knowledge Graph Visualization")
    plt.show()

if __name__ == '__main__':
    visualize_graph()