import pandas as pd

def main():
    print("Loading annotations...")
    ann = pd.read_feather("data/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    print(f"Annotations shape: {ann.shape}")
    
    # Traced neurons
    traced_mask = ann['status'] == 'Traced'
    traced_bodies = set(ann.loc[traced_mask, 'bodyId'])
    num_traced = len(traced_bodies)
    print(f"Number of traced neurons: {num_traced}")

    print("Loading connectome weights...")
    weights = pd.read_feather("data/connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights columns: {weights.columns}")
    print(weights.head())

    # Connections between traced neurons
    # Let's inspect column names in weights (usually 'bodyId_pre' and 'bodyId_post' or 'pre' and 'post' or 'body_pre' and 'body_post' or 'weight')
    col_pre = [c for c in weights.columns if 'pre' in c.lower()][0]
    col_post = [c for c in weights.columns if 'post' in c.lower()][0]
    col_weight = [c for c in weights.columns if 'weight' in c.lower() or 'syn' in c.lower() or 'count' in c.lower()][0]
    
    print(f"Identified columns: pre={col_pre}, post={col_post}, weight={col_weight}")
    
    # Filter connections where both pre and post are in traced_bodies
    is_traced_conn = weights[col_pre].isin(traced_bodies) & weights[col_post].isin(traced_bodies)
    traced_weights = weights[is_traced_conn]
    
    num_connections = len(traced_weights)
    total_synapses = traced_weights[col_weight].sum()
    
    print(f"Number of connections between traced neurons: {num_connections}")
    print(f"Number of synapses carried by those connections: {total_synapses}")

if __name__ == "__main__":
    main()
