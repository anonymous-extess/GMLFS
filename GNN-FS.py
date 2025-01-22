def create_feature_label_data_object(X, Y, edge_index, edge_weight, G, corr_fl):
    
    X_np = X.copy()
    Y_np = Y.copy()
 
    X_np, Y_np = add_random_one_if_all_zero(X_np, Y_np)

    X_tensor = torch.tensor(X_np, dtype=torch.float)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float)

    n_samples, n_features = X_np.shape
    n_labels = Y_np.shape[1]
    N = n_features + n_labels  


    mean = torch.mean(X_tensor, dim=0).unsqueeze(1)  # (n_features, 1)
    std = torch.std(X_tensor, dim=0).unsqueeze(1)  # (n_features, 1)
    median = torch.median(X_tensor, dim=0).values.unsqueeze(1)
    variance = torch.var(X_tensor, dim=0).unsqueeze(1)

  
    degrees_feat = torch.tensor([val for _, val in G.degree(range(n_features))],
                                dtype=torch.float).unsqueeze(1)
    clustering_feat = torch.tensor(list(nx.clustering(G, nodes=range(n_features)).values()),
                                   dtype=torch.float).unsqueeze(1)

    betweenness = nx.betweenness_centrality(G)

    betweenness_feat = torch.tensor(
        [betweenness[i] for i in range(n_features)],
        dtype=torch.float
    ).unsqueeze(1)





    chi2_values, _ = chi2(X_np, Y_np)
    chi2_values = torch.tensor(chi2_values, dtype=torch.float).unsqueeze(1)


    file_path = "arts-mi_sum.pt"

 
    if os.path.exists(file_path):
        mi_sum = torch.load(file_path)
    else:
        mi_sum = compute_multilabel_mutual_info(X_np, Y_np)
        mi_sum = torch.tensor(mi_sum, dtype=torch.float).unsqueeze(1)
        torch.save(mi_sum, file_path)


    feature_nodes_stats = torch.cat([
        mean, std, median, variance,
        degrees_feat, clustering_feat,
        betweenness_feat,
        chi2_values,mi_sum
    ], dim=1)  # shape: (n_features, D_f)


    avg_label_corr = torch.tensor(
        [np.mean(corr_fl[:, j]) for j in range(n_labels)],
        dtype=torch.float
    ).unsqueeze(1)


    degrees_label = torch.tensor([val for _, val in G.degree(range(n_features, N))],
                                 dtype=torch.float).unsqueeze(1)
    clustering_label = torch.tensor(list(nx.clustering(G, nodes=range(n_features, N)).values()),
                                    dtype=torch.float).unsqueeze(1)
    betweenness_label = torch.tensor(
        [betweenness[i] for i in range(n_features, N)],
        dtype=torch.float
    ).unsqueeze(1)


    label_nodes_stats = torch.cat([
        avg_label_corr,
        degrees_label,
        clustering_label,
        betweenness_label,
    ], dim=1)  # shape: (n_labels, D_l)

 
    D_f = feature_nodes_stats.shape[1]
    D_l = label_nodes_stats.shape[1]
    max_D = max(D_f, D_l)

    if D_f < max_D:
        pad_f = max_D - D_f
        feature_nodes_stats = torch.cat([feature_nodes_stats,
                                         torch.zeros(n_features, pad_f)], dim=1)
    if D_l < max_D:
        pad_l = max_D - D_l
        label_nodes_stats = torch.cat([label_nodes_stats,
                                       torch.zeros(n_labels, pad_l)], dim=1)


    x_feature_label = torch.cat([feature_nodes_stats, label_nodes_stats], dim=0)  # (N, max_D)


    data = Data(
        x=x_feature_label,
        edge_index=edge_index.clone(),
        edge_attr=edge_weight.clone()
    )
    return data



class FeatureSelectLayer(nn.Module):
    def __init__(self, num_features, total_nodes, l1_lambda=1.0, k=None):
        super(FeatureSelectLayer, self).__init__()
        self.num_features = num_features
        self.total_nodes = total_nodes
        self.l1_lambda = l1_lambda
        self.k = k
     
        self.mask = nn.Parameter(torch.ones(num_features) * 0.9999)

    def forward(self, x, selection=True):
        mask = F.relu(self.mask)  # [num_features]

        if selection and self.k is not None:
            topk_values, _ = torch.topk(mask, self.k, largest=True, sorted=True)
            threshold = topk_values[-1]
            mask = torch.where(mask >= threshold, mask, torch.zeros_like(mask))  # [num_features]

        label_mask = torch.ones(self.total_nodes - self.num_features, device=x.device)  # [n_labels]
        full_mask = torch.cat([mask, label_mask], dim=0).unsqueeze(1)  # [N, 1]

        x = x * full_mask  # [N, D]

        return x