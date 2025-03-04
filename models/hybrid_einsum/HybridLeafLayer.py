from .Layer import Layer
from .ExponentialFamilyArray import *

class HybridLeafLayer(Layer):
    """
    Computes all EiNet leaves in parallel, where each leaf is a vector of factorized distributions, where factors are
    from exponential families.

    In FactorizedLeafLayer, we generate an ExponentialFamilyArray with array_shape = (num_dist, num_replica), where
        num_dist is the vector length of the vectorized distributions (K in the paper), and
        num_replica is picked large enough such that "we compute enough leaf densities". At the moment we rely that
            the PC structure (see Class Graph) provides the necessary information to determine num_replica. In
            particular, we require that each leaf of the graph has the field einet_address.replica_idx defined;
            num_replica is simply the max over all einet_address.replica_idx.
            In the future, it would convenient to have an automatic allocation of leaves to replica, without requiring
            the user to specify this.
    The generate ExponentialFamilyArray has shape (batch_size, num_var, num_dist, num_replica). This array of densities
    will contain all densities over single RVs, which are then multiplied (actually summed, due to log-domain
    computation) together in forward(...).
    """

    def __init__(self, leaves, num_dims, exponential_family_args, use_em=False):
        """
        :param leaves: list of PC leaves (DistributionVector, see Graph.py)
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of RVs (int)
        :param exponential_family: type of exponential family (derived from ExponentialFamilyArray)
        :param ef_args: arguments of exponential_family
        :param use_em: use on-board EM algorithm? (boolean)
        """
        super(HybridLeafLayer, self).__init__(use_em=use_em)

        self.nodes = leaves
        self.num_var = sum([len(e[2]) for e in exponential_family_args])
        self.num_dims = num_dims

        num_dist = list(set([n.num_dist for n in self.nodes]))
        if len(num_dist) != 1:
            raise AssertionError("All leaves must have the same number of distributions.")
        num_dist = num_dist[0]

        replica_indices = set([n.einet_address.replica_idx for n in self.nodes])
        if sorted(list(replica_indices)) != list(range(len(replica_indices))):
            raise AssertionError("Replica indices should be consecutive, starting with 0.")
        num_replica = len(replica_indices)

        # this computes an array of (batch, num_var, num_dist, num_repetition) exponential family densities
        # see ExponentialFamilyArray
        ef_arrays = []
        ef_scopes = []
        for ef, ef_args, ef_scope in exponential_family_args:
            num_ef_var = len(ef_scope)
            ef_array = ef(num_ef_var, num_dims, (num_dist, num_replica), use_em=use_em, **ef_args)
            ef_arrays.append(ef_array)
            ef_scopes.append(ef_scope)
        self.ef_arrays = torch.nn.ModuleList(ef_arrays)
        self.ef_scopes = ef_scopes

        # self.scope_tensor indicates which densities in self.ef_array belongs to which leaf.
        # TODO: it might be smart to have a sparse implementation -- I have experimented a bit with this, but it is not
        # always faster.
        self.register_buffer('scope_tensor', torch.zeros((self.num_var, num_replica, len(self.nodes))))
        for c, node in enumerate(self.nodes):
            self.scope_tensor[node.scope, node.einet_address.replica_idx, c] = 1.0
            node.einet_address.layer = self
            node.einet_address.idx = c

    # --------------------------------------------------------------------------------
    # Implementation of Layer interface

    def default_initializer(self):
        return self.ef_array.default_initializer()

    def initialize(self, initializer=None):
        for ef_array in self.ef_arrays:
            ef_array.initialize(initializer)

    def forward(self, x=None):
        prob = 0
        for ef_array, ef_scope in zip(self.ef_arrays, self.ef_scopes):
            if x is not None:
                x_s = x[:, ef_scope]  
            else:
                x_s = None
            scope_s = self.scope_tensor[ef_scope]
            prob += torch.einsum('bxir,xro->bio', ef_array(x_s), scope_s)

        self.prob = prob

    def backtrack(self, dist_idx, node_idx, mode='sample', **kwargs):
        """
        Backtrackng mechanism for EiNets.

        :param dist_idx: list of N indices into the distribution vectors, which shall be sampled.
        :param node_idx: list of N indices into the leaves, which shall be sampled.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: keyword arguments
        :return: samples (Tensor). Of shape (N, self.num_var, self.num_dims).
        """
        if len(dist_idx) != len(node_idx):
            raise AssertionError("Invalid input.")
        
        device = next(iter(self.parameters())).device
        with torch.no_grad():
            N = len(dist_idx)
            array_shape = self.ef_arrays[0].array_shape
            if mode == 'sample':
                ef_values = torch.zeros((N, self.num_var, self.num_dims, *(array_shape)), device=device) # TODO: add device and dtype
            elif mode == 'argmax':
                ef_values = torch.zeros((self.num_var, self.num_dims, *(array_shape)), device=device) # TODO: add device and dtype
            else:
                raise AssertionError('Unknown backtracking mode {}'.format(mode))

            for ef_array, ef_scope in zip(self.ef_arrays, self.ef_scopes):
                if mode == 'sample':
                    ef_values[:, ef_scope] = ef_array.sample(N, **kwargs)
                elif mode == 'argmax':
                    ef_values[ef_scope] = ef_array.argmax(**kwargs)

                
               # ef_values[]  = ef_value

            values = torch.zeros((N, self.num_var, self.num_dims), device=ef_values.device, dtype=ef_values.dtype)

            for n in range(N):
                cur_value = torch.zeros(self.num_var, self.num_dims, device=ef_values.device, dtype=ef_values.dtype)
                if len(dist_idx[n]) != len(node_idx[n]):
                    raise AssertionError("Invalid input.")
                for c, k in enumerate(node_idx[n]):
                    scope = list(self.nodes[k].scope)
                    rep = self.nodes[k].einet_address.replica_idx
                    if mode == 'sample':
                        cur_value[scope, :] = ef_values[n, scope, :, dist_idx[n][c], rep]
                    elif mode == 'argmax':
                        cur_value[scope, :] = ef_values[scope, :, dist_idx[n][c], rep]
                    else:
                        raise AssertionError('Unknown backtracking mode {}'.format(mode))
                values[n, :, :] = cur_value

            return values

    # def project_params(self, params):
    #     self.ef_array.project_params(params)

    # def reparam(self, params):
    #     return self.ef_array.reparam(params)

    # --------------------------------------------------------------------------------

    def set_marginalization_mask(self, mask):
        """Set the binary mask of marginalized variables."""
        for ef_array, ef_scope in zip(self.ef_arrays, self.ef_scopes):
            ef_array.set_marginalization_mask(mask[:, ef_scope])

    def get_marginalization_mask(self):
        """Get the binary mask of marginalized variables."""
        ef_array_masks = []
        for ef_array in self.ef_arrays:
            ef_array_mask = ef_array.get_marginalization_mask()
            if ef_array_mask is None:  # TODO: correct None handling
                return None
            ef_array_masks.append(ef_array_mask)
        
        max_cols = self.num_var  
        num_rows = ef_array_masks[0].shape[0]  # assume this is the same for all ef_array_masks, TODO: do checkup in future

        mask = torch.zeros((num_rows, max_cols), dtype=ef_array_masks[0].dtype, device=ef_array_masks[0].device)

        for ef_array_mask, scope in zip(ef_array_masks, self.ef_scopes):
            mask[:, scope] = ef_array_mask
    
        return mask

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        # self.ef_array.em_set_hyperparams(online_em_frequency, online_em_stepsize, purge)
        pass

    def em_purge(self):
        # self.ef_array.em_purge()
        pass

    def em_process_batch(self):
        # self.ef_array.em_process_batch()
        pass

    def em_update(self):
        # self.ef_array.em_update()
        pass

    def project_params(self, params):
        # self.ef_array.project_params(params)
        pass


if __name__ == '__main__':
    from einsum import Graph, EinsumNetwork

    num_var = 10
    num_dims = 1
    depth = 2
    num_repetitions = 10
    num_input_distributions = 10

    N_CAT = 256 

    graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)
    for node in Graph.get_leaves(graph):
            node.num_dist = num_input_distributions

    graph_layers = Graph.topological_layers(graph)

    exponential_family_args = [
                            (EinsumNetwork.CategoricalArray, {'K': N_CAT}, range(0, 5)),
                            (EinsumNetwork.NormalArray, {}, range(5, 10))
                               ]

    input_layer = HybridLeafLayer(graph_layers[0],
                                    num_dims,
                                    exponential_family_args)
    input_layer.initialize(initializer='default')
    
    c_distr = torch.distributions.Categorical(logits=torch.randn(5, N_CAT))
    n_distr = torch.distributions.Normal(torch.randn(5), torch.rand(5)+1e-3)
    
    num_samples = 64
    c_data = c_distr.sample((num_samples, ))
    n_data = n_distr.sample((num_samples, ))

    x = torch.cat((c_data, n_data), dim=-1)
    # x = cat_data
    print(input_layer)
    input_layer(x=x)
    print(input_layer.prob.shape)
    print(len(list(input_layer.parameters())))

