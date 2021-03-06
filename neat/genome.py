"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function

from itertools import count
from random import choice, random, shuffle, randint

import sys

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle
from neat.six_util import iteritems, iterkeys


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('conn_add_num', int),
                        ConfigParameter('conn_delete_num', int),
                        ConfigParameter('node_add_num', int),
                        ConfigParameter('node_delete_num', int),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected'),
                        ConfigParameter('num_layer', int),
                        ConfigParameter('num_cnn_layer', int),
                        ConfigParameter('kernal_size', int),
                        ConfigParameter('input_size', int),
                        ConfigParameter('init_channel_num', int),
                        ConfigParameter('num_downsampling', int),
                        ConfigParameter('num_dense_layer', int),
                        ConfigParameter('full_connect_input', bool)]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1','yes','true','on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0','no','false','off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write('initial_connection      = {0} {1}\n'.format(self.initial_connection,
                                                                 self.connection_fraction))
        else:
            f.write('initial_connection      = {0}\n'.format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if not 'initial_connection' in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key, config):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}
        self.layer = []
        self.downsampling_mask = []
        self.nodes_every_layers = []
        #self.bn = []

        # Compute node number in every layer.
        assert config.num_cnn_layer >= config.num_downsampling

        if config.num_cnn_layer == config.num_downsampling:
            self.layer_num_in_middle_group = 1
            self.layer_num_in_last_group = 1
            self.layer_num_in_first_group = 1
            self.group_num = config.num_downsampling
            self.downsampling_mask = [0] * config.num_layer
            for i in range(config.num_cnn_layer):
                self.downsampling_mask[i] = 1
        else:
            # The layers are arranged in groups.
            # The middle groups have num_cnn_layer // num_downsampling layers
            # The first and last group have (num_cnn_layer // num_downsampling) + (num_cnn_layer % num_downsampling) layers.
            # The first group have one more layer than last group if the total layer is not even.
            self.layer_num_in_middle_group = config.num_cnn_layer // config.num_downsampling
            self.layer_num_in_last_group = (self.layer_num_in_middle_group + (config.num_cnn_layer % config.num_downsampling)) // 2
            self.layer_num_in_first_group = (self.layer_num_in_middle_group + (config.num_cnn_layer % config.num_downsampling)) - self.layer_num_in_last_group
            self.group_num = config.num_downsampling + 1

            # Generate downsampling mask.
            self.downsampling_mask = [0] * config.num_layer
            for i in range(self.layer_num_in_first_group, (config.num_cnn_layer - (self.layer_num_in_last_group - 1))):
                if (i - self.layer_num_in_first_group) % self.layer_num_in_middle_group == 0:
                    self.downsampling_mask[i] = 1
                else:
                    self.downsampling_mask[i] = 0

            # Convert downsampling mask to downsampling num list
            times = 0
            self.downsampling_time = [0] * config.num_layer
            for i in range(config.num_layer):
                if self.downsampling_mask[i] == 1:
                    times += 1
                self.downsampling_time[i] = times

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create layer: cnn layer & all to all layer
        for i in range(config.num_cnn_layer):
            self.layer.append(['cnn', set()])
        for i in range(config.num_cnn_layer, config.num_layer):
            self.layer.append(['fc', set()])

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key, config.num_layer - 1)
        # Add output layer nodes
        self.layer[-1][1] = set(config.output_keys)


        # Generate node number in each layer.
        # self.nodes_every_layers.append(config.num_inputs)
        for i in range(self.layer_num_in_first_group):
            self.nodes_every_layers.append(config.init_channel_num)

        for i in range(1, self.group_num - 1):
            for j in range(self.layer_num_in_middle_group):
                self.nodes_every_layers.append(config.init_channel_num * (2 ** i))

        # Note "layer_num_in_last_group - 1"
        #for i in range(self.layer_num_in_last_group - 1):
        for i in range(self.layer_num_in_last_group):
            self.nodes_every_layers.append(config.init_channel_num * (2 ** config.num_downsampling))

        # Generate node number in fc layers.
        # Note the last layer have been assigned nodes.
        for i in range(config.num_cnn_layer, config.num_layer - 1):
            self.nodes_every_layers.append(config.init_channel_num * (2 ** config.num_downsampling))

        num_hidden = sum(self.nodes_every_layers)

        # Add hidden nodes if requested.
        hidden_node_key = set()
        if num_hidden > 0:
            for i in range(num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key, -198043)
                self.nodes[node_key] = node
                hidden_node_key.add(node_key)

        # Assign nodes to layers.
        # Note the last layer have been assigned nodes.
        for i in range(config.num_layer - 1):
            for j in range(self.nodes_every_layers[i]):
                node_key = hidden_node_key.pop()
                self.layer[i][1].add(node_key)
                self.nodes[node_key].layer = i

        # Add the last fc layer node number.
        self.nodes_every_layers.append(config.num_outputs)

        """
        # Generate node number in each layer.
        self.nodes_every_layers = list()
        self.nodes_every_layers_cfg = list()
        self.nodes_every_layers.append(config.num_inputs)
        for i in range(layer_num_in_first_group):
            self.nodes_every_layers.append(config.init_channel_num)
            self.nodes_every_layers_cfg.append(config.init_channel_num)

        for i in range(1, group_num - 1):
            for j in range(layer_num_in_middle_group):
                self.nodes_every_layers.append(config.init_channel_num * (2 ** i))
                self.nodes_every_layers_cfg.append(config.init_channel_num * (2 ** i))

        # Note "layer_num_in_last_group - 1"
        for i in range(layer_num_in_last_group - 1):
            self.nodes_every_layers.append(config.init_channel_num * (2 ** config.num_downsampling))
            self.nodes_every_layers_cfg.append(config.init_channel_num * (2 ** config.num_downsampling))
        self.nodes_every_layers_cfg.append(config.init_channel_num * (2 ** config.num_downsampling))

        # Generate node number in fc layers.
        # Note the last layer have been assigned nodes.
        for i in range(config.num_cnn_layer, config.num_layer - 1):
            self.nodes_every_layers.append(config.init_channel_num * (2 ** config.num_downsampling))
            self.nodes_every_layers_cfg.append(config.init_channel_num * (2 ** config.num_downsampling))

        num_hidden = sum(self.nodes_every_layers)

        # Add hidden nodes if requested.
        hidden_node_key = set()
        if num_hidden > 0:
            for i in range(num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key, -198043)
                self.nodes[node_key] = node
                hidden_node_key.add(node_key)

        # Assign nodes to layers.
        # Note the last layer have been assigned nodes.
        for i in range(config.num_layer - 1):
            for j in range(self.nodes_every_layers[i]):
                node_key = hidden_node_key.pop()
                self.layer[i][1].add(node_key)
                self.nodes[node_key].layer = i

        # Add the last fc layer node number.
        self.nodes_every_layers.append(config.num_outputs)
        self.nodes_every_layers_cfg.append(config.num_outputs)
        """
        """
        # Assign nodes to layers, make sure every layer has at least one node
        if (len(hidden_node_key) >= config.num_layer - 1):
            for i in range(config.num_layer - 1):
                node_key = hidden_node_key.pop()
                self.layer[i][1].add(node_key)
                self.nodes[node_key].layer = i
        else:
            raise RuntimeError("Too less nodes.")

        # Assign nodes to the first fc layer.
        # (The first fc layer already has one node, so add num_first_fc_layer_node - 1 nodes.
        if (config.num_first_fc_layer_node > 0) and (len(hidden_node_key) >= config.num_first_fc_layer_node - 1):
            for i in range(config.num_first_fc_layer_node - 1):
                node_key = hidden_node_key.pop()
                self.layer[config.num_cnn_layer][1].add(node_key)
                self.nodes[node_key].layer = config.num_cnn_layer
        else:
            raise RuntimeError("Too less nodes.")
        

        # Assign the left nodes to layers randomly
        while hidden_node_key:
            layer_nums = randint(0, config.num_layer - 2)
            if (layer_nums == config.num_cnn_layer): # Do not add node to the first fc layer.
                continue
            node_key = hidden_node_key.pop()
            self.layer[layer_nums][1].add(node_key)
            self.nodes[node_key].layer = layer_nums
        """

        # Add connections based on initial connectivity type.

        # fs_neat is not used in cnn
        """
        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep='\n', file=sys.stderr);
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr);
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = partial_nodirect {0};".format(
                            config.connection_fraction),
                        "\tif not, set initial_connection = partial_direct {0}".format(
                            config.connection_fraction),
                        sep='\n', file=sys.stderr);
                self.connect_partial_nodirect(config)
        """
        if config.initial_connection == 'full':
            self.connect_full(config)
        elif config.initial_connection == 'partial':
            self.connect_partial(config)
        else:
            print("Only full and partial connection allowed in CNN!")

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """

        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

        # Create layer: cnn layer & all to all layer
        for i in range(config.num_cnn_layer):
            self.layer.append(['cnn', set()])
        for i in range(config.num_cnn_layer, config.num_layer):
            self.layer.append(['fc', set()])

        # Add layer according to nodes in new genome
        for node in iteritems(self.nodes):
            self.layer[node[1].layer][1].add(node[1].key)

        # Compute node num in every layer
        self.nodes_every_layers = [0] * config.num_layer
        for i in range(config.num_layer):
            self.nodes_every_layers[i] = len(self.layer[i][1])

    def mutate(self, config): 
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1,(config.node_add_prob + config.node_delete_prob +
                         config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob/div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob)/div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob)/div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob)/div):
                self.mutate_delete_connection(config)
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection(config)

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    # Added by Andrew @20181107
    # Add a node to the network, if the added node in the first layer then judge if should add a full connection
    # to all the input. Then add one connection to the former layer and one to the after layer.
    # Note: The node cannot be added to the last layer!
    # TODO: Add connection according to the conncetion type parameter in the config file
    def mutate_add_node(self, config):
        num = 0
        for i in range(config.node_add_num):
            num += 1
            # Choose the layer to add node (not the last layer)
            layer_num = randint(0, config.num_layer - 2)

            # Revise the nodes_every_layers list
            self.nodes_every_layers[layer_num] += 1

            new_node_id = config.get_new_node_key(self.nodes)
            ng = self.create_node(config, new_node_id, layer_num)

            self.layer[layer_num][1].add(new_node_id)
            self.nodes[new_node_id] = ng

            # if the added node in first layer
            connections = []
            '''
            if layer_num == 0: #TODO: Add connections to the following layers
                # Add full connection between input and the first layer
                if config.full_connect_input:
                    for input_id in config.input_keys:
                        connections.append((input_id, new_node_id))

                # Add one connction between input and the first layer
                else:
                    input_id = choice(config.input_keys)
                    connections.append((input_id, new_node_id))
            '''

            # Add connection to input, if the added node layer not more than num_dense_layer
            if layer_num < config.num_dense_layer:
                for input_id in config.input_keys:
                    connections.append((input_id, new_node_id))

            #node_id = choice(list(self.layer[layer_num - 1][1]))
            #connection = self.create_connection(config, node_id, new_node_id)
            #self.connections[connection.key] = connection

            # Add to support dense connection. by Andrew 2019.3.18
            # Add connections to the layer before
            if layer_num <= config.num_cnn_layer: # if the layer of the added node is in cnn layer or in the first fc layer
                for i in (range(config.num_dense_layer)):
                    if layer_num-i-1 >= 0:
                        for j in list(self.layer[layer_num-i-1][1]):
                            connections.append((j, new_node_id))
            else:
                for j in list(self.layer[layer_num - 1][1]):
                    connections.append((j, new_node_id))

            # Add connections to the layer after
            if layer_num < config.num_cnn_layer: # if the layer of the added node is in cnn layer
                for i in (range(config.num_dense_layer)):
                    if layer_num+i+1 <= config.num_cnn_layer: # connect to following cnn layer or the first fc layer
                        for j in list(self.layer[layer_num+i+1][1]):
                            connections.append((new_node_id, j))
            else: # the added node is in fc layers
                for j in list(self.layer[layer_num + 1][1]):
                    connections.append((new_node_id, j))


            if config.initial_connection == 'full':
                for node1, node2 in connections:
                    connection = self.create_connection(config, node1, node2)
                    self.connections[connection.key] = connection
            elif config.initial_connection == 'partial':
                assert 0 <= config.connection_fraction <= 1
                shuffle(connections)
                num_to_add = int(round(len(connections) * config.connection_fraction))
                for input_id, output_id in connections[:num_to_add]:
                    connection = self.create_connection(config, input_id, output_id)
                    self.connections[connection.key] = connection
            else:
                print("Only full and partial connection allowed in CNN!")
            '''
            out_node_layer_distance = randint(left, right)
            out_node = choice(list(self.layer[layer_num + out_node_layer_distance][1]))
            connection = self.create_connection(config, new_node_id, out_node)
            self.connections[connection.key] = connection
            '''

        print("{0} nodes added!".format(num))

    """
    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id, -198043)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)
    """
    # Not used
    def add_connection(self, config, input_key, output_key, weight, enabled):

        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)

        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    """
    def mutate_add_connection(self, config):
        """"""
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """"""
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    """
    # Added by Andrew @20181107
    # Add a connection to the network.
    # TODO: Add connection with the probability according to its connections already has.
    # TODO: Add connection according to the conncetion type parameter in the config file
    def mutate_add_connection(self, config):
        num = 0
        for i in range(config.conn_add_num):

            # Choose the outnode layer
            layer_num = randint(0, config.num_layer - 1)

            # If choose out_node form the first layer, the input_node should choose from input of the network.
            if layer_num == 0:
                out_node = choice(list(self.layer[layer_num][1]))
                in_node = choice(config.input_keys)
            else:
                out_node = choice(list(self.layer[layer_num][1]))
                #in_node = choice(list(self.layer[layer_num - 1][1]))
                # Changed to support dense connection. by Andrew 2019.3.18
                left = 1
                right = layer_num if layer_num < config.num_dense_layer else config.num_dense_layer
                in_node_layer_distance = randint(left, right)
                in_node = choice(list(self.layer[layer_num - in_node_layer_distance][1]))

            # Don't duplicate connections.
            key = (in_node, out_node)
            if key in self.connections:
                # TODO: Should this be using mutation to/from rates? Hairy to configure...
                if config.check_structural_mutation_surer():
                    self.connections[key].enabled = True
                continue

            # Don't allow connections between two output nodes
            if in_node in config.output_keys and out_node in config.output_keys:
                continue

            # No need to check for connections between input nodes:
            # they cannot be the output end of a connection (see above).

            # For feed-forward networks, avoid creating cycles.
            if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), key):
                continue

            cg = self.create_connection(config, in_node, out_node)
            self.connections[cg.key] = cg
            num += 1
        print("{0} connections added!".format(num))

    def mutate_delete_node(self, config):
        num = 0
        for i in range(config.node_delete_num):
            # Do nothing if there are no non-output nodes.
            available_nodes = [k for k in iterkeys(self.nodes) if k not in config.output_keys]
            if not available_nodes:
                continue

            del_key = choice(available_nodes)

            # Cannot delete node in the first fc layer
            #if self.nodes[del_key].layer == config.num_cnn_layer:
            #    return -1

            # Cannot delete node in the last (output) layer
            if self.nodes[del_key].layer == config.num_layer:
                continue

            # If there is only one node
            if len(self.layer[self.nodes[del_key].layer][1]) <= 1:
                continue

            connections_to_delete = set()
            for k, v in iteritems(self.connections):
                if del_key in v.key:
                    connections_to_delete.add(v.key)

            for key in connections_to_delete:
                del self.connections[key]

            self.layer[self.nodes[del_key].layer][1].remove(del_key)

            # Revise the nodes_every_layers list
            self.nodes_every_layers[self.nodes[del_key].layer] -= 1

            del self.nodes[del_key]

            num += 1
        print("{0} nodes deleted!".format(num))

    def mutate_delete_connection(self, config):
        num = 0
        for i in range(config.conn_delete_num):
            if self.connections:
                key = choice(list(self.connections.keys()))
                #TODO: add judgement to avoid del the last connection between two layers
                del self.connections[key]
                num += 1
        print("{0} connections deleted!".format(num))

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)

        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)

        s += "\nLayers:"
        for i in range(len(self.layer)):
            s += "\n\t" + self.layer[i][0] + ": "
            l = list(self.layer[i][1])
            l.sort()
            for node in l:
                s += " {0}".format(node)
        return s

    @staticmethod
    def create_node(config, node_id, layer):
        node = config.node_gene_type(node_id, layer)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in iterkeys(self.nodes) if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.output_keys]
        output = [i for i in iterkeys(self.nodes) if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections

    def compute_full_connections_with_layer(self, config):
        """
        Compute connections for a fully-connected cnn genome--each node in one
        layer connected to all nodes in the next layer
        """
        connections = []

        # Revised to add dense connections from input to the following num_dense_layers
        for i in range(config.num_dense_layer):
            for node in self.layer[i][1]:
                for input_id in config.input_keys:
                    connections.append((input_id, node))

        #Add dense connection 2019.3.18
        for i in range(config.num_cnn_layer - 1):
            for j in (range(config.num_dense_layer)):
                if i+j+1 < config.num_cnn_layer:
                    for node1 in self.layer[i][1]:
                        for node2 in self.layer[i+j+1][1]:
                            connections.append((node1, node2))

        for i in range(config.num_cnn_layer - 1, config.num_layer - 1):
            for node1 in self.layer[i][1]:
                for node2 in self.layer[i + 1][1]:
                    connections.append((node1, node2))

        '''
        # Original none dense connention
        for i in range(len(self.layer) - 1):
             for node1 in self.layer[i][1]:
                    for node2 in self.layer[i+1][1]:
                        connections.append((node1, node2))
        '''

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections

    def connect_full(self, config):
        """
        Create a fully-connected cnn genome
        """
        for node1, node2 in self.compute_full_connections_with_layer(config):
            connection = self.create_connection(config, node1, node2)
            self.connections[connection.key] = connection

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections_with_layer(config)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
