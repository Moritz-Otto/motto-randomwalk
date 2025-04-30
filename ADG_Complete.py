#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:26:25 2024

@author: motto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:49:17 2024

@author: motto
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

################ Coalescence Tree Classes and Functions ######################
class Tree:
    def __init__(self, ind_label, copy_label=0, copy_size=0, branch_length=0):
        self.ind_label = ind_label
        self.copy_label = copy_label
        self.copy_size = copy_size
        self.label = f'{self.ind_label}.{self.copy_label}'
        
        self.branch_length = branch_length
        self.children = []
        self.parent = None
        
        self.mutations = 0
        self.x_pos = 0
        self.y_pos = 0
        
        self.node_type = 'ancestor'
        self.lineage_copies = 1
        
    def add_child(self,other):
        self.children += [other]
        other.parent = self
        
    def insert_midparent(self,other,branch_length):
        #Given a node and its original parent, split the branch at position branch_length
        #and insert a midpoint-parent
      
        other.children += [self]
        other.parent = self.parent
        other.branch_length = branch_length
        self.parent.children.remove(self)
        self.parent.children += [other]
        
        
        self.branch_length -= branch_length
        self.parent = other
        
    def duplicate(self,dupl_count=1):
        new_node = Tree(f'{self.ind_label}',copy_label=dupl_count)
        #print(self.label)
        #print(new_node.label)
        for child in self.children:
            new_node.add_child(child.duplicate(dupl_count))
        new_node.branch_length = self.branch_length
        return(new_node)

    def get_nodes(self,out=False):
        
        nodes = [self]
        if out: 
            print(self.label)
        for child in self.children:
            nodes += child.get_nodes(out)
        return(nodes)
    
    def get_leaves(self,out=False):
        leaves = []
        if (len(self.children) == 0):
            leaves += [self]
            self.node_type = 'leaf'
            if out: 
                print(self.label)
        for child in self.children:
            leaves += child.get_leaves(out)
        return(leaves)
       
        
    def draw(self,level=0):
        print(" " * level * 2, self.label, "--", self.mutations, "--", np.round(self.branch_length,decimals=2))
        for child in self.children:
            child.draw(level+1)




def make_std_coal_tree(n, N):
  #generate a standard coalescence tree of n samples with population size N

  leaves = [Tree(f'{i}') for i in range(1,n+1)]
  while len(leaves)>1:
      t = np.random.exponential(2 * N / (len(leaves) * (len(leaves) - 1)))
      i, j = random.sample(range(len(leaves)), 2)
      
      if i > j:
          i, j = j, i
      
      for leaf in leaves:
          leaf.branch_length += t
            
      
      new_node = Tree(f'A[{leaves[i].label}-{leaves[j].label}]')
      new_node.add_child(leaves[i])
      new_node.add_child(leaves[j])
      
      del leaves[j]
      del leaves[i]
      
      leaves.append(new_node)
      

  return(leaves[0])
      

def make_duplications(tree,d_rate,out=False,CCM=True):
    #given standard coalescence tree, place duplications like mutations on the tree and duplicate the subtree
    #uses insert_midparent and duplicate features of Tree class
    #If CCM (compound copy model) is activated, the duplicated subtree can also duplicate, hence the number
    #of duplication events on a single branch is not poisson distributed, but negative binomial distributed
    #just as in the standard yule birth process
    
    dupl_count = 0
    all_nodes = tree.get_nodes()
    for node in all_nodes:
        duplication_events = np.random.poisson(d_rate * node.branch_length)
        while duplication_events > 0:
            duplication_events -= 1
            dupl_count += 1
            
            new_node = Tree(f'D{dupl_count}')
            new_node.node_type = 'duplication'
            node.insert_midparent(new_node,np.random.uniform()*node.branch_length)
            duplicated_node = node.duplicate(dupl_count)
            if CCM:
                all_nodes += duplicated_node.get_nodes()
            new_node.add_child(duplicated_node)
    if out:
        print(f'Number of Duplication evens: {dupl_count}')
    

        
def make_structured_coal_tree(pop, d, N,out=False):
    
    leaves = []  
    for i, counts in enumerate(pop,start=1):
        for k in range(counts):
            new_node = Tree(f'{i}',k,counts)
            new_node.node_type = "leaf"
            leaves.append(new_node)
   
    #Subroutines: track_path, get_coalescence_pairs, make_coalescence, get_cupl_pairs, make_duplication
    #Each tree-node / lineage has an 'individual' label, i.e. on which individual it is located
    #If a coalescence event happens, two individuals with the same copy size are chosen and each copy is paired randomly
    #with another
    #An unduplication event reduces the copy number by one
    
    #The path represents the current copy number configuration in the sample along the tree
    #For example, (0,3,4) means three individuals with 2 copies and four with 3 copies
    def track_path(leaves,out): 
        tree_df = pd.DataFrame(columns=["ind_label", "copy_label", "copy_size"])
        for i in range(len(leaves)):
            tree_df = tree_df._append({"ind_label": leaves[i].ind_label, "copy_label": leaves[i].copy_label , "copy_size": leaves[i].copy_size}, ignore_index=True)
                
                
        filtered_df = tree_df[tree_df['copy_label'] == 0]
        
        # Erstellen der Frequency-Tabelle für die Spalte 'copy_size'
        frequency_table = filtered_df['copy_size'].value_counts().reset_index()
        frequency_table.columns = ['copy_size', 'frequency']
        
        #print(frequency_table)
        
        # Maximalen Wert in 'copy_size' ermitteln
        max_value = tree_df['copy_size'].max()
        
        # Initialisieren eines Vektors mit Nullen
        path = [0] * max_value
        
        # Den Vektor mit den Werten aus der Frequency-Tabelle füllen
        for i in range(len(frequency_table)):
            size = frequency_table['copy_size'][i]
            count = frequency_table['frequency'][i]
            path[int(size) - 1] = int(count)

        if(out):
            print(tree_df)
            print("\n\n")
        return(path)
    
    def get_coalescence_pairs(leaves):
        #Step 1:choose two individuals of same copy number size to coalesce
        coalescence_pairs = []
        ind_repr_indices = []
        #the indices from the list of leaves, which give the unique representative system of the indiv labels
        #example: leaves.ind_labels = [3,3,3,1,2,1,2,2,5]
        #return: [0,3,4,8]
        for i in range(len(leaves)):
            if (leaves[i].copy_label == 0):
                ind_repr_indices.append([i,leaves[i].copy_size])
        
        for i in range(len(ind_repr_indices)):
           for j in range(i+1, len(ind_repr_indices)):
               if (ind_repr_indices[i][1] == ind_repr_indices[j][1]):
                   coalescence_pairs.append([
                       leaves[ind_repr_indices[i][0]].ind_label, leaves[ind_repr_indices[j][0]].ind_label
                       ])
        
        
        if (len(coalescence_pairs) > 0):
            label1, label2 = random.sample(coalescence_pairs,1)[0]
            
            #Step 2: For individuals of same copy number size, get a random pairing of all copies
            #return the indices of the leaves-list of all coalescence pairs that happen
            
            index_list1 = []
            index_list2 = []
            for i in range(len(leaves)):
                if (leaves[i].ind_label == label1):
                    index_list1.append(i)
                if (leaves[i].ind_label == label2):
                    index_list2.append(i)
                        

            
            index_list1 = random.sample(index_list1,len(index_list1))
            index_list2 = random.sample(index_list2,len(index_list2))


            
            combined_list = []

            combined_list = [[index_list1[i], index_list2[i]] for i in range(len(index_list1))]
            return(combined_list)
            
        else:
            #print('No coalescence event possible')
            return([])
        
    def make_coalescence(leaves,c_pairs,t,out=False):
        if(out):
            print("Coalescence-Pairs:")
            print(leaves[c_pairs[0][0]].ind_label,leaves[c_pairs[0][1]].ind_label)
            print(c_pairs)
            print("\n\n\n\n\n")

        for leaf in leaves:
            leaf.branch_length += t
        for k in range(len(c_pairs)):
            i,j = c_pairs[k]
                        
            new_node = Tree(f'A[{leaves[i].ind_label}-{leaves[j].ind_label}]',k,len(c_pairs))
            new_node.add_child(leaves[i])
            new_node.add_child(leaves[j])
            new_node.node_type = 'ancestor'
            
            leaves.append(new_node)
            
        for k in sorted([index for pair in c_pairs for index in pair],reverse=True):
            del leaves[k]

        
        return(leaves)
            

    
    def get_dupl_pairs(leaves):
        #Step 1:choose two individuals of same copy number size to coalesce
        dupl_pairs = []
        #the indices from the list of leaves, which give the unique representative system of the indiv labels
        #example: leaves.ind_labels = [3,3,3,1,2,1,2,2,5]
        #return: [0,3,4,8]
        for i in range(len(leaves)):
            for j in range(i+1,len(leaves)):
                if (leaves[i].ind_label == leaves[j].ind_label):
                    dupl_pairs.append([i,j])
                            
        return(dupl_pairs)
            
    
    def make_duplication(leaves,d_pairs,t,out=False):
        i,j = random.sample(d_pairs,1)[0]
        
        if i > j:
            i,j = j,i
        
        if(out):
            print("Duplication-Pairs:")
            print(leaves[i].label,leaves[j].label)
            print("\n\n\n\n\n")
            
        for leaf in leaves:
            leaf.branch_length += t
        
        new_copy_label = min(leaves[i].copy_label, leaves[j].copy_label)
        new_ind_label = leaves[i].ind_label
        new_copy_size = leaves[i].copy_size
        
        
        
        new_node = Tree(new_ind_label,new_copy_label,new_copy_size)
        new_node.add_child(leaves[i])
        new_node.add_child(leaves[j])
        new_node.node_type = 'duplication'
        
        
        del leaves[j]
        del leaves[i]
        
        leaves.append(new_node)
        
        for leaf in leaves:
            if(leaf.ind_label == new_ind_label):
                leaf.copy_size -= 1
     
        return(leaves)
        
    n = len(pop)
    total_time = 0.0
    node_in_graph = track_path(leaves,out)

    path = [[node_in_graph,total_time]]
    while len(leaves)>1:
        
        if (n > 1):
            t_c = np.random.exponential(2 * N / (n * (n - 1)))
            c_pairs = get_coalescence_pairs(leaves)
        else:
            t_c = 0
            c_pairs = []


        #t_d = np.random.exponential(1/(n*d))
        t_d = np.random.exponential(1/(len(leaves)*d))
        d_pairs = get_dupl_pairs(leaves)
        
        if ((t_c <= t_d) and (len(c_pairs)>0)):
            leaves = make_coalescence(leaves,c_pairs,t_c,out)
            total_time += t_c
            n -= 1
        elif ((t_c > t_d) and (len(d_pairs)>0)):
            leaves = make_duplication(leaves,d_pairs,t_d,out)
            total_time += t_d
        elif (len(d_pairs)==0 and len(c_pairs)==0):
            break
        elif (len(c_pairs)==0 and t_c < t_d):
            if out:
                print("\n!!!No coalescence possible!!!\n")
            leaves = make_duplication(leaves,d_pairs,t_d,out)
            total_time += t_d
        elif (len(d_pairs)==0 and t_c > t_d):
            if out:
                print("\n!!!No duplication possible!!!\n")
            leaves = make_coalescence(leaves,c_pairs,t_c,out)
            total_time += t_c
            n -= 1
            
        node_in_graph = track_path(leaves,out)
        path.append([tuple(node_in_graph),round(total_time,4)])

    
    return(leaves[0], path)




def make_mutations(tree,mu_rate,out=False):
    #Given a tree, place mutations with rate mu according to a poisson distribution
    all_nodes = tree.get_nodes()
    mut_count = 0
    for node in all_nodes:
        mutation_events = np.random.poisson(mu_rate * node.branch_length)
        node.mutations += mutation_events
        mut_count += mutation_events
    if out:
        print(f'Number of Mutation events: {mut_count}')

def genotype_matrix(tree):
    all_nodes = tree.get_nodes()
    all_leaves = tree.get_leaves()
    SNP_mat = np.zeros( (len(all_leaves),1) )
    
    for node in all_nodes:
        new_SNPs = 0
        new_SNPs = node.mutations
        while new_SNPs > 0:
            new_SNPs -= 1
            new_row = np.zeros((len(all_leaves),1))
            node_leaves = node.get_leaves()
            indices = [all_leaves.index(x) for x in node_leaves]
            new_row[indices,0] = 1
            SNP_mat = np.hstack((SNP_mat,new_row))
    if SNP_mat.shape[1]>1:
        leaf_names = [x.label for x in tree.get_leaves()]
        k = len(leaf_names)
        SNP_mat = SNP_mat[:,1:]

        # Arrays für Prefix und Zahlen initialisieren
        prefix = np.empty(k, dtype=object)
        identifier = np.empty(k, dtype=int)

        # Namen aufteilen
        for i in range(k):
            parts = leaf_names[i].split('.')
            identifier[i] = parts[0]
            prefix[i] = "D"+parts[1]
            
        # Kombinierte Matrix erstellen
        combined_matrix = np.column_stack((identifier, prefix, SNP_mat))
        sorted_combined_matrix = combined_matrix[np.lexsort((combined_matrix[:, 1], combined_matrix[:, 0].astype(int)))]
        return(sorted_combined_matrix)
    else:
        return(SNP_mat)


def draw_png_tree(tree,x_scale=1):
    tree.get_leaves()
    nodes = [tree]
    new_nodes = nodes
    level = 0
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if (tree.node_type == 'duplication'):
        ax.scatter(tree.x_pos, tree.y_pos, color='blue', s =150)
    if (tree.node_type == 'ancestor'):
        ax.scatter(tree.x_pos, tree.y_pos, color='red', s =100)

    while len(new_nodes)>0:
        level += 1
        nodes = new_nodes
        new_nodes = []
        
        for leaf in nodes:
            if len(leaf.children)>0:
                child1, child2 = leaf.children
                child1.x_pos = child1.parent.x_pos - 1/(np.exp(x_scale * np.log(level)))
                child2.x_pos = child2.parent.x_pos + 1/(np.exp(x_scale * np.log(level)))
                child1.y_pos = child1.parent.y_pos - child1.branch_length
                child2.y_pos = child2.parent.y_pos - child2.branch_length
                
                ax.plot([child1.x_pos, child1.x_pos], [child1.y_pos, child1.parent.y_pos], color='black')  # Vertikale Linie
                ax.text(child1.x_pos, (child1.parent.y_pos + child1.y_pos)/2, child1.mutations, color='red')
                
                ax.plot([child2.x_pos, child2.x_pos], [child2.y_pos, child2.parent.y_pos], color='black')  # Vertikale Linie
                ax.text(child2.x_pos, (child2.parent.y_pos + child2.y_pos)/2, child2.mutations, color='red')
                
                ax.plot([child1.x_pos, child2.x_pos], [child1.parent.y_pos, child1.parent.y_pos], color='black') # Horizontale Linie
             
                if (child1.node_type == 'duplication'):
                    ax.scatter(child1.x_pos, child1.y_pos, color='blue', s =150)
                if (child1.node_type == 'ancestor'):
                    ax.scatter(child1.x_pos, child1.y_pos, color='red', s =100)
                if (child2.node_type == 'duplication'):
                    ax.scatter(child2.x_pos, child2.y_pos, color='blue', s =150)
                if (child2.node_type == 'ancestor'):
                    ax.scatter(child2.x_pos, child2.y_pos, color='red', s =100)
                if (child1.node_type == 'leaf'):
                    ax.text(child1.x_pos, child1.y_pos, child1.label)
                if (child2.node_type == 'leaf'):
                    ax.text(child2.x_pos, child2.y_pos, child2.label)
                
                
                new_nodes += [child1, child2]

    plt.show()



################ Random Walk on Graph Classes and Functions ######################
class Node:
    def __init__(self,label):
        self.label=label
        self.no_ind = sum(label) if label is not None else 0
        self.no_copies = sum((i + 1) * label[i] for i in range(len(label))) if label is not None else 0
        self.out_edges = []
        self.in_edges = []
        self.x_pos = self.no_ind
        self.y_pos = self.no_copies

class Graph:
    def __init__(self):
        self.nodes = []
    
    def is_in(self,label):
        for node in self.nodes:
            if node.label == label:
                return True
        return False

    def get_node(self, label):
        for node in self.nodes:
            if node.label == label:
                return node
        return None

    def get_all_node_labels(self):
        return([node.label for node in graph.nodes])
    
    def get_all_out_edges(self):
        edges = []
        for from_node in self.nodes:
            for to_node, rate, prob in from_node.out_edges:
                edges.append([from_node.label, to_node.label, rate, prob])
        return(edges)
    
    def get_all_in_edges(self):
        edges = []
        for from_node in self.nodes:
            for to_node, rate, prob in from_node.in_edges:
                edges.append([from_node.label, to_node.label, rate, prob])
        return(edges)
    
    def add_node_with_testing(self, label):
        if not self.is_in(label):
            node = Node(label)
            self.nodes.append(node)
            return node
        return None
    
    def add_node(self, label):
        node = Node(label)
        self.nodes.append(node)
        return node
        
    def add_edge(self, from_node, to_node, rate, prob):
        from_node.out_edges.append([to_node,rate,prob])
        to_node.in_edges.append([from_node,rate,prob])

def make_graph(start_label, N, d):
    
    #subroutines#       
    def generate_nodes(graph, start_node):
        stack = [start_node]
        added_labels = [start_node.label]
        while stack:
            current_node = stack.pop()
            current_label = current_node.label
            graph.add_node(current_label)
            if current_label == [1]:
                continue                

            for i in range(len(current_label)):
                if current_label[i] > 1:
                    
                    next_label = current_label[:]
                    next_label[i] -= 1
                    if next_label not in added_labels:
                        added_labels.append(next_label)
                        stack.append(Node(next_label))
                        
                if i > 0 and current_label[i] > 0:
                    next_label = current_label[:]
                    next_label[i] -= 1
                    next_label[i-1] += 1
                    if next_label not in added_labels:
                        added_labels.append(next_label)
                        stack.append(Node(next_label))
    
    def generate_edges(graph,d,N):
        all_nodes = graph.nodes
        for node in all_nodes:
            p_out = 0.
            for i in range(len(node.label)):
                if node.label[i] > 0:
                    c_label = node.label[:]
                    c_label[i] += 1
                    ###################################
                    #Choose here the coalescence rates#
                    ###################################
                    c_rate = 1/N * node.label[i]
                    #c_rate = 1/N * ((node.label[i]+1) * node.label[i]) / 2
                    if graph.is_in(c_label):
                        graph.add_edge(node, graph.get_node(c_label), c_rate, c_rate)
                    else:
                        p_out += c_rate
                    d_label = node.label[:]
                    d_label[i] -= 1
                    if i == (len(node.label) - 1):
                        d_label += [1]
                    else:
                        d_label[i+1] += 1
                    
                    ###################################
                    #Choose here the duplication rates#
                    ###################################
                    d_rate = d * (i+1) * node.label[i]
                    #d_rate = d * (i+2) * (i+1) / 2 * (d_label[i+1])
                    
                    if graph.is_in(d_label):
                        graph.add_edge(node, graph.get_node(d_label), d_rate, d_rate)
                    else:
                        p_out += d_rate
                    
            if p_out > 0:
                graph.add_edge(node, graph.get_node([-1]), p_out, p_out)
        
        #difference of prob and rate
        for node in all_nodes:
            total_rate = sum([rate for node, rate, prob in node.out_edges])
            if total_rate == 0.:
                total_rate = 1.
            for i in range(len(node.out_edges)):
                node.out_edges[i][2] /= total_rate
                
                next_node = node.out_edges[i][0]
                for j in range(len(next_node.in_edges)):
                    if next_node.in_edges[j][0].label == node.label:
                        next_node.in_edges[j][2] /= total_rate
                
            
            
    
    def det_positions(graph):
        pos_dict = {}
        
        # Populate the dictionary with nodes
        for node in graph.nodes:
            pos = (node.x_pos, node.y_pos)
            if pos not in pos_dict:
                pos_dict[pos] = []
            pos_dict[pos].append(node)
        
        # Adjust x positions for nodes with the same (x, y) coordinates
        for pos, nodes in pos_dict.items():
            if len(nodes) > 1:
                n = len(nodes)
                x_offset = np.linspace(-0.25, 0.25, n)
                for i, node in enumerate(nodes):
                    node.x_pos = pos[0] + x_offset[i]
    

    #main#
    graph = Graph()
    #print("Generate Nodes")
    generate_nodes(graph, Node(start_label))
    graph.add_node([-1])
    #print("Generate Edges")
    generate_edges(graph, d, N)
    #print("Determine positions")
    det_positions(graph)
    return graph

def draw_graph(graph,nodes=True,coal=True,dupl=True,out=False,size=(32,16)):
    fig, ax = plt.subplots(figsize=size)
    min_ind = 0
    if out:
        min_ind = -2
        
    for node in tqdm(graph.nodes):
        if node.no_ind > min_ind:
            ax.plot(node.x_pos, node.y_pos, 'o', markersize=10, color='grey')
            ax.text(node.x_pos, node.y_pos-0.1, f' {node.label}', fontsize=10, va='top',ha='center')
            if (dupl or coal):
                for to_node, rate, prob in node.out_edges:
                    if to_node.no_ind > min_ind:
                        if ((node.no_ind == to_node.no_ind) and dupl):
                            ax.annotate('', xy=(to_node.x_pos, to_node.y_pos), xytext=(node.x_pos, node.y_pos),
                                        arrowprops=dict(arrowstyle="->", color='blue', lw=0.5))
                        if ((node.no_ind != to_node.no_ind)  and coal):
                            ax.annotate('', xy=(to_node.x_pos, to_node.y_pos), xytext=(node.x_pos, node.y_pos),
                                        arrowprops=dict(arrowstyle="->", color='red', lw=0.5))                
    #plt.savefig('ADG_Random_walk.pdf')
    plt.show()

def success_prob(graph, my_label):
    start = [1] + (len(my_label) - 1) * [0]
    nodes = graph.nodes
    n = len(nodes)
    index_map = {tuple(node.label): i for i, node in enumerate(nodes)}

    A = np.zeros((n, n))
    B1 = np.zeros(n)
    B2 = np.zeros(n)
    
    for node in nodes:
        i = index_map[tuple(node.label)]
        if node.label == my_label:
            A[i, i] = 1
            B1[i] = 1
            B2[i] = 0
        elif node.label == [-1]:
            A[i, i] = 1
            B1[i] = 0
            B2[i] = 1
        else:
            A[i, i] = 1
            for next_node, rate, prob in node.out_edges:
                j = index_map[tuple(next_node.label)]
                A[i, j] -= prob

    P1 = np.linalg.solve(A, B1)
    P2 = np.linalg.solve(A, B2)

    start_index = index_map[tuple(start)]
    return P1[start_index], P2[start_index], P1[start_index] / P2[start_index]
