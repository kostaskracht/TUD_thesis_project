# import libraries
import numpy as np
import scipy.io as spio
import os
import pandas as pd

class InitDataPrep():
    def __init__(self):
        self.init_data_path = "data/init_data/sioux_falls"
        self.output_data_path = "data/extracted_data/sioux_falls"

        # total number of components per type
        # self.max_interstate = 12
        # self.max_prim = 47
        # self.max_second = 26

        # working number of components per type
        # self.intrstate_comp = 8
        # self.primr_comp = 12
        # self.sec_comp = 0
        # self.ncomp_pav = self.intrstate_comp + self.primr_comp + self.sec_comp  # number of component pavements
        # tot_comp = ncomp_deck + ncomp_pav

        # number of states
        self.nstcomp_CCI = 6  # number of states per pavement CCI
        self.nstcomp_IRI = 5  # number of states per pavement IRI

        # number of observations (should be equal to states)
        # no-inspection-{1,2,3 mainetenance}, low fed_insp-{1,2,3 mainetenance}, high_fed_insp-{1,2,3 mainetenance}, and replacement
        self.nobs_CCI = 6  # number of observations for CCI
        self.nobs_IRI = 5  # number of observations IRI

        # number of actions
        self.nacomp = 10  # number of actions per component = 3*3 +1 when replacement

        # Get the indices of the used components
        # self.pav_index1 = [x for x in range(12) if x not in [3, 4, 5, 11]]
        #
        # self.pav_index22 = [3, 4, 10, 12, 13, 15, 34, 41, 42, 44, 45, 46]  ## subtract the position of 18-21 link
        # self.pav_index222 = [12 + nn for nn in self.pav_index22]

        # self.pav_index = self.pav_index1+self.pav_index222
        self.ncomp_pav = 76
        self.pav_index = np.arange(self.ncomp_pav)

    def get_topology(self):

        filename = f'{self.init_data_path}/SiouxFalls_net.csv'
        network = pd.read_csv(filename, delimiter='\t')
        # mat = spio.loadmat(filename, squeeze_me=True)
        # nodes = mat['Nodes'][:,:]

        nodes = np.reshape(network.index.tolist(), (-1, 1))
        edges = network[["init_node", "term_node"]].to_numpy()
        len_comp = network["length"].to_numpy()
        capacity = network["capacity"].to_numpy()


        return nodes, edges, len_comp, capacity

    def get_num_lanes_area(self, len_comp):
        # Load network geometry
        # filename = f'{self.init_data_path}/link_lengths.mat'
        # mat1 = spio.loadmat(filename, squeeze_me=True)
        # len_comp = mat1['length1'] # length of each link in miles
        # len_comp = len_comp[self.pav_index]
        # Assume two lanes in all road segments
        n_lane_pav = np.ones((len_comp.shape))*2

        area_pav = np.zeros(self.ncomp_pav)
        for i in range(self.ncomp_pav):
            area_pav[i] = (1.61*1000) * len_comp[i] * n_lane_pav[i] * 3.7    # in m sq.

        total_area_pav = np.sum(area_pav)

        return n_lane_pav, area_pav, [total_area_pav]

    def get_trans_probs(self):
        # Compute transition probabilities
        # Load necessary files
        # file_pav06 = (f'{self.init_data_path}/Smoothed_TP_MSI_06.mat')
        # mat_pav06 = spio.loadmat(file_pav06, squeeze_me=True)
        # file_pav08 = (f'{self.init_data_path}/Smoothed_TP_MSI_08.mat')
        # mat_pav08 = spio.loadmat(file_pav08, squeeze_me=True)
        file_pav20 = (f'{self.init_data_path}/../Smoothed_TP_MSI_20.mat')
        mat_pav20 = spio.loadmat(file_pav20, squeeze_me=True)

        # Do nothing
        # CCI
        tp_cci_dn = np.zeros((self.nstcomp_CCI, self.nstcomp_CCI, 20, self.ncomp_pav))
        # TODO - final tp_cci_dn has dimensions 6x6x20x20 instead of 6x6x20x84 --> it's because of ncomp_pav
        # first fifteen components are type I, next 47 comp are type II and the rest are type III
        for i in range(self.ncomp_pav):
            # if i < self.intrstate_comp:
            #     tp_cci_dn[:,:,:,i] = mat_pav06['prob2']
            # elif i >= self.intrstate_comp and i < (self.intrstate_comp + self.primr_comp):
            #     tp_cci_dn[:,:,:,i] = mat_pav08['prob2']
            # elif i>=(self.intrstate_comp + self.primr_comp) and i < self.ncomp_pav:
            tp_cci_dn[:,:,:,i] = mat_pav20['prob2']

        # IRI: Dimensions (nstcomp_CCI x nstcomp_CCI)
        # TODO: Should add the time, and component dimension!
        tp_iri_dn = np.array([[0.839,	0.121,	0.039,	  0.,    0.],
                              [ 0.,    0.787,	0.142,	0.07,    0.],
                              [ 0.,     0.,    0.708,	0.192,	0.099],
                              [0.,     0.,       0.,  0.578,	0.421],
                              [0.,    0.,       0.,    0.,     1]])

        # Minor repair
        # TODO - Current shape: 6x6. Should be 6x6x20 (time dimension)
        # CCI
        tp_cci_minor= np.array([[0.97,	0.03,	0,	0,	0,	0],
                                [0.87,	0.1,	0.03,	0,	0,	0],
                                [0.4,	0.47,	0.1,	0.03,	0,	0],
                                [0,	0.4,	0.47,	0.1,	0.03,	0],
                                [0,	0,	0.4,	0.47,	0.1,	0.03],
                                [0,	0,	0,	0.4,	0.47,	0.13]])

        # IRI
        tp_iri_minor = np.array([[0.97,	0.03,	0,	0,	0],
                                 [0.85,	0.12,	0.03,	0,	0],
                                 [0.45,	0.4,	0.12,	0.03,	0],
                                 [0,	0.45,	0.4,	0.12,	0.03],
                                 [0,	0,	0.45,	0.4,	0.15]])

        # Major repair
        # TODO - Current shape: 6x6. Should be 6x6x20 (time dimension)
        # CCI
        tp_cci_major = np.array([[1,	      0,	0,   0,	0,	0],
                                 [0.96,	0.04,	0,	 0,	0,	0],
                                 [0.8,	0.2,	0,	 0,	0,	0],
                                 [0.65,	0.25,	0.1,	0,	0,	0],
                                 [0.5,	0.3,	0.2,	0,	0,	0],
                                 [0.4,	0.3,	0.3,	0,	0,	0]])

        # IRI
        tp_iri_major = np.array([[1,	0,	0,	0,	0],
                                 [0.95,	0.05,	0,	0,	0],
                                 [0.80,	0.20,	0,	0,	0],
                                 [0.7,	0.25,	0.05,	0,	0],
                                 [0.45,	0.35,	0.2,	0,	0]])

        # Replacement
        # CCI - Custom created
        tp_cci_replace = np.array([[1,	      0,	0,   0,	0,	0],
                                   [1,	      0,	0,   0,	0,	0],
                                   [1,	      0,	0,   0,	0,	0],
                                   [1,	      0,	0,   0,	0,	0],
                                   [1,	      0,	0,   0,	0,	0],
                                   [1,	      0,	0,   0,	0,	0]])

        # IRI - Custom created
        tp_iri_replace = np.array([[1,	0,	0,	0,	0],
                                   [1,	0,	0,	0,	0],
                                   [1,	0,	0,	0,	0],
                                   [1,	0,	0,	0,	0],
                                   [1,	0,	0,	0,	0]])

        # tp_cci_dn: 6x6x20x20
        # tp_cci_minor: 6x6
        # tp_cci_major: 6x6
        # tp_cci_replace: 6x6
        # tp_iri_dn: 5x5
        # tp_iri_minor: 5x5
        # tp_iri_major: 5x5
        # tp_iri_replace: 5x5

        return \
            tp_cci_dn, \
            tp_cci_minor, \
            tp_cci_major, \
            tp_cci_replace, \
            tp_iri_dn, \
            tp_iri_minor, \
            tp_iri_major, \
            tp_iri_replace

    def get_insp_probs(self):
        # Ispection probabilities
        pobs_insp_cci_low = np.array([[0.688, 0.258, 0.054, 0.000, 0.000, 0.000],
                                      [0.277, 0.422, 0.297, 0.004, 0, 0],
                                      [0.024, 0.14, 0.648, 0.166, 0.022, 0.001],
                                      [0, 0.003, 0.266, 0.455, 0.249, 0.027],
                                      [0, 0, 0.031, 0.223, 0.486, 0.26],
                                      [0, 0, 0, 0.006, 0.061, 0.936]])

        pobs_insp_cci_high = np.array([[0.803, 0.195, 0.002, 0, 0, 0],
                                       [0.152,	0.664,	0.183,	0,	0,	0],
                                       [0.001,	0.078,	0.822,	0.1,	0,	0],
                                       [0,	0,	0.149,	0.693,	0.158,	0],
                                       [0,	0,	0.001,	0.137,	0.718,	0.144],
                                       [0,	0,	0,	0,	0.045,	0.97]])

        pobs_insp_iri_low = np.array([[0.80,	0.20,	0.00,   0.00,	0.00],
                                      [0.20,	0.60,	0.20,	0.00,	0.00],
                                      [0.00,	0.20,	0.60,	0.20,	0.00],
                                      [0.00,	0.00,	0.20,	0.60,	0.20],
                                      [0.00,	0.00,	0.00,	0.20,	0.80]])

        pobs_insp_iri_high = np.array([[0.90,	0.10,	0.00,	0.00,	0.00],
                                       [0.05,	0.90,	0.05,	0.00,	0.00],
                                       [0.00,	0.05,	0.90,	0.05,	0.00],
                                       [0.00,	0.00,	0.05,	0.90,	0.05],
                                       [0.00,	0.00,	0.00,	0.10,	0.90]])

        pobs_iri = np.zeros((self.ncomp_pav, self.nacomp, self.nstcomp_IRI, self.nobs_IRI))
        pobs_cci = np.zeros((self.ncomp_pav, self.nacomp, self.nstcomp_CCI, self.nobs_CCI))

        # for CCI
        for i in range(self.ncomp_pav):
            for j in range(self.nacomp):
                if j in [0,1,2]:
                    pobs_cci[i,j,:,:] = 1/self.nstcomp_CCI
                elif j in [3,4,5]:
                    pobs_cci[i,j,:,:] = pobs_insp_cci_low[:,:]
                elif j in [6,7,8]:
                    pobs_cci[i,j,:,:] = pobs_insp_cci_high[:,:]
                elif j ==9:
                    pobs_cci[i,j,:,:] = 0
                    pobs_cci[i,j,:,0] = 1

        # for IRI
        for i in range (self.ncomp_pav):
            for j in range(10):
                if j in [0,1,2]:
                    pobs_iri[i,j,:,:] = 1/self.nstcomp_IRI
                elif j in [3,4,5]:
                    pobs_iri[i,j,:,:] = pobs_insp_iri_low[:,:]
                elif j in [6,7,8]:
                    pobs_iri[i,j,:,:] = pobs_insp_iri_high[:,:]
                elif j ==9:
                    pobs_iri[i,j,:,:] = 0
                    pobs_iri[i,j,:,0] = 1

        # pobs_cci: 20x10x5x5
        # pobs_iri: 20x10x6x6
        return \
            pobs_cci, \
            pobs_iri

    def get_action_costs(self, len_comp_loc, n_lane_pav_loc):
        # Cost of actions
        # Maintenance actions costs
        # 3x3 matrix: rows --> do nothing/minor/major/replace, columns --> road type
        c_maint = 3.7*np.array([[0, 20, 75, 350],
                                [0, 16, 68, 330],
                                [0, 10 ,52, 250]])  # pav Action costs per meter of lane in USD

        # Inspection actions cost
        # Do nothing/low/high
        c_insp = 3.7*np.array([0, 0.10, 0.20]) #pav inspection cost per meter lane in USD

        cost_comp_action_pav = np.zeros((self.ncomp_pav, self.nacomp))
        cost_comp_obsr_pav = np.zeros((self.ncomp_pav, self.nacomp))

        for i in range(self.ncomp_pav):
            for j in range(9):
                cost_comp_action_pav[i,j] = -(1.61*1000) * len_comp_loc[i] * (c_maint[2][j % 3]) * n_lane_pav_loc[i] / 1000000  #2 bc of 2 lanes
                cost_comp_obsr_pav[i,j] = -(1.61*1000) * len_comp_loc[i] * (c_insp[j // 3]) * n_lane_pav_loc[i] / 1000000
            # For replacement
            for j in [9]:
                cost_comp_obsr_pav[i,j] = 0
                cost_comp_action_pav[i,j] = -(1.61*1000) * len_comp_loc[i] * (c_maint[2][3]) * n_lane_pav_loc[i] / 1000000  #2 bc of 2 lanes

        # cost_comp_action_pav: (20x10)
        # cost_comp_obsr_pav: (20x10)
        return cost_comp_action_pav, cost_comp_obsr_pav

    def get_delay_cost(self):
        # Load the cost of delay and duration of actions
        cost_delay = np.load(f'{self.init_data_path}/Dlay_Cost.npy')
        intdx = list([])
        intdx += self.pav_index
        cost_delay = cost_delay[intdx,:] # shape: num_components x num_actions

        # cost_delay: 20x10
        return cost_delay

    def get_action_duration(self, area_pav):
        act_duration = np.array([0, 0.0006, 0.0011, 0, 0.0006, 0.0011, 0, 0.0006, 0.0011, 0.006])
        # act_duration = act_duration[self.pav_index,:] # shape: num_components x num_actions
        act_duration_fin = np.reshape(area_pav, (-1, 1)) @ np.reshape(act_duration, (1, -1))

        # finding indices find component ids with an action that takes more than 365 days???
        actions_long = (act_duration_fin > 365)*1==1

        # act_duration: 20x10
        # actions_long: 20x10

        return act_duration_fin, actions_long

    def save_attribute(self, attr_tuple):
        (filename, attr, out_type) = attr_tuple
        if out_type == "npy":
            np.save(f"{self.output_data_path}/{filename}.npy", attr)
        elif out_type == "csv":
            print(filename)
            np.savetxt(f"{self.output_data_path}/{filename}.csv", attr, delimiter=",")

if __name__ == "__main__":
    os.chdir("../..")

    prep = InitDataPrep()

    nodes, edges, len_comp, capacity = prep.get_topology()

    n_lane_pav, area_pav, total_area_pav = prep.get_num_lanes_area(len_comp)

    tp_cci_dn, \
    tp_cci_minor, \
    tp_cci_major, \
    tp_cci_replace, \
    tp_iri_dn, \
    tp_iri_minor, \
    tp_iri_major, \
    tp_iri_replace = prep.get_trans_probs()

    pobs_cci, pobs_iri = prep.get_insp_probs()

    cost_comp_action_pav, cost_comp_obsr_pav = prep.get_action_costs(len_comp, n_lane_pav)

    # cost_delay = prep.get_delay_cost()

    act_duration, actions_long = prep.get_action_duration(area_pav)

    attr_to_save = [("nodes", nodes, "csv"),
                    ("edges", edges, "csv"),
                    ("len_comp", len_comp * 1.60934, "csv"),
                    ("capacity", capacity, "csv"),
                    ("n_lane_pav", n_lane_pav, "csv"),
                    ("area_pav", area_pav, "csv"),
                    ("total_area_pav", total_area_pav, "csv"),
                    ("tp_cci_dn", tp_cci_dn, "npy"),
                    ("tp_cci_minor", tp_cci_minor, "csv"),
                    ("tp_cci_major", tp_cci_major, "csv"),
                    ("tp_cci_replace", tp_cci_replace, "csv"),
                    ("tp_iri_dn", tp_iri_dn, "csv"),
                    ("tp_iri_minor", tp_iri_minor, "csv"),
                    ("tp_iri_major", tp_iri_major, "csv"),
                    ("tp_iri_replace", tp_iri_replace, "csv"),
                    ("pobs_cci", pobs_cci, "npy"),
                    ("pobs_iri", pobs_iri, "npy"),
                    ("cost_comp_action_pav", cost_comp_action_pav, "csv"),
                    ("cost_comp_obsr_pav", cost_comp_obsr_pav, "csv"),
                    # ("cost_delay", cost_delay, "csv"),
                    ("act_duration", act_duration, "csv"),
                    ("actions_long", actions_long, "csv")]

    for attribute in attr_to_save:
        prep.save_attribute(attribute)
