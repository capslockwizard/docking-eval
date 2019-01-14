from numba import jit
import MDAnalysis
from MDAnalysis.analysis import align
import numpy as np
import pandas as pd
from drsip_common import get_best_fit_rot_mat

@jit('void(f4[:,:],f4[:,:],f4,i4[:,:])', nopython=True)
def pairwiseContact(crdA, crdB, cutOff, contactMap):
    cutOff_sq = cutOff**2
    lenA = crdA.shape[0]
    lenB = crdB.shape[0]

    for indexA in xrange(lenA):
        for indexB in xrange(lenB):
            sum_diff_sqr = 0
            x_diff = crdA[indexA, 0] - crdB[indexB, 0]
            if np.abs(x_diff) > cutOff:
                contactMap[indexA, indexB] = 0
                continue
            sum_diff_sqr += x_diff ** 2

            y_diff = crdA[indexA, 1] - crdB[indexB, 1]
            if np.abs(y_diff) > cutOff:
                contactMap[indexA, indexB] = 0
                continue
            sum_diff_sqr += y_diff ** 2
            if sum_diff_sqr > cutOff_sq:
                contactMap[indexA, indexB] = 0
                continue

            z_diff = crdA[indexA, 2] - crdB[indexB, 2]
            if np.abs(z_diff) > cutOff:
                contactMap[indexA, indexB] = 0
                continue
            sum_diff_sqr += z_diff ** 2
            if sum_diff_sqr < cutOff_sq:
                contactMap[indexA, indexB] = 1
            else:
                contactMap[indexA, indexB] = 0


@jit('void(int32[:],int32[:],int32[:,:],int32[:,:])', nopython=True)
def atomlevel2reslevel(proA_n_list, proB_n_list, atomlevelContactMatrix, reslevelContactMatrix):
    n_residuesA = len(proA_n_list) - 1
    n_residuesB = len(proB_n_list) - 1

    for resi_indexA in xrange(n_residuesA):

        for resi_indexB in xrange(n_residuesB):
            iscontact = 0
            reslevelContactMatrix[resi_indexA, resi_indexB] = 0

            for row_index in xrange(proA_n_list[resi_indexA], proA_n_list[resi_indexA + 1]):

                if iscontact:
                    break

                for column_index in range(proB_n_list[resi_indexB], proB_n_list[resi_indexB + 1]):
                    iscontact = atomlevelContactMatrix[row_index, column_index]

                    if iscontact:
                        reslevelContactMatrix[resi_indexA,
                                              resi_indexB] = iscontact
                        break


class Native_Contacts(object):

    def __init__(self, ref_atomsel, subunit_1_sel_str, subunit_2_sel_str, cutoff=5.0):
        self.ref_atomsel = ref_atomsel
        self.subunit_1_sel_str = subunit_1_sel_str
        self.subunit_2_sel_str = subunit_2_sel_str
        self.subunit_1_sel = self.ref_atomsel.select_atoms(
            self.subunit_1_sel_str)
        self.subunit_2_sel = self.ref_atomsel.select_atoms(
            self.subunit_2_sel_str)
        self.cutoff = np.float32(cutoff)

        self.get_native_contact_mat()

    def get_native_contact_mat(self):
        temp_atomlevelContactMatrix = np.zeros(
            (self.subunit_1_sel.n_atoms, self.subunit_2_sel.n_atoms), dtype='int32')
        temp_reslevelContactMatrix = np.zeros(
            (self.subunit_1_sel.n_residues, self.subunit_2_sel.n_residues), dtype='int32')

        pairwiseContact(self.subunit_1_sel.positions, self.subunit_2_sel.positions,
                        self.cutoff, temp_atomlevelContactMatrix)
        atomlevel2reslevel(self.getNumberOfAtomPerResList(self.subunit_1_sel), self.getNumberOfAtomPerResList(self.subunit_2_sel), temp_atomlevelContactMatrix, temp_reslevelContactMatrix)

        contact_mat = pd.DataFrame(temp_reslevelContactMatrix,
                                    index = self.getMultiIndex(self.subunit_1_sel),
                                    columns = self.getMultiIndex(self.subunit_2_sel),
                                    dtype ='float')
        contact_idx = np.nonzero(contact_mat.values)

        self.native_contact_table = contact_mat.iloc[np.unique(
            contact_idx[0]), :].iloc[:, np.unique(contact_idx[1])]
        self.native_contact_mat = self.native_contact_table.values.astype(
            'bool')
        self.total_native_contacts = self.native_contact_mat.sum().astype('float32')
        
        subunit_1_sel_strs = []
        subunit_2_sel_strs = []

        for segid in self.native_contact_table.index.get_level_values(0).unique():
            subunit_1_sel_strs.append('(segid {} and resid {})'.format(segid, ' '.join([str(resid) for resid in self.native_contact_table.loc[segid].index.get_level_values(0)])))

        for segid in self.native_contact_table.columns.get_level_values(0).unique():
            subunit_2_sel_strs.append('(segid {} and resid {})'.format(segid, ' '.join([str(resid) for resid in self.native_contact_table[segid].columns.get_level_values(0)])))

        self.subunit_1_int_sel_str = self.subunit_1_sel_str + ' and ({})'.format(' or '.join(subunit_1_sel_strs))
        self.subunit_2_int_sel_str = self.subunit_2_sel_str + ' and ({})'.format(' or '.join(subunit_2_sel_strs))

        self.subunit_1_int_sel = self.subunit_1_sel.select_atoms(
            self.subunit_1_int_sel_str)
        self.subunit_2_int_sel = self.subunit_2_sel.select_atoms(
            self.subunit_2_int_sel_str)

        self.subunit_1_int_num_atoms = self.subunit_1_int_sel.n_atoms
        self.subunit_2_int_num_atoms = self.subunit_2_int_sel.n_atoms
        self.subunit_1_int_num_res = self.subunit_1_int_sel.n_residues
        self.subunit_2_int_num_res = self.subunit_2_int_sel.n_residues

        self.subunit_1_int_num_atom_per_res = self.getNumberOfAtomPerResList(
            self.subunit_1_int_sel)
        self.subunit_2_int_num_atom_per_res = self.getNumberOfAtomPerResList(
            self.subunit_2_int_sel)

        self.atomlevelContactMatrix = np.zeros(
            (self.subunit_1_int_num_atoms, self.subunit_2_int_num_atoms), dtype='int32')
        self.reslevelContactMatrix = np.zeros(
            (self.subunit_1_int_num_res, self.subunit_2_int_num_res), dtype='int32')

    def getMultiIndex(self, atom_selection):
        """create an pandas multiIndex object

            Args:
                atom_selection
            Returns:
                multiIndex object
        """
        return pd.MultiIndex.from_arrays(
            [atom_selection.residues.segids, atom_selection.residues.resids, atom_selection.residues.resnames],
            names=['segIds', 'resIds', 'resNames'])

    def calc_percent_native_contacts(self, subunit_1_int_coord, subunit_2_int_coord):
        
        if subunit_1_int_coord.shape[0] != self.subunit_1_int_num_atoms:
            raise Exception(
                'The number of atoms in the interface of subunit 1 is not: %d' % self.subunit_1_int_num_atoms)

        if subunit_2_int_coord.shape[0] != self.subunit_2_int_num_atoms:
            raise Exception(
                'The number of atoms in the interface of subunit 2 is not: %d' % self.subunit_2_int_num_atoms)

        if subunit_1_int_coord.shape[1] != 3 or subunit_2_int_coord.shape[1] != 3:
            raise Exception('The coordinates are not Nx3')

        pairwiseContact(subunit_1_int_coord, subunit_2_int_coord,
                        self.cutoff, self.atomlevelContactMatrix)
        atomlevel2reslevel(self.subunit_1_int_num_atom_per_res, self.subunit_2_int_num_atom_per_res,
                           self.atomlevelContactMatrix, self.reslevelContactMatrix)
        
        return (self.reslevelContactMatrix * self.native_contact_mat).sum() / self.total_native_contacts * 100

    def getNumberOfAtomPerResList(self, sel):
        current_residue_obj = sel[0].residue
        n_list = np.zeros(sel.n_residues + 1, dtype="int32")
        residue_idx = 1
        count = 0
        for atom in sel:
            if current_residue_obj != atom.residue:
                n_list[residue_idx] = count
                residue_idx += 1
                current_residue_obj = atom.residue
            count += 1
        n_list[residue_idx] = count

        return n_list


class lRMSD(object):
    """Ligand RMSD
    """

    def __init__(self, ref_atomsel, rec_sel_str, lig_sel_str):
        self.ref_atomsel = ref_atomsel
        self.rec_sel_str = rec_sel_str + ' and backbone'
        self.lig_sel_str = lig_sel_str + ' and backbone'

        self.ref_rec_coord = self.ref_atomsel.select_atoms(self.rec_sel_str).positions
        self.ref_lig_coord = self.ref_atomsel.select_atoms(self.lig_sel_str).positions
        self.ref_lig_num_atoms = self.ref_lig_coord.shape[0]

        self.ref_rec_COM = self.ref_rec_coord.mean(axis=0)
        self.ref_rec_ori_coord = self.ref_rec_coord - self.ref_rec_COM

    def calc_lRMSD(self, rec_coord, lig_coord):
        rec_COM = rec_coord.mean(axis=0)
        rec_ori_coord = rec_coord - rec_COM

        rot_mat = get_best_fit_rot_mat(rec_ori_coord, self.ref_rec_ori_coord).astype('float32')

        aligned_lig_coord = (lig_coord - rec_COM).dot(rot_mat.T) + self.ref_rec_COM

        return np.sqrt(np.sum((aligned_lig_coord - self.ref_lig_coord)**2) / self.ref_lig_num_atoms)


class iRMSD(object):
    """Interface RMSD (iRMSD)

    Identifies the interface residues by checking heavy atoms within 10 Angstrom between the 2 subunits in the reference structure.
    The iRMSD is the RMSD of the backbone atoms of these interface residues between the reference and predicted poses. The poses are superimposed at the interface residues' backbone atoms.
    """

    def __init__(self, ref_atomsel, subunit_1_sel_str, subunit_2_sel_str, cutoff=10.0):
        self.ref_atomsel = ref_atomsel
        self.subunit_1_sel_str = subunit_1_sel_str
        self.subunit_2_sel_str = subunit_2_sel_str
        self.cutoff = cutoff

        self.native_contact_obj = Native_Contacts(
            self.ref_atomsel, self.subunit_1_sel_str, self.subunit_2_sel_str, cutoff=self.cutoff)

        self.subunit_1_int_sel_str = self.native_contact_obj.subunit_1_int_sel_str + ' and backbone'
        self.subunit_2_int_sel_str = self.native_contact_obj.subunit_2_int_sel_str + ' and backbone'

        self.subunit_1_int_sel = self.native_contact_obj.subunit_1_int_sel.select_atoms(
            'backbone')
        self.subunit_2_int_sel = self.native_contact_obj.subunit_2_int_sel.select_atoms(
            'backbone')

        self.subunit_1_int_num_atoms = self.subunit_1_int_sel.n_atoms
        self.subunit_2_int_num_atoms = self.subunit_2_int_sel.n_atoms
        self.int_total_num_atoms = self.subunit_1_int_num_atoms + self.subunit_2_int_num_atoms

        self.ref_int_coord = np.zeros(
            (self.int_total_num_atoms, 3), dtype=np.float32)

        self.ref_int_coord[:self.subunit_1_int_num_atoms,
                           :] = self.subunit_1_int_sel.positions
        self.ref_int_coord[self.subunit_1_int_num_atoms:,
                           :] = self.subunit_2_int_sel.positions
        self.ref_int_coord -= np.mean(self.ref_int_coord, axis=0)

    def calc_iRMSD(self, int_coord):

        if int_coord.shape[0] != self.int_total_num_atoms:
            raise Exception('The number of atoms in the interface is not: %d but %d' % (
                self.int_total_num_atoms, int_coord.shape[0]))

        if int_coord.shape[1] != 3:
            raise Exception('The coordinates are not Nx3')

        int_coord_origin = int_coord - np.mean(int_coord, axis=0)

        rot_mat = np.array(align.rotation_matrix(
            int_coord_origin, self.ref_int_coord)[0])
        int_coord_superimposed = rot_mat.dot(int_coord_origin.T).T

        return np.sqrt(np.sum((int_coord_superimposed - self.ref_int_coord)**2) / self.int_total_num_atoms)


class CAPRI_Criteria(object):

    def __init__(self):
        self.irmsd_bin_edges = np.array([4.0, 2.0, 1.0, -0.1])
        self.lrmsd_bin_edges = np.array([10.0, 5.0, 1.0, -0.1])
        self.perc_nat_contact_bin_edges = np.array([0, 10.0, 30.0, 50.0])
        self.CAPRI_pred_class_num_to_name = {
            0: 'Incorrect', 1: 'Acceptable', 2: 'Medium', 3: 'High'}

    def classify(self, nat_contact, lrmsd, irmsd):
        highest_class = np.max([np.digitize(lrmsd, self.lrmsd_bin_edges, right=True), np.digitize(irmsd, self.irmsd_bin_edges, right=True)])

        return self.CAPRI_pred_class_num_to_name[np.min([np.digitize(nat_contact, self.perc_nat_contact_bin_edges) - 1, highest_class])]


def assign_Cn_symm_CAPRI_class(ref_atomsel, static_sel_str, mobile_sel_str, dock_atomsel, num_mers, predicted_num_mers):
    """CAPRI classification of docking results

    Native contact and iRMSD calculation:
    The interface residues were determined from the reference PDB structure
    """
    capri_criteria_inst = CAPRI_Criteria()

    # Find the residues at the interface of the two subunits & their selection string
    # Then swap their chain IDs
    native_contacts_obj = Native_Contacts(
        ref_atomsel, static_sel_str, mobile_sel_str)

    irmsd_obj = iRMSD(ref_atomsel, static_sel_str, mobile_sel_str)
    
    lrmsd_obj = lRMSD(ref_atomsel, static_sel_str, mobile_sel_str)
    swapped_lrmsd_obj = lRMSD(ref_atomsel, mobile_sel_str, static_sel_str)

    lrmsd_rec_sel = dock_atomsel.select_atoms(lrmsd_obj.rec_sel_str)
    lrmsd_rec_coord = lrmsd_rec_sel.positions
    lrmsd_lig_sel = dock_atomsel.select_atoms(lrmsd_obj.lig_sel_str)

    # Swapped
    static_segid_sel_str = 'segid ' + \
    np.unique(native_contacts_obj.subunit_1_sel.segids)[0]
    
    mobile_segid_sel_str = 'segid ' + \
    np.unique(native_contacts_obj.subunit_2_sel.segids)[0]
     
    swapped_native_contacts_static_int_sel_str = native_contacts_obj.subunit_2_int_sel_str.replace(
        mobile_segid_sel_str, static_segid_sel_str)
    swapped_native_contacts_mobile_int_sel_str = native_contacts_obj.subunit_1_int_sel_str.replace(
        static_segid_sel_str, mobile_segid_sel_str)

    swapped_irmsd_static_int_sel_str = irmsd_obj.subunit_2_int_sel_str.replace(
        mobile_segid_sel_str, static_segid_sel_str)
    swapped_irmsd_mobile_int_sel_str = irmsd_obj.subunit_1_int_sel_str.replace(
        static_segid_sel_str, mobile_segid_sel_str)
    
    # Original
    native_contacts_static_int_sel_str = native_contacts_obj.subunit_1_int_sel_str
    native_contacts_mobile_int_sel_str = native_contacts_obj.subunit_2_int_sel_str

    irmsd_static_int_sel_str = irmsd_obj.subunit_1_int_sel_str
    irmsd_mobile_int_sel_str = irmsd_obj.subunit_2_int_sel_str
    
    native_contact_static_int_sel = dock_atomsel.select_atoms(native_contacts_static_int_sel_str)
    native_contact_static_int_coord = native_contact_static_int_sel.positions
    
    swapped_native_contact_static_int_sel = dock_atomsel.select_atoms(swapped_native_contacts_static_int_sel_str)
    swapped_native_contact_static_int_coord = swapped_native_contact_static_int_sel.positions
    
    irmsd_static_int_sel = dock_atomsel.select_atoms(irmsd_static_int_sel_str)
    irmsd_static_int_coord = irmsd_static_int_sel.positions
    irmsd_static_int_num_atoms = irmsd_static_int_sel.n_atoms
    irmsd_total_num_atoms = dock_atomsel.select_atoms('(' + irmsd_static_int_sel_str + ') or (' + irmsd_mobile_int_sel_str + ')').n_atoms
    
    swapped_irmsd_static_int_sel = dock_atomsel.select_atoms(swapped_irmsd_static_int_sel_str)
    swapped_irmsd_static_int_coord = swapped_irmsd_static_int_sel.positions
    swapped_irmsd_static_int_num_atoms = swapped_irmsd_static_int_sel.n_atoms
    swapped_irmsd_total_num_atoms = dock_atomsel.select_atoms('(' + swapped_irmsd_static_int_sel_str + ') or (' + swapped_irmsd_mobile_int_sel_str + ')').n_atoms
    
    if irmsd_total_num_atoms != swapped_irmsd_total_num_atoms:
        raise Exception('The number of atoms in the static and mobile subunits are not the same')
    
    native_contact_mobile_int_sel = dock_atomsel.select_atoms(native_contacts_mobile_int_sel_str)
    irmsd_mobile_int_sel = dock_atomsel.select_atoms(irmsd_mobile_int_sel_str)
    swapped_native_contact_mobile_int_sel = dock_atomsel.select_atoms(swapped_native_contacts_mobile_int_sel_str)
    swapped_irmsd_mobile_int_sel = dock_atomsel.select_atoms(swapped_irmsd_mobile_int_sel_str)

    universe = dock_atomsel.universe

    per_nat_con_n_lRMSD_n_iRMSD, CAPRI_class = compute_Cn_symm_CAPRI_class(irmsd_total_num_atoms, swapped_irmsd_static_int_num_atoms, irmsd_static_int_num_atoms, native_contact_static_int_coord, swapped_native_contact_static_int_coord, irmsd_static_int_coord, swapped_irmsd_static_int_coord, lrmsd_rec_coord, lrmsd_lig_sel, universe, native_contacts_obj, irmsd_obj, lrmsd_obj, swapped_lrmsd_obj, native_contact_mobile_int_sel, swapped_native_contact_mobile_int_sel, irmsd_mobile_int_sel, swapped_irmsd_mobile_int_sel, capri_criteria_inst)

    CAPRI_class_table = pd.DataFrame(per_nat_con_n_lRMSD_n_iRMSD, index=range(
        1, universe.trajectory.n_frames + 1), columns=['% Native Contacts', 'lRMSD', 'iRMSD'])
    CAPRI_class_table['Classification'] = CAPRI_class

    # Additional criterion to ensure that the order of symmetry is the same as in the reference structure
    CAPRI_class_table.loc[CAPRI_class_table['Order of Symmetry'] != num_mers] = 'Incorrect'

    return CAPRI_class_table

def compute_Cn_symm_CAPRI_class(irmsd_total_num_atoms, swapped_irmsd_static_int_num_atoms, irmsd_static_int_num_atoms, native_contact_static_int_coord, swapped_native_contact_static_int_coord, irmsd_static_int_coord, swapped_irmsd_static_int_coord, lrmsd_rec_coord, lrmsd_lig_sel, universe, native_contacts_obj, irmsd_obj, lrmsd_obj, swapped_lrmsd_obj, native_contact_mobile_int_sel, swapped_native_contact_mobile_int_sel, irmsd_mobile_int_sel, swapped_irmsd_mobile_int_sel, capri_criteria_inst):
    iRMSD_int_coord = np.zeros((irmsd_total_num_atoms, 3), dtype=np.float32)
    iRMSD_int_coord[:irmsd_static_int_num_atoms,
            :] = irmsd_static_int_coord
    
    swapped_irmsd_mobile_int_num_atoms = irmsd_total_num_atoms - swapped_irmsd_static_int_num_atoms
    swapped_iRMSD_int_coord = np.zeros((irmsd_total_num_atoms, 3), dtype=np.float32)
    swapped_iRMSD_int_coord[swapped_irmsd_mobile_int_num_atoms:,
            :] = swapped_irmsd_static_int_coord

    per_nat_con_n_lRMSD_n_iRMSD = np.zeros(
        (universe.trajectory.n_frames, 3), 'float32')
    CAPRI_class = []

    temp_percent_native_contacts = np.zeros(2, dtype=np.float32)
    swap = np.zeros(1, dtype=np.bool)

    for ts in universe.trajectory:
        idx = ts.frame
        pose_num = idx + 1

        native_contact_mobile_int_coord = native_contact_mobile_int_sel.positions
        swapped_native_contact_mobile_int_coord = swapped_native_contact_mobile_int_sel.positions

        temp_percent_native_contacts[0] = native_contacts_obj.calc_percent_native_contacts(native_contact_static_int_coord, native_contact_mobile_int_coord)
        temp_percent_native_contacts[1] = native_contacts_obj.calc_percent_native_contacts(swapped_native_contact_mobile_int_coord, swapped_native_contact_static_int_coord)

        (per_nat_con_n_lRMSD_n_iRMSD[idx, 0], swap[:]) = (np.max(temp_percent_native_contacts), np.argmax(temp_percent_native_contacts))
        
        if per_nat_con_n_lRMSD_n_iRMSD[idx, 0] >= 10:
            lRMSD_lig_coord = lrmsd_lig_sel.positions

            if swap:
                swapped_iRMSD_int_coord[:swapped_irmsd_mobile_int_num_atoms, :] = swapped_irmsd_mobile_int_sel.positions
                per_nat_con_n_lRMSD_n_iRMSD[idx, 1] = np.min([lrmsd_obj.calc_lRMSD(lRMSD_lig_coord, lrmsd_rec_coord), swapped_lrmsd_obj.calc_lRMSD(lrmsd_rec_coord, lRMSD_lig_coord)])
                per_nat_con_n_lRMSD_n_iRMSD[idx, 2] = irmsd_obj.calc_iRMSD(swapped_iRMSD_int_coord)

            else:
                iRMSD_int_coord[irmsd_static_int_num_atoms:, :] = irmsd_mobile_int_sel.positions
                per_nat_con_n_lRMSD_n_iRMSD[idx, 1] = np.min([lrmsd_obj.calc_lRMSD(lrmsd_rec_coord, lRMSD_lig_coord), swapped_lrmsd_obj.calc_lRMSD(lRMSD_lig_coord, lrmsd_rec_coord)])
                per_nat_con_n_lRMSD_n_iRMSD[idx, 2] = irmsd_obj.calc_iRMSD(iRMSD_int_coord)

            CAPRI_class.append(capri_criteria_inst.classify(
                *per_nat_con_n_lRMSD_n_iRMSD[idx]))

        else:
            per_nat_con_n_lRMSD_n_iRMSD[idx, 1:] = 100.0
            CAPRI_class.append('Incorrect')

    return (per_nat_con_n_lRMSD_n_iRMSD, CAPRI_class)

def assign_soluble_CAPRI_class(ref_atomsel, static_sel_str, mobile_sel_str, dock_atomsel):
    """CAPRI classification of docking results

    Native contact and iRMSD calculation:
    The interface residues were determined from the reference PDB structure
    """
    capri_criteria_inst = CAPRI_Criteria()
    
    # Find the residues at the interface of the two subunits & their selection string
    # Then swap their chain IDs
    native_contacts_obj = Native_Contacts(
        ref_atomsel, static_sel_str, mobile_sel_str)

    irmsd_obj = iRMSD(ref_atomsel, static_sel_str, mobile_sel_str)
    lrmsd_obj = lRMSD(ref_atomsel, static_sel_str, mobile_sel_str)

    lrmsd_rec_sel = dock_atomsel.select_atoms(lrmsd_obj.rec_sel_str)
    lrmsd_rec_coord = lrmsd_rec_sel.positions
    lrmsd_lig_sel = dock_atomsel.select_atoms(lrmsd_obj.lig_sel_str)

    native_contacts_static_int_sel_str = native_contacts_obj.subunit_1_int_sel_str
    native_contacts_mobile_int_sel_str = native_contacts_obj.subunit_2_int_sel_str

    irmsd_static_int_sel_str = irmsd_obj.subunit_1_int_sel_str
    irmsd_mobile_int_sel_str = irmsd_obj.subunit_2_int_sel_str
        
    native_contact_static_int_sel = dock_atomsel.select_atoms(native_contacts_static_int_sel_str)
    native_contact_static_int_coord = native_contact_static_int_sel.positions
    
    irmsd_static_int_sel = dock_atomsel.select_atoms(irmsd_static_int_sel_str)
    irmsd_static_int_coord = irmsd_static_int_sel.positions
    irmsd_static_int_num_atoms = irmsd_static_int_sel.n_atoms
    irmsd_total_num_atoms = dock_atomsel.select_atoms('(' + irmsd_static_int_sel_str + ') or (' + irmsd_mobile_int_sel_str + ')').n_atoms

    native_contact_mobile_int_sel = dock_atomsel.select_atoms(native_contacts_mobile_int_sel_str)
    irmsd_mobile_int_sel = dock_atomsel.select_atoms(irmsd_mobile_int_sel_str)

    universe = dock_atomsel.universe

    per_nat_con_n_lRMSD_n_iRMSD, CAPRI_class = compute_soluble_CAPRI_class(irmsd_total_num_atoms, irmsd_static_int_num_atoms, native_contact_static_int_coord, irmsd_static_int_coord, lrmsd_rec_coord, lrmsd_lig_sel, universe, native_contacts_obj, irmsd_obj, lrmsd_obj, native_contact_mobile_int_sel, irmsd_mobile_int_sel, capri_criteria_inst)

    CAPRI_class_table = pd.DataFrame(per_nat_con_n_lRMSD_n_iRMSD, index=range(
        1, universe.trajectory.n_frames + 1), columns=['% Native Contacts', 'lRMSD', 'iRMSD'])
    CAPRI_class_table['Classification'] = CAPRI_class

    return CAPRI_class_table

def compute_soluble_CAPRI_class(irmsd_total_num_atoms, irmsd_static_int_num_atoms, native_contact_static_int_coord, irmsd_static_int_coord, lrmsd_rec_coord, lrmsd_lig_sel, universe, native_contacts_obj, irmsd_obj, lrmsd_obj, native_contact_mobile_int_sel, irmsd_mobile_int_sel, capri_criteria_inst):
    iRMSD_int_coord = np.zeros((irmsd_total_num_atoms, 3), dtype=np.float32)
    iRMSD_int_coord[:irmsd_static_int_num_atoms,
            :] = irmsd_static_int_coord

    per_nat_con_n_lRMSD_n_iRMSD = np.zeros(
        (universe.trajectory.n_frames, 3), 'float32')
    CAPRI_class = []

    temp_percent_native_contacts = np.zeros(2, dtype=np.float32)

    for ts in universe.trajectory:
        idx = ts.frame
        pose_num = idx + 1

        native_contact_mobile_int_coord = native_contact_mobile_int_sel.positions

        per_nat_con_n_lRMSD_n_iRMSD[idx, 0] = native_contacts_obj.calc_percent_native_contacts(native_contact_static_int_coord, native_contact_mobile_int_coord)

        if per_nat_con_n_lRMSD_n_iRMSD[idx, 0] >= 10:
            lRMSD_lig_coord = lrmsd_lig_sel.positions

            iRMSD_int_coord[irmsd_static_int_num_atoms:, :] = irmsd_mobile_int_sel.positions
            per_nat_con_n_lRMSD_n_iRMSD[idx, 1] = lrmsd_obj.calc_lRMSD(lrmsd_rec_coord, lRMSD_lig_coord)
            per_nat_con_n_lRMSD_n_iRMSD[idx, 2] = irmsd_obj.calc_iRMSD(iRMSD_int_coord)

            CAPRI_class.append(capri_criteria_inst.classify(
                *per_nat_con_n_lRMSD_n_iRMSD[idx]))

        else:
            per_nat_con_n_lRMSD_n_iRMSD[idx, 1:] = 100.0
            CAPRI_class.append('Incorrect')

    return (per_nat_con_n_lRMSD_n_iRMSD, CAPRI_class)
