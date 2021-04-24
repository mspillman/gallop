import numpy as np
import json

#
# Things to add - extract occupancies from the ZM
#

class Z_matrix(object):
    """
    The GALLOP representation of Z-matrices.
    This has convenience methods for reading in DASH Z-matrices, as well as
    a very limited capability to read in gaussian formatted Z-matrices.
    Methods are included to strip out hydrogen atoms, as well as generate
    Cartesian coordinates.

    One common issue that is encountered is when refineable torsion angles are
    defined in terms of hydrogen atoms. This causes the method to remove the H
    atoms to fail.

    A simple fix for this is to:
        1. If not already available, generate a CIF of the structure from which
            the ZMs are being derived.
        2. Reorder the atoms in the CIF such that all non-H atoms come first.
        3. Regenerate the ZMs using makeZmatrix.exe (bundled with DASH)

    The new ZMs will no longer have H-atoms defining torsion angles and should
    not give any errors with this class.
    """
    def __init__(self, filename=None, zmformat="DASH"):
        if filename is not None:
            # Read in the Z-matrix
            self.filename = filename
            if zmformat.lower() == "dash":
                self.read_DASH_zm(filename)
            elif zmformat.lower() == "gaussian":
                self.read_Gaussian_zm(filename)
            self.coords_radians = self.zm_angles_to_radians(self.coords)

            # Create a new Z-matrix with no hydrogen atoms
            self.remove_H_from_zm()
            self.coords_radians_no_H=self.zm_angles_to_radians(self.coords_no_H)
            # Zero indexing needed in Python
            self.bond_connection -= 1
            self.angle_connection -= 1
            self.torsion_connection -= 1
            self.bond_connection_no_H -= 1
            self.angle_connection_no_H -= 1
            self.torsion_connection_no_H -= 1
            # Generate some initial Cartesian coordinates for the Z-matrices
            self.initial_cartesian = self.zm_to_cart(self.coords_radians,
                                                self.bond_connection,
                                                self.angle_connection,
                                                self.torsion_connection)
            try:
                self.initial_cartesian_no_H = self.zm_to_cart(
                                                self.coords_radians_no_H,
                                                self.bond_connection_no_H,
                                                self.angle_connection_no_H,
                                                self.torsion_connection_no_H)
                self.H_atom_torsion_defs = False
            except IndexError:
                print("Error in Z-matrix " + self.filename + \
                ": check to see if refineable torsions are defined in terms of \
                hydrogen atoms in original Z-matrix")
                print("All atoms refineable torsions = ",
                                            self.torsion_refineable.sum())
                print("Non-H atoms refineable torsions = ",
                                            self.torsion_refineable_no_H.sum())
                self.H_atom_torsion_defs = True

            if self.bond_refineable.sum() > 0:
                print("Refineable bonds not yet supported")
            if self.angle_refineable.sum() > 0:
                print("Refineable angles not yet supported")

            self.get_degrees_of_freedom()

            self.initial_D2      = self.get_initial_D2_for_torch(
                        self.coords_radians, self.torsion_refineable_indices)
            self.initial_D2_no_H = self.get_initial_D2_for_torch(
                self.coords_radians_no_H, self.torsion_refineable_indices_no_H)

    def __repr__(self):
        file_info = "Filename: " + self.filename
        n_non_H_atoms = "\nNon-H atoms: " + str(len(self.elements_no_H))
        n_ref_torsions = "\nRefineable torsions: " + str(np.sum(
                                                    self.torsion_refineable))
        if self.degrees_of_freedom > 3:
            dof = "\nDegrees of freedom: "+str(self.degrees_of_freedom)+\
                            " (7 + "+str(np.sum(self.torsion_refineable))+")"
        else:
            dof = "\nDegrees of freedom: "+str(self.degrees_of_freedom)

        return file_info + n_non_H_atoms + n_ref_torsions + dof

    def from_json(self, attributes, is_json=True):
        """
        Load Z-matrix from a JSON formatted string

        Args:
            attribute_string (str): json string with all attributes of a ZM
        """
        if is_json:
            attributes = json.loads(attributes)
        for k, v in attributes.items():
            if v[1]:
                setattr(self, k, np.array(v[0]))
            else:
                setattr(self, k, v[0])

    def to_json(self, return_json=True):
        """
        Save the Z-matrix to a JSON formatted string

        Returns:
            str: JSON formatted string of the Z-matrix object attributes
        """
        dumpable = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                dumpable[k] = [v.tolist(), True]
            elif isinstance(v, np.int32):
                dumpable[k] = [int(v), False]
            else:
                dumpable[k] = [v, False]
        if return_json:
            return json.dumps(dumpable)
        else:
            return dumpable

    def get_degrees_of_freedom(self):
        if len(self.elements) == 1:
            self.degrees_of_freedom = 3
            self.external_degrees_of_freedom = 3
            self.position_degrees_of_freedom = 3
            self.rotation_degrees_of_freedom = 0
            self.internal_degrees_of_freedom = 0
        else:
            self.degrees_of_freedom = 7 + self.torsion_refineable.sum()
            self.external_degrees_of_freedom = 7
            self.position_degrees_of_freedom = 3
            self.rotation_degrees_of_freedom = 4
            self.internal_degrees_of_freedom = self.torsion_refineable.sum()

    #def update_refineable_indices(self):
    #    self.bond_refineable_indices = np.where(self.bond_refineable == 1)[0]
    #    self.angle_refineable_indices = np.where(self.angle_refineable == 1)[0]
    #    self.torsion_refineable_indices = np.where(
    #                                           self.torsion_refineable == 1)[0]
    #    self.remove_H_from_zm()
    #    self.coords_radians_no_H = self.zm_angles_to_radians(self.coords_no_H)

    def read_DASH_zm(self, input_filename):
        """
        Read in a DASH Z-matrix

        Args:
            input_filename (string): filename of ZM to read in. Expected to be
            in the current directory, or a properly formatted path.
        """
        element = []
        dw_factors = {}
        bond_length, bond_connection, bond_refineable = [], [], []
        angle, angle_connection, angle_refineable = [], [], []
        torsion, torsion_connection, torsion_refineable = [], [], []
        with open(input_filename, "r") as in_zm:
            i = 0
            for line in in_zm:
                line = list(filter(None, line.strip().split(" ")))
                if i > 2:
                    #if i == 2:
                    #    natoms = int(line[0])
                    #else:
                    element.append(line[0])
                    bond_length.append(line[1])
                    bond_refineable.append(line[2])
                    bond_connection.append(line[7])
                    angle.append(line[3])
                    angle_refineable.append(line[4])
                    angle_connection.append(line[8])
                    torsion.append(line[5])
                    torsion_refineable.append(line[6])
                    torsion_connection.append(line[9])
                    if element[-1] not in dw_factors:
                        dw_factors[element[-1]] = float(line[10])
                i+=1
        in_zm.close()
        bond_length = np.array(bond_length)
        self.bond_connection = np.array(bond_connection).astype(int)
        self.bond_refineable = np.array(bond_refineable).astype(int)

        angle = np.array(angle)
        self.angle_connection = np.array(angle_connection).astype(int)
        self.angle_refineable = np.array(angle_refineable).astype(int)

        torsion = np.array(torsion)
        self.torsion_connection = np.array(torsion_connection).astype(int)
        self.torsion_refineable = np.array(torsion_refineable).astype(int)

        self.coords = np.array([bond_length, angle, torsion]).astype(float).T
        self.elements = element
        self.dw_factors = dw_factors
        self.bond_refineable_indices = np.where(self.bond_refineable == 1)[0]
        self.angle_refineable_indices = np.where(self.angle_refineable == 1)[0]
        self.torsion_refineable_indices = np.where(
                                                self.torsion_refineable == 1)[0]

    def read_Gaussian_zm(self, filename):
        """
        This might work, it might not. It's been tested *very* briefly on how it
        handles rigid bodies, which appears to work fine. However, the Gaussian
        format has no flags to determine which torsion angles are refineable.
        In addition, there may be further issues with the definition of torsions
        where if one of the angles is set to refine, some of the attached atoms
        will rotate and some won't.
        A Z-matrix constructed with the MakeZmatrix.exe program that is bundled
        with DASH is a *much* better option if it is available.
        """
        variables_dict = {}
        zmat = []
        variables = False
        coords = False
        with open(filename, "r") as infile:
            i = 0
            for line in infile:
                line = list(filter(None, line.strip().split(" ")))
                if len(line) > 0:
                    if len(line) == 1:
                        coords = True
                    if variables:
                        if len(line) > 0:
                            variables_dict[line[0].split("=")[0]]=float(line[1])
                    else:
                        if line[0] == "Variables:":
                            variables = True
                        else:
                            if coords:
                                if len(line) == 7:
                                    zmat.append(line)
                                # Replace the zeros that aren't needed in
                                # internal coordinates to bring in line with
                                # DASH format
                                else:
                                    temp = []
                                    for j in range(7-len(line)):
                                        if j%2 == 0:
                                            temp.append("0")
                                        else:
                                            temp.append("0.000")
                                    zmat.append(line+temp)
                i += 1
        # Replace the labels in the zmatrix with the variables
        for i, line in enumerate(zmat):
            for j, item in enumerate(line):
                if item in variables_dict.keys():
                    zmat[i][j] = variables_dict[item]

        self.elements = [row[0] for row in zmat]
        self.bond_connection    = np.array([int(row[1]) for row in zmat])
        self.angle_connection   = np.array([int(row[3]) for row in zmat])
        self.torsion_connection = np.array([int(row[5]) for row in zmat])

        bond    = [float(row[2]) for row in zmat]
        angle   = [float(row[4]) for row in zmat]
        torsion = [float(row[6]) for row in zmat]
        self.coords = np.vstack([bond, angle, torsion]).T

        self.bond_refineable = np.zeros_like(self.bond_connection)
        self.angle_refineable = np.zeros_like(self.angle_connection)
        self.torsion_refineable = np.zeros_like(self.torsion_connection)
        self.dw_factors = {}

        self.bond_refineable_indices = np.where(self.bond_refineable == 1)[0]
        self.angle_refineable_indices = np.where(self.angle_refineable == 1)[0]
        self.torsion_refineable_indices = np.where(
                                                self.torsion_refineable == 1)[0]

    def remove_H_from_zm(self):
        """
        Remove hydrogen atoms from the Z-matrix.
        """
        coords_no_H, bond_connection_no_H = [], []
        angle_connection_no_H, torsion_connection_no_H = [], []
        bond_refineable_no_H, angle_refineable_no_H = [], []
        torsion_refineable_no_H, elements_no_H = [], []
        old_vs_new_index = []
        n_H_connected = np.zeros_like(self.bond_connection)
        i = 1
        j = 0
        for x in zip(self.elements, self.coords, self.bond_refineable,
                            self.angle_refineable, self.torsion_refineable,
                            self.bond_connection, self.angle_connection,
                                                self.torsion_connection):
            if x[0] != "H":
                old_vs_new_index.append([i, i-j])
                elements_no_H.append(x[0])
                coords_no_H.append(x[1])
                bond_refineable_no_H.append(x[2])
                angle_refineable_no_H.append(x[3])
                torsion_refineable_no_H.append(x[4])
                bond_connection_no_H.append(x[5])
                angle_connection_no_H.append(x[6])
                torsion_connection_no_H.append(x[7])
            if x[0] == "H":
                n_H_connected[x[5]] += 1
                j+=1
            i+=1
        old_vs_new_index = np.vstack(old_vs_new_index)
        coords_no_H = np.array(coords_no_H)
        bond_connection_no_H    = np.array(bond_connection_no_H).astype(int)
        angle_connection_no_H   = np.array(angle_connection_no_H).astype(int)
        torsion_connection_no_H = np.array(torsion_connection_no_H).astype(int)
        bond_refineable_no_H    = np.array(bond_refineable_no_H).astype(int)
        angle_refineable_no_H   = np.array(angle_refineable_no_H).astype(int)
        torsion_refineable_no_H = np.array(torsion_refineable_no_H).astype(int)
        n_H_connected = n_H_connected[np.array(self.elements) != "H"]

        for x in old_vs_new_index:
            bond_connection_no_H[bond_connection_no_H == x[0]] = x[1]
            angle_connection_no_H[angle_connection_no_H == x[0]] = x[1]
            torsion_connection_no_H[torsion_connection_no_H == x[0]] = x[1]


        self.coords_no_H = coords_no_H
        self.bond_connection_no_H    = bond_connection_no_H
        self.angle_connection_no_H   = angle_connection_no_H
        self.torsion_connection_no_H = torsion_connection_no_H
        self.bond_refineable_no_H    = bond_refineable_no_H
        self.angle_refineable_no_H   = angle_refineable_no_H
        self.torsion_refineable_no_H = torsion_refineable_no_H
        self.elements_no_H = elements_no_H
        self.n_H_connected = n_H_connected

        self.bond_refineable_indices_no_H = np.where(
                                        self.bond_refineable_no_H == 1)[0]
        self.angle_refineable_indices_no_H = np.where(
                                        self.angle_refineable_no_H == 1)[0]
        self.torsion_refineable_indices_no_H = np.where(
                                        self.torsion_refineable_no_H == 1)[0]


    def zm_angles_to_radians(self, zm):
        """
        Convert angles from degrees to radians

        Args:
            zm (numpy array): the Z-matrix coordinates array, shape = n_atoms, 3
                where the second and third columns are bond angles and torsions
                in degrees.

        Returns:
            numpy array: Z-matrix coordinates array, shape = n_atoms, 3
                where the second and third columns are bond angles and torsions
                in radians.
        """
        zm_radians = np.copy(zm)
        zm_radians[:,1] = np.deg2rad(zm_radians[:,1])
        zm_radians[:,2] = np.deg2rad(zm_radians[:,2])
        return zm_radians


    def zm_to_cart(self,zm,bond_connection,angle_connection,torsion_connection):
        """
        Uses the Natural Extension Reference Frame method to convert from
        Internal to Cartesian coordinates.
        Paper here: https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.20237
        """

        def cross_product(a, b):
            """
            Performs the cross-product of two three-dimensional vectors.
            """
            return np.array([a[1]*b[2] - a[2]*b[1],
                            a[2]*b[0] - a[0]*b[2],
                            a[0]*b[1] - a[1]*b[0]])

        def get_D2(zm):
            """
            Produces the D2 matrix described in the NERF paper linked above.
            """
            R = zm[:,0]
            c_t = np.cos(zm[:,1])
            s_t = np.sin(zm[:,1])
            c_p = np.cos(zm[:,2])
            s_p = np.sin(zm[:,2])
            D2 = np.empty((zm.shape[0], 3))
            for i in range(D2.shape[0]):
                D2[i] = R[i] * np.array([-c_t[i], c_p[i]*s_t[i], s_p[i]*s_t[i]])
            return D2

        def normalize(vector):
            """
            Normalize a vector
            """
            return vector / np.sqrt(np.dot(vector, vector))

        D2 = get_D2(zm)
        cart = np.zeros_like(D2)
        cart[0:3] = D2[0:3]
        for i in range(3, zm.shape[0]):
            C = cart[bond_connection[i]]
            B = cart[angle_connection[i]]
            A = cart[torsion_connection[i]]
            bc = normalize(C - B)
            AB = B - A
            n = normalize(cross_product(AB, bc))
            n_bc = cross_product(n, bc)
            M = np.vstack((bc, n_bc, n))
            cart[i] = np.dot(M.T, D2[i]) + C
        return cart

    def get_initial_D2_for_torch(self, zm, ref):
        """
        Used by PyTorch based SDPD for converting internal to external coords.
        The idea is that the NeRF method of internal -> Cartesian uses a D2
        matrix. The elements of this matrix can be modified to accommodate
        changes in the molecule such as torsion angles, bond angles or bond
        lengths.

        Currently, only the torsion angles are refineable, and therefore the
        rest of the D2 matrix is static. All that needs to be done is to
        generate the matrix as normal (see paper for details) and then multiply
        the relevant elements by the torsion angles.

        To accomplish this, generate the D2 matrix, then divide the relevant
        elements by the existing refineable torsion angle values.

        Args:
            zm (numpy array): the molecular z-matrix
            ref (numpy array): The indices of the refineable torsion angles

        Returns:
            numpy array: D2 matrix, with the elements that rely on the
                        refineable torsion angles divided by the initial torsion
                        angles, to speed up internal -> Cartesian conversion.
        """

        R = zm[:,0] # lengths
        c_t = np.cos(zm[:,1]) # angles
        s_t = np.sin(zm[:,1])
        c_p = np.cos(zm[:,2]) # torsions
        s_p = np.sin(zm[:,2])
        D2 = np.empty((zm.shape[0], 3))
        for i in range(D2.shape[0]):
            D2[i] = R[i] * np.array([-c_t[i], c_p[i]*s_t[i], s_p[i]*s_t[i]])
        D2[:,1][ref] = D2[:,1][ref] / c_p[ref]
        D2[:,2][ref] = D2[:,2][ref] / s_p[ref]
        return D2
