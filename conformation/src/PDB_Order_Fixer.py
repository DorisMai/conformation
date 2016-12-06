'''
Many PDB Files have an issue where a single chemical residue (e.g., a single histidine) will have each of its atoms, while
	chemically bonded, spread out across the PDB file. This confuses many PDB readers, as it will assume each non-contiguous
	line for that residue corresponds to separate residues. This class has a function that takes as inputs the name of the original
	pdb file and the name of a new file to which the user would like to write. It then finds all residues corresponding to proteins,
	and stores them in a dict, with keys as a tuple of (residue_name, residue_chain, and residue_number) and with values as a 
	list of physical lines in the PDB file that correspond to that unique residue identifier. The function then prints the initial header,
	prints the exact same PDB file lines just in a different order that maintains contiguousness within each residue, and then prints
	all membrane, solvent, and other lines.


'''
class PDB_Order_Fixer:

	def __init__(self, filename, new_filename):
		self.filename= filename
		self.new_filename = new_filename

	def fix_pdb(self):
		filename = self.filename
		new_filename = self.new_filename

		residue_lines = {}
		residue_order = []
		non_protein_lines = []
		conect_records = []

		pdb_file = open(filename, "rb")
		lines = pdb_file.readlines()
		i = 0

		while lines[i][0:4] != "ATOM" and lines[i][0:6] != "HETATM":
			residue_key = (i, 0, 0)
			residue_order.append(residue_key)
			residue_lines[residue_key] = [i]
			i += 1

		#print("Found first atom line, line %d" %i)

		while i < (len(lines) - 1):
			line = lines[i].split()

			if line[0] == "TER":
				i += 1
				continue

			if len(line) < 2:
				non_protein_lines.append(i)
				i += 1
				continue

			if line[0] == "CONECT" or line[0] == "SSBOND":
				conect_records.append(i)
				i += 1
				continue

			res_name = line[3]

			if res_name[0:3] in ["POP", "TIP", "HOH", "MEM"]:
				non_protein_lines.append(i)
				i += 1
				continue

			chain = line[4]
			chain = 'A'
			
			lines[i] = lines[i][:21] + 'A' + lines[i][22:]

			res_number = line[5]
			residue_key = (res_name, chain, res_number)


			if residue_key in list(residue_lines.keys()):
				residue_lines[residue_key].append(i)
			else:
				residue_lines[residue_key] = [i]
				residue_order.append(residue_key)

			i += 1

		#print("read a total of %d lines" %i)
		#print("finished reading file")

		new_file = open(new_filename, "wb")

		for residue_key in residue_order:
			for line_number in residue_lines[residue_key]:
				new_file.write(lines[line_number])

		for line_number in non_protein_lines:
			new_file.write(lines[line_number])

		for line_number in conect_records:
			new_file.write(lines[line_number])

		#print("finished writing new file")

		new_file.close()

